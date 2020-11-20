import torch
from .Encoder import Encoder
from .Decoder import GlobalDecoder, LocalDecoder
from .data import MQRNN_dataset
from torch.utils.data import DataLoader

def calc_loss(cur_series_covariate_tensor : torch.Tensor, 
            next_covariate_tensor: torch.Tensor,
            cur_real_vals_tensor: torch.Tensor, 
            encoder: Encoder,
            gdecoder: GlobalDecoder,
            ldecoder: LocalDecoder,
            device):
    loss = torch.tensor([0.0], device=device)

    cur_series_covariate_tensor = cur_series_covariate_tensor.double() #[batch_size, seq_len, 1+covariate_size]
    next_covariate_tensor = next_covariate_tensor.double() # [batch_size, seq_len, covariate_size * horizon_size]
    cur_real_vals_tensor = cur_real_vals_tensor.double() # [batch_size, seq_len, horizon_size]

    cur_series_covariate_tensor = cur_series_covariate_tensor.to(device)
    next_covariate_tensor = next_covariate_tensor.to(device)
    cur_real_vals_tensor = cur_real_vals_tensor.to(device)
    encoder.to(device)
    gdecoder.to(device)
    ldecoder.to(device)

    cur_series_covariate_tensor = cur_series_covariate_tensor.permute(1,0,2) #[seq_len, batch_size, 1+covariate_size]
    next_covariate_tensor = next_covariate_tensor.permute(1,0,2) #[seq_len, batch_size, covariate_size * horizon_size]
    cur_real_vals_tensor = cur_real_vals_tensor.permute(1,0,2) #[seq_len, batch_size, horizon_size]
    enc_hs = encoder(cur_series_covariate_tensor) #[seq_len, batch_size, hidden_size]
    hidden_and_covariate = torch.cat([enc_hs, next_covariate_tensor], dim=2) #[seq_len, batch_size, hidden_size+covariate_size * horizon_size]
    gdecoder_output = gdecoder(hidden_and_covariate) #[seq_len, batch_size, (horizon_size+1)*context_size]

    context_size = ldecoder.context_size
    
    quantile_size = ldecoder.quantile_size
    horizon_size = encoder.horizon_size
    total_loss = torch.tensor([0.0],device=device)

    local_decoder_input = torch.cat([gdecoder_output, next_covariate_tensor], dim=2) #[seq_len, batch_size,(horizon_size+1)*context_size + covariate_size * horizon_size]
    local_decoder_output = ldecoder( local_decoder_input) #[seq_len, batch_size, horizon_size* quantile_size]
    seq_len = local_decoder_output.shape[0]
    batch_size = local_decoder_output.shape[1]
    
    local_decoder_output = local_decoder_output.view(seq_len, batch_size, horizon_size, quantile_size) #[[seq_len, batch_size, horizon_size, quantile_size]]
    for i in range(quantile_size):
      p = ldecoder.quantiles[i]
      errors = cur_real_vals_tensor - local_decoder_output[:,:,:,i]
      cur_loss = torch.max( (p-1)*errors, p*errors ) # CAUTION
      total_loss += torch.sum(cur_loss)
    return total_loss


def train_fn(encoder:Encoder, 
            gdecoder: GlobalDecoder, 
            ldecoder: LocalDecoder,
            dataset: MQRNN_dataset, 
            lr: float, 
            batch_size: int,
            num_epochs: int, 
            device):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=lr)
    gdecoder_optimizer = torch.optim.Adam(gdecoder.parameters(),lr=lr)
    ldecoder_optimizer = torch.optim.Adam(ldecoder.parameters(), lr=lr)

    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    l_sum = 0.0
    for i in range(num_epochs):
        #print(f"epoch_num:{i}")
        epoch_loss_sum = 0.0
        total_sample = 0
        for (cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor) in data_iter:
            batch_size = cur_series_tensor.shape[0]
            seq_len = cur_series_tensor.shape[1]
            horizon_size = cur_covariate_tensor.shape[-1]
            total_sample += batch_size * seq_len * horizon_size
            encoder_optimizer.zero_grad()
            gdecoder_optimizer.zero_grad()
            ldecoder_optimizer.zero_grad()
            loss = calc_loss(cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor, 
                             encoder, gdecoder, ldecoder,device)
            loss.backward()
            encoder_optimizer.step()
            gdecoder_optimizer.step()
            ldecoder_optimizer.step()
            epoch_loss_sum += loss.item()
        epoch_loss_mean = epoch_loss_sum/ total_sample
        if (i+1)%5 == 0:
            print(f"epoch_num {i+1}, current loss is: {epoch_loss_mean}")