import torch
import torch.nn
from Encoder import Encoder
from Decoder import GlobalDecoder,LocalDecoder
from train_func import train_fn
from data import MQRNN_dataset

class MQRNN(object):
    """
    This class holds the encoder and the global decoder and local decoder.
    """
    def __init__(self, 
                horizon_size:int, 
                hidden_size:int, 
                quantiles:list,
                columns:list, 
                dropout:float,
                layer_size:int,
                by_directoin:bool,
                lr:float,
                batch_size:int, 
                num_epochs:int, 
                context_size:int, 
                covariate_size:int,
                device):
        print(f"device is: {device}")
        self.device = device
        self.horizon_size = horizon_size
        self.quantile_size = len(quantiles)

        self.encoder = Encoder(horizon_size=horizon_size,
                               covariate_size=covariate_size,
                               hidden_size=hidden_size, 
                               dropout=dropout,
                               layer_size=layer_size,
                               by_direction=by_direction,
                               device=device)
        
        self.gdecoder = GlobalDecoder(hidden_size=hidden_size,
                                    covariate_size=covariate_size,
                                    horizon_size=horizon_size,
                                    context_size=context_size)
        self.ldecoder = LocalDecoder(covariate_size=covariate_size,
                                    quantile_size=quantile_size,
                                    context_size=context_size,
                                    quantiles=quantiles,
                                    horizon_size=horizon_size)
        self.encoder.double()
        self.gdecoder.double()
        self.ldecoder.double()
    
    def train(dataset:MQRNN_dataset):
        
        train_fn(encoder=self.encoder, 
                gdecoder=self.gdecoder, 
                ldecoder=self.ldecoder,
                dataset=dataset,
                lr=self.lr,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                device=self.device)
        print("training finished")
    
    def predict(train_target_df, train_covariate_df, test_covariate_df, col_name):

        input_target_tensor = torch.tensor(train_target_df[[col_name]].to_numpy())
        full_covariate = train_covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate)

        next_covariate = test_covariate_df.to_numpy()
        next_covariate = next_covariate.reshape(-1, self.horizon_size)
        next_covariate_tensor = torch.tensor(next_covariate) #[1,horizon_size * covariate_size]

        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)
        next_covariate_tensor = next_covariate_tensor.to(self.device)

        with torch.no_grad():
            input_target_covariate_tensor = torch.cat([input_target_tensor, input_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim= 0) #[1, seq_len, 1+covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2) #[seq_len, 1, 1+covariate_size]
            print(f"input_target_covariate_tensor shape: {input_target_covariate_tensor.shape}")
            outputs = encoder(input_target_covariate_tensor) #[seq_len,1,hidden_size]
            hidden = torch.unsqueeze(outputs[-1],dim=0) #[1,1,hidden_size]

            next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0)
            #next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0) # [1,1, covariate_size * horizon_size]

            print(f"hidden shape: {hidden.shape}")
            print(f"next_covariate_tensor: {next_covariate_tensor.shape}")
            gdecoder_input = torch.cat([hidden, next_covariate_tensor], dim=2) #[1,1, hidden + covariate_size* horizon_size]
            gdecoder_output = gdecoder( gdecoder_input) #[1,1,(horizon_size+1)*context_size]

            local_decoder_input = torch.cat([gdecoder_output, next_covariate_tensor], dim=2) #[1, 1,(horizon_size+1)*context_size + covariate_size * horizon_size]
            local_decoder_output = ldecoder( local_decoder_input) #[seq_len, batch_size, horizon_size* quantile_size]
            local_decoder_output = local_decoder_output.view(self.horizon_size,self.quantile_size)
            output_array = local_decoder_output.cpu().numpy()
            result_dict= {}
            for i in range(self.quantile_size):
                result_dict[self.quantiles[i]] = output_array[:,i]
            return result_dict