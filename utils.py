import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from Encoder import Encoder
from Decoder import GlobalDecoder, LocalDecoder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def read_df(config:dict):
    """
    Reading in the dataframe, and split into training and testing.
    """

    file_name = config['input_df_name']
    sep = config['sep']
    index_col = config['index_col']
    parse_dates = config['parse_dates']
    decimal = config['decimal']

    horizon_size = config['horizon_size']
    total_df = pd.read_csv(file_name,sep=sep,index_col=index_col, parse_dates=parse_dates, decimal=decimal)
    full_cols = config['columns']
    target_df = total_df[full_cols]
    del total_df
    target_df = target_df.iloc[-5000:,:] # truncate because my local machine dont have much capacity
    scale_dict = {}
    for col in full_cols:
        scale_dict[col] = np.mean(target_df[col])
        target_df[col] = target_df[col]/scale_dict[col] # normalize each input colum
    covariate_df = pd.DataFrame(index=target_df.index, data={'hour': target_df.index.hour,
                                                             'dayofweek':target_df.index.dayofweek,
                                                             'month':target_df.index.month,
                                                             'minute':target_df.index.minute})
    for col in covariate_df.columns:
        covariate_df[col] = (covariate_df[col] - np.mean(covariate_df[col]))/ np.std(covariate_df[col])
    
    train_target_df = target_df.iloc[:-horizon_size,:]
    test_target_df = target_df.iloc[-horizon_size:,:]
    train_covariate_df = covariate_df.iloc[:-horizon_size,:]
    test_covariate_df = covariate_df.iloc[-horizon_size:,:]
    return train_target_df, test_target_df, train_covariate_df, test_covariate_df, scale_dict



class MQRNN_dataset(Dataset):

    def __init__(self, series_df:pd.DataFrame, covariate_df:pd.DataFrame, horizon_size:int,quantile_size:int):
        assert(series_df.shape[0] == covariate_df.shape[0])
        self.series_df = series_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size
        self.quantile_size = quantile_size

    def __len__(self):
        return self.series_df.shape[1]
    
    def __getitem__(self, idx):
        cur_series = np.array(self.series_df.iloc[: -self.horizon_size, idx])
        cur_covariate = np.array(self.covariate_df.iloc[:-self.horizon_size, :])
        real_vals_list = []
        for i in range(1, self.horizon_size+1):
            real_vals_list.append(np.array(self.covariate_df.iloc[i: self.series_df.shape[0]-self.horizon_size+i]))
        cur_series_tensor = torch.tensor(cur_series)
        cur_covariate_tensor = torch.tensor(cur_covariate)
        cur_real_vals_tensor = torch.tensor(real_vals_list)
        return cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor


def calc_loss(cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor, encoder,gdecoder,ldecoder,device):
    loss = torch.tensor([0.0], device=device)
    cur_series_tensor = cur_series_tensor.double() #[batch_size, seq_len, input]
    cur_covariate_tensor = cur_covariate_tensor.double() #[batch_size, seq_len, covariate_size]
    cur_real_vals_tensor = cur_real_vals_tensor.double() #[batch_size, seq_len, horizon_size]

    cur_series_tensor = cur_series_tensor.permute(1,0,2)
    cur_covariate_tensor = cur_covariate_tensor.permute(1,0,2)
    cur_real_vals_tensor = cur_real_vals_tensor.permute(1,0,2)

    enc_hs = encoder(cur_series_tensor) #[seq_len, batch_size, hidden_size]
    hidden_and_covariate = torch.cat([enc_hs,cur_covariate_tensor],dim=2) #[seq_len, batch_size,hidden_size+covariate_size]
    gdecoder_output = gdecoder(hidden_and_covariate) #[seq_len, batch_size, horizon_size+1]

    horizon_context_tensor = gdecoder_output[:,:,:-1] #[seq_len, batch_size, horizon_size]
    agnostic_tensor = gdecoder_output[:,:,-1] # [seq_len, batch_size, 1]
    #TODO


    



def train_fn(encoder, gdecoder, ldecoder,dataset, lr, batch_size, num_epochs, device):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=lr)
    gdecoder_optimizer = torch.optim.Adam(gdecoder.parameters(),lr=lr)
    ldecoder_optimizer = torch.optim.Adam(ldecoder.parameters(), lr=lr)

    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    l_sum = 0.0
    for i in range(num_epochs):
        #TODO
        pass


