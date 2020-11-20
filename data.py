import pandas as pd 
import numpy as np


def read_df(config:dict):
    """
    This function is for reading the sample testing dataframe
    """
    time_range = pd.date_range('2010-01-01','2020-12-01',freq='12h')
    time_len = len(time_range)
    series_dict = {}
    for i in range(5):
        cur_vals = [np.sin(i*t) for t in range(time_len)]
        series_dict[i] = cur_vals
    target_df = pd.DataFrame(index=time_range, 
                             data=series_dict)
    horizon_size = config['horizon_size']
    covariate_df = pd.DataFrame(index=target_df.index,
                                data={'hour':target_df.index.hour,
                                      'dayofweek':target_df.index.dayofweek,
                                      'month': target_df.index.month
                                })
    for col in covariate_df.columns:
        covariate_df[col] = (covariate_df[col] - np.mean(covariate_df))/np.std(covariate_df[col])
    train_target_df = target_df.iloc[:-horizon_size,:]
    test_target_df = target_df.iloc[-horizon_size:,:]
    train_covariate_df = covariate_df.iloc[:-horizon_size,:]
    test_covariate_df = covariate_df.iloc[-horizon_size:,:]
    return train_target_df, test_target_df, train_covariate_df, test_covariate_df


class MQRNN_dataset(Dataset):
    
    def __init__(self,
                series_df:pd.DataFrame,
                covariate_df:pd.DataFrame, 
                horizon_size:int,
                quantile_size:int):
        
        self.series_df = series_df
        self.covaraite_df = covariate_df
        self.horizon_size = horizon_size
        self.quantile_size = quantile_size
        full_covariate = []
        print(f"self.covariate_df.shape[0] : {self.covariate_df.shape[0]}")
        for i in range(1, self.covariate_df.shape[0] - horizon_size+1):
            cur_covariate = []
            #for j in range(horizon_size):
            cur_covariate.append(self.covariate_df.iloc[i:i+horizon_size,:].to_numpy())
            full_covariate.append(cur_covariate)
        full_covariate = np.array(full_covariate)
        print(f"full_covariate shape: {full_covariate.shape}")
        full_covariate = full_covariate.reshape(-1, horizon_size * covariate_size)
        self.next_covariate = full_covariate
    
    def __len__(self):
        return self.series_df.shape[1]
    
    def __getitem__(self,idx):
        cur_series = np.array(self.series_df.iloc[: -self.horizon_size, idx])
        cur_covariate = np.array(self.covariate_df.iloc[:-self.horizon_size, :]) # covariate used in generating hidden states

        covariate_size = self.covariate_df.shape[1]
        #next_covariate = np.array(self.covariate_df.iloc[1:-self.horizon_size+1,:]) # covariate used in the MLP decoders

        real_vals_list = []
        for i in range(1, self.horizon_size+1):
            real_vals_list.append(np.array(self.series_df.iloc[i: self.series_df.shape[0]-self.horizon_size+i, idx]))
        real_vals_array = np.array(real_vals_list) #[horizon_size, seq_len]
        real_vals_array = real_vals_array.T #[seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)
        
        cur_series_tensor = torch.unsqueeze(cur_series_tensor,dim=1) # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate) #[seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor],dim=1)
        next_covariate_tensor = torch.tensor(self.next_covariate) #[seq_len, horizon_size * covariate_size]

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor


