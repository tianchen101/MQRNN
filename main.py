import numpy as np
import pandas as pd 
import torch
from Encoder import Encoder
from Decoder import GlobalDecoder,LocalDecoder
from utils import MQRNN_dataset,read_df,train_fn


config = {
    'horizon_size':40,
    'hidden_size':50,
    'quantiles': ['30', '50', '90'],
    'input_df_name':'LD2011_2014.txt',
    'columns':['MT_001', 'MT_002', 'MT_003',
               'MT_004', 'MT_005','MT_006',
               'MT_007', 'MT_008', 'MT_009'],
    'sep': ';',
    'index_col': 0,
    'parse_dates': True,
    'decimal': ',',
    'dropout': 0.3,
    'layer_size':2,
    'by_direction':True,
    'lr': 1e-4,
    'batch_size': 3,
    'num_epoch': 32
}


if __name__ == '__main__':
    train_target_df, test_target_df, train_covariate_df, test_covariate_df, scale_dict = read_df(config)
    quantiles = config['quantiles']
    quantile_size = len(quantiles)
    horizon_size = config['horizon_size']
    hidden_size = config['hidden_size']
    dropout = config['dropout']
    layer_size = config['layer_size']
    by_direction = config['by_direction']
    covariate_size = train_covariate_df.shape[1]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = MQRNN_dataset(train_target_df,train_covariate_df, horizon_size,quantile_size)

    encoder = Encoder(horizon_size= horizon_size, covariate_size=train_covariate_df.shape[1],hidden_size=hidden_size,
                      dropout=dropout, layer_size=layer_size, by_direction= by_direction,device=device)
    gdecoder = GlobalDecoder(hidden_size=hidden_size, covariate_size=covariate_size, horizon_size=horizon_size)
    ldecoder = LocalDecoder(covariate_size=covariate_size, quantile_size=quantile_size)

    lr = config['lr']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']

    train_fn(encoder=encoder,gdecoder=gdecoder, ldecoder=ldecoder, dataset=train_dataset, 
             num_epochs=num_epochs, batch_size=batch_size, device=device)
    
    
    

    