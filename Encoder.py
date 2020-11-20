import torch
import torch.nn as nn
import pandas as pd 
import numpy as np

class Encoder(nn.Module):
    """
    Encoder of the Encoder-Decoder Structure. For MQ_RNN, this encoder
    is the same as the traditional seq2seq model, which is based on the LSTM.

    """
    def __init__(self, 
                horison_size:int, 
                covariate_size:int, 
                hidden_size:int, 
                dropout:float, 
                layer_size:int, 
                by_direction:bool,
                device):
        super(Encoder, self).__init__()
        self.horizon_size =horison_size
        self.covariate_size = covariate_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.by_direction = by_direction
        self.dropout = dropout
        self.LSTM = nn.LSTM(input_size= covariate_size+1, 
                            hidden_size=hidden_size, 
                            num_layers=layer_size, 
                            dropout=dropout,
                            bidirectional=by_direction)
        for param in self.LSTM.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    
    def forward(self, input):
        """
        For the RNN(LSTM), the input shape is [seq_len,batch_size,input_size]
        where input_size = covariate_size +1
        """
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        input_size = input.shape[2]
        layer_size = self.layer_size
        direction_size =1
        if self.by_direction:
            direction_size = 2
        outputs,_ = self.LSTM(input)
        outputs_reshape = outputs.view(seq_len,batch_size,direction_size,self.hidden_size)
        outputs_last_layer = outputs_reshape[:,:,-1,:]
        final_outputs = outputs_last_layer.view(seq_len,batch_size,self.hidden_size)
        return final_outputs