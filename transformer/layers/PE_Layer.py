import torch
import torch.nn as nn
import numpy as np
from .Constant import Constant

class Position_Embedding(nn.Module):
    def __init__(self, d_model, num_token = 200):
        super().__init__()

        self.d_model = d_model
        self.num_token = num_token
        self.register_buffer('pos_table',self._get_pos_encoding_table(num_token, d_model))
    
    def _get_pos_encoding_table(self, num_token, d_model):

        def get_position_angle_vec(position):
            
            return [position / np.power( Constant, 2 * (emb_i // 2) / d_model ) for emb_i in range(d_model)]

        posemb_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_token)])

        posemb_table[:,0::2] = np.sin(posemb_table[:,0::2])
        posemb_table[:,1::2] = np.cos(posemb_table[:,1::2])

        return torch.from_numpy(posemb_table).float().unsqueeze(0)

    def forward(self,x):
        return x + self.pos_table[:,:x.size(1)]

    
        
