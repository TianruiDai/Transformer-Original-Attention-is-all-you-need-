import torch
import torch.nn as nn
import torch.nn.functional as F
from .ult import RMS_Layernorm

class Two_Layers(nn.Module):

    ''' 两层FFN 用于扩大视野，增强表示'''
    
    def __init__(self,d_in,d_hid,dropout = 0.1):
        super().__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.dropout = nn.Dropout(p = dropout)
        self.layer1 = nn.Linear(d_in,d_hid)
        self.layer2 = nn.Linear(d_hid,d_in)
        self.layernorm = self.layer_norm = RMS_Layernorm(eps = 1e-5, d = d_in)
    
    def forward(self,x):
        res = x
        x = self.layer2(F.relu(self.layer1(x)))

        x = x + res

        x = self.layernorm(x)

        return x
        



