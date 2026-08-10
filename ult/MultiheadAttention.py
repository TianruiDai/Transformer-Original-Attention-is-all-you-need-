import torch
import torch.nn as nn
from .ScaledDotProductAttention import Scaled_Dot_Product_Attention
from .ult import RMS_Layernorm
import math

class Multihead_Attention(nn.Module):

    '''多头注意力层'''
    
    def __init__(self, num_head, model_d, d_k, d_v, dropout = 0.1):
        super().__init__()

        self.num_head = num_head
        self.model_d = model_d
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(p = dropout)
        self.attention = Scaled_Dot_Product_Attention(scaling = math.sqrt(d_k))
        self.layer_norm = RMS_Layernorm(eps = 1e-5, d = model_d)
        self.w_q = nn.Linear(model_d, num_head*d_k, bias =False)
        self.w_k = nn.Linear(model_d, num_head*d_k, bias =False)
        self.w_v = nn.Linear(model_d, num_head*d_v, bias =False)
        self.fc = nn.Linear(num_head*d_v, model_d, bias = False)
    
    def forward(self, q, k, v, mask = None):

        d_k, d_v, num_head = self.d_k, self.d_v, self.num_head

        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        res = q

        q = self.w_q(q).view(batch_size,len_q,num_head,d_k)
        k = self.w_k(k).view(batch_size,len_k,num_head,d_k)
        v = self.w_v(v).view(batch_size,len_v,num_head,d_v)

        q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q,attn = self.attention(q,k,v,mask = mask)
        q = q.transpose(1,2).contiguous().view(batch_size,len_q,-1)
        q = self.dropout(self.fc(q))

        q = q + res

        q = self.layer_norm(q)

        return q, attn


