import numpy as np
from torch import dropout
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class ScaledDotProductAttention(nn.Module):

    def __init__(self,temperature,attn_dropout = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self,q,k,v,mask=None):

        ''' 
        The size of Q and K are (Batch_size,num_head,num_token,d_k)
        First calculate QK^transport along dimension 2 and 3, we get the attn size (Batch_size,num_head,num_token,attention with other tokens = num_token)
        If there is a mask then masked_fill the mask part
        Then, just softmax (calculate the probability along the final dimension)

        '''

        attn = torch.matmul(q / self.temperature, k.transpose(2,3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn,dim=-1))
        output = torch.matmul(attn,v)

        return output,attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d, d_k, d_v, dropout = 0.1):
        
        super(MultiHeadAttention,self).__init__()
        
        self.head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d = d

        self.w_q = nn.Linear(d, n_head * d_k, bias = False)
        self.w_k = nn.Linear(d, n_head * d_k, bias = False)
        self.w_v = nn.Linear(d, n_head * d_v, bias = False)
        self.fc = nn.Linear(n_head * d_v, d ,bias=False)

        self.attention = ScaledDotProductAttention(temperature = math.sqrt(d_k))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d, eps = 1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.head

        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_q(q).view(batch_size,len_q,n_head,d_k)
        k = self.w_k(k).view(batch_size,len_k,n_head,d_k)
        v = self.w_v(v).view(batch_size,len_v,n_head,d_v)

        q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)


        q,attn = self.attention(q,k,v,mask=mask)
        q = q.transpose(1,2).contiguous().view(batch_size,len_q,-1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class TwoLayersFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) 
        self.w_2 = nn.Linear(d_hid, d_in) 
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x