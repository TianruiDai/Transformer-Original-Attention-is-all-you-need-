from ult import *
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_k,d_v,num_head,d_inner,dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_head = num_head
        self.d_inner = d_inner
        self.dropout =nn.Dropout( p = dropout)
        self.attention = Multihead_Attention(
            num_head, model_d = d_model, d_k = d_k, d_v = d_v, dropout = dropout
            )
        self.twolayer = Two_Layers(d_model,d_inner,dropout = dropout)

    def forward(self,enc_input,enc_mask = None):
        enc_output,enc_attn = self.attention(enc_input,enc_input,enc_input,enc_mask)
        enc_output = self.twolayer(enc_output)

        return enc_output, enc_attn

