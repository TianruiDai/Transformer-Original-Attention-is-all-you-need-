from ult import *
import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
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
        self.cross_attention = Multihead_Attention(
            num_head, model_d = d_model, d_k = d_k, d_v = d_v, dropout = dropout
            )
        self.twolayer = Two_Layers(d_model,d_inner,dropout = dropout)

    def forward(self,enc_output,dec_input,dec_enc_mask,dec_self_mask = None):
        dec_output,dec_self_attn = self.attention(dec_input,dec_input,dec_input,dec_self_mask)
        dec_output,dec_enc_attn = self.cross_attention(dec_input,enc_output,enc_output,dec_enc_mask )
        dec_output = self.twolayer(dec_output)

        return dec_output, dec_enc_attn, dec_self_attn