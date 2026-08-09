import torch.nn as nn
import torch
from transformer.Multiheadblock import MultiHeadAttention, TwoLayersFeedForward, ScaledDotProductAttention

class EncoderLayer(nn.Module):

    def __init__(self,d_model,d_inner,n_head,d_k,d_v,dropout = 0.1):
        super(EncoderLayer,self).__init__()
        self.attn = MultiHeadAttention(n_head,d_model,d_k,d_v,dropout = dropout)
        self.ffn = TwoLayersFeedForward(d_model,d_inner,dropout = dropout)

    def forward(self, enc_input, attn_mask = None):
        enc_output,enc_attn = self.attn(
            enc_input,enc_input,enc_input,mask = attn_mask
        )

        enc_output = self.ffn(enc_output)

        return enc_output, enc_attn

class DecoderLayer(nn.Module):

    def __init__(self,d_model,d_inner,n_head,d_k,d_v,dropout = 0.1):
        super(DecoderLayer,self).__init__()
        self.attn = MultiHeadAttention(n_head,d_model,d_k,d_v,dropout = dropout)
        self.enc_attn = MultiHeadAttention(n_head,d_model,d_k,d_v,dropout = dropout)
        self.ffn = TwoLayersFeedForward(d_model,d_inner,dropout = dropout)

    def forward(
        self, dec_input, enc_output,
        attn_mask = None, dec_enc_attn_mask = None
    ):
        dec_output, dec_attn = self.attn(
            dec_input,dec_input,dec_input,mask = attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output,enc_output,enc_output,mask = dec_enc_attn_mask
        )
        dec_output = self.ffn(dec_output)

        return dec_output, dec_attn, dec_enc_attn