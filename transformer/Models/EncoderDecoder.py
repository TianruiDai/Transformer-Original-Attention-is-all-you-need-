import torch
import torch.nn as nn
from layers import *
from ults import *
import math

class Encoder(nn.Module):
    def __init__(
        self, n_src_vocab, d_word_vec, n_layer, n_head, d_k, d_v,
        d_model, d_inner,
        pad_idx, dropout =0.1, n_token = 200, scale_emb  = False):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx = pad_idx)
        self.position_emb = Position_Embedding(d_word_vec, num_token = n_token)
        self.dropout = nn.Dropout( p = dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_k, d_v, n_head, d_inner, dropout = dropout)
            for _ in range(n_layer)
        ])
        self.layer_norm = RMS_Layernorm(eps = 1e-6, d = d_model)

        self.scale_emb = scale_emb

        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns = False):
        enc_attn_list = []

        enc_output = self.src_word_emb(src_seq)

        if self.scale_emb:
            enc_output *= math.sqrt(self.d_model)

        enc_output = self.dropout(self.position_emb(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(enc_output, enc_mask = None)
            enc_attn_list += [enc_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_attn_list
        else:
            return enc_output,


class Decoder(nn.Module):
    def __init__(
        self, n_trg_vocab, d_word_vec, n_layer, n_head, d_k, d_v,
        d_model, d_inner,
        pad_idx, dropout =0.1, n_token = 200, scale_emb  = False):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx = pad_idx)
        self.position_emb = Position_Embedding(d_word_vec, num_token = n_token)
        self.dropout = nn.Dropout( p = dropout)
        self.layer_stack = nn.ModuleList([
           DecoderLayer(d_model, d_k, d_v, n_head, d_inner, dropout = dropout)
            for _ in range(n_layer)
        ])
        self.layer_norm = RMS_Layernorm(eps = 1e-6, d = d_model)

        self.scale_emb = scale_emb

        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns = False):

        dec_attn_list = []
        dec_enc_attn_list = []


        dec_output = self.trg_word_emb(trg_seq)

        if self.scale_emb:
            dec_output *= math.sqrt(self.d_model)

        dec_output = self.dropout(self.position_emb(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_attn, dec_enc_attn = dec_layer(
                enc_output, dec_output, src_mask, trg_mask)

            dec_attn_list += [dec_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn ] if return_attns else []

        if return_attns:
            return dec_output, dec_attn_list, dec_enc_attn_list
        else:
            return dec_output,      
    
