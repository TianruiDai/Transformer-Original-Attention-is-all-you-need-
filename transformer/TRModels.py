from math import sin
import torch
from torch.functional import _return_inverse
import torch.nn as nn
import numpy as np
from torch.nn.modules import padding
from transformer.EncDec import EncoderLayer,DecoderLayer
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_hid, n_position = 200):

        super(PositionEncoding,self).__init__()

        # 把一个 张量注册成模块的 buffer，而不是可训练参数。
        self.register_buffer('pos_table',self._get_pos_encoding_table(n_position,d_hid))

    def _get_pos_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):

            # get an array wich equals to pos/{(10000)^{2i/2}}
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
            
        sinencoding_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # the shape of sineconding_table is (n_position,d_hid)

        sinencoding_table[:, 0::2] = np.sin(sinencoding_table[:,0::2]) # sin operated on 2i position

        sinencoding_table[:, 1::2] = np.cos(sinencoding_table[:,0::2]) # cos operated on 2i+1 position

        return torch.FloatTensor(sinencoding_table).unsqueeze(0) # adding an extra dimension which represent the batch

    def forward(self,x):

        # the shape of x is (num_batch,num_token,feature_dimension)
        return x + self.pos_table[:,:x.size(1)].clone().detach()

class Encoder(nn.Module):
    
    def __init__(
        self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
        d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        # using embedding layer to embedded the vocab to the word vector with some special padding 
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx = pad_idx)
        
        self.position_enc = PositionEncoding(d_word_vec, n_position = n_position)

        self.dropout = nn.Dropout(p = dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model,d_inner,n_head,d_k,d_v,dropout = dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model,eps = 1e-6)

        self.scale_emb = scale_emb

        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns = False):

        enc_attn_list = []

        enc_output = self.src_word_emb(src_seq)

        if self.scale_emb:
            enc_output *= math.sqrt(self.d_model)

        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(enc_output, attn_mask = None )

            enc_attn_list += [enc_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_attn_list
        else:
            return enc_output,

class Decoder(nn.Module):

    def __init__(
        self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
        d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init_()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx = pad_idx)
        self.position_enc = PositionEncoding(d_word_vec, n_position = n_position)
        self.dropout = nn.Dropout(p = dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps = 1e-6)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model,d_inner,n_head,d_k,d_v,dropout=dropout)
            for _ in range(n_layers)
        ])
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
            
        dec_output = self.trg_word_emb(trg_seq)

        dec_attn_list = []
        dec_enc_attn_list = []

        if self.scale_emb:
             dec_output *= math.sqrt(self.d_model)
        
        dec_output = self.dropout(self.position_enc(dec_output))

        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:

            dec_output, dec_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, attn_mask = src_mask, dec_enc_mask = src_mask
            )

        dec_attn_list += [dec_attn] if return_attns else []
        dec_enc_attn_list += [dec_enc_attn] if return_attns else []


        if return_attns:
            return dec_output,dec_attn_list,dec_enc_attn_list
        else:
            return dec_output,

class Transformer(nn.Module):

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__

        self.src_pad_idx,self.trg_pad_idx = src_pad_idx,trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)


        

