import torch
import torch.nn as nn
from .EncoderDecoder import Encoder, Decoder
from ults.padding import get_pad_mask, get_subsequent_mask

class Transformer(nn.Module):
    def __init__(
        self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
        d_word_vec = 512, d_model = 512, d_inner = 2048,
        n_layer = 6, n_head = 8, d_k = 64, d_v = 64, dropout = 0.1, n_token =200,
        trg_emb_prj_weight_sharing = True, emb_src_trg_weight_sharing = True,
        scale_emb_or_prj = 'prj'
    ):
        super().__init__()

        assert scale_emb_or_prj in ['emb','prj','none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab = n_src_vocab,
            d_word_vec = d_word_vec,
            n_layer = n_layer,
            n_head = n_head,
            d_k = d_k,
            d_v = d_v,
            d_model = d_model,
            d_inner = d_inner,
            pad_idx = src_pad_idx,
            dropout = dropout,
            n_token = n_token,
            scale_emb = scale_emb
        )

        self.decoder = Decoder(
            n_trg_vocab,
            d_word_vec,
            n_layer,
            n_head,
            d_k,d_v,d_model,d_inner,
            trg_pad_idx,
            dropout = dropout,
            n_token = n_token,
            scale_emb = scale_emb
        )

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias = False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec

        if trg_emb_prj_weight_sharing:

            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq,src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)

        if self.scale_prj:
            seq_logit *= self.d_model ** (-0.5)

        return seq_logit.view(-1,seq_logit.size(2))


            

