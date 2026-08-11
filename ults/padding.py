import torch

def get_pad_mask(seq,pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):

    batch_size, len_seq = seq.size()

    subsequent_mask =(1 - torch.triu(
        torch.ones((1, len_seq, len_seq), device = seq.device), diagonal = 1).bool()
    )

    return subsequent_mask.bool()