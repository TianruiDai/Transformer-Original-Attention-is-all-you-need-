import torch.nn as nn
import torch
from .ult import safesoftmax

    

class Scaled_Dot_Product_Attention(nn.Module):
    
    '''使用现代safesoftmax的带scale缩放的attention点乘算子'''

    def __init__(self, scaling, attn_dropout = 0.1):
        super().__init__()

        self.scaling = scaling
        self.attn_dropout = nn.Dropout(p = attn_dropout)
    
    def forward(self,q,k,v,mask=None):

        attn = torch.matmul(q / self.scaling, k.transpose(2,3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.attn_dropout(safesoftmax(attn,dim=-1))
        output = attn @ v

        return output, attn

