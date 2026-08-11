import torch
import torch.nn as nn



def safesoftmax(x,dim = -1): # safesoftmax 用于替代传统softmax
    x_max = torch.max(x,dim=dim,keepdim=True)[0]
    x_exp = torch.exp(x-x_max)
    x_safesoftmax = x_exp / torch.sum(x_exp,dim = dim, keepdim = True)
    
    return x_safesoftmax


class RMS_Layernorm(nn.Module):

    ''' RMS_Layernorm 不去中心化, 增加可学习逐feature放缩参数gamma'''

    def __init__(self, eps, d):
        super().__init__()
        self.d = d
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self,x):
        x_sqsum = torch.sqrt(torch.sum(x.pow(2), dim = -1, keepdim = True)/self.d + self.eps)
        x = x/x_sqsum
        return x*self.gamma
        

        