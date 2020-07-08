import torch
from torch import nn
from torch.nn import functional as F


class TRNeck(nn.Module):
    '''
    For now, we only apply global average pooling.
    '''
    def __init__(self):
        '''
        ich: in channel
        och: out channel for both necks
        kch, vch: key and value channels for APNB
        ach: SE block internal channel
        '''
        super(TRNeck, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        '''
        '''
        return self.pool(x)