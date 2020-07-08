import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class TRHead(nn.Module):
    '''
    For now, this is just a linear classifier.
    '''
    def __init__(self, ich, nclass):
        '''
        och: number of classes
        '''
        super(TRHead, self).__init__()
        self.classifier = nn.Conv2d(ich, nclass, 1, bias=True)
        self.nclass = nclass

    def forward(self, x):
        '''
        '''
        out = self.classifier(x)
        return out.view(-1, self.nclass)