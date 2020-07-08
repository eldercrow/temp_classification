# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def _outer3(x):
    N = x.shape[0]
    return torch.matmul( \
        torch.reshape(x, (N, 3, 1)), \
        torch.reshape(x, (N, 1, 3))).view((N, 9))

def structure_tensor_loss(pred, label, weight=None):
    '''
    pred: (N, 3)
    label: (N, 3)
    '''
    pred = _outer3(pred)
    label = _outer3(label)
    diff = (pred - label).abs()
    loss = diff.sum(dim=1)
    if weight is not None:
        loss *= weight
    return loss.mean()

def l1_loss(pred, label, weight=None):
    '''
    '''
    N = pred.shape[0]
    diff = (torch.reshape(pred, (N, 3)) - torch.reshape(label, (N, 3))).abs()
    loss = diff.sum(dim=1)
    if weight is not None:
        loss *= weight
    return loss.mean()

def bce_loss(pred, label, weight=None):
    '''
    '''
    N, n_cls = pred.shape
    loss = F.cross_entropy(pred, label, reduction='none')
    if weight is not None:
        loss *= weight
    return loss.mean()