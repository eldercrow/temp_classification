# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

# from sc.config import cfg
# from sc.models.loss import structure_tensor_loss, l1_loss, bce_loss
from sc.models.backbone import get_backbone


class ModelBuilder(nn.Module):
    def __init__(self, cfg):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.NAME,
                                     **cfg.BACKBONE.KWARGS)
        # build output head
        self.head = nn.Linear(1000, cfg.NUM_CLASSES, bias=True)

    def classify(self, x):
        # get feature
        feat = self.backbone(x)
        head = self.head(feat)
        return F.softmax(head, dim=-1)

    def forward(self, data):
        """ only used in training
        """
        images, labels = data['image'], data['label']

        # get feature
        feat = self.backbone(images)
        head = self.head(feat)

        loss = F.cross_entropy(head, labels)

        outputs = {}
        outputs['loss'] = loss
        return outputs
