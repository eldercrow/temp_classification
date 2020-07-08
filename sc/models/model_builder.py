# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from sc.config import cfg
# from sc.models.loss import structure_tensor_loss, l1_loss, bce_loss
from sc.models.backbone import get_backbone
from sc.models.neck.tr_neck import TRNeck
from sc.models.head.tr_head import TRHead


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # build non-local necks
        self.neck = TRNeck(**cfg.NECK.KWARGS)
        # build output head
        self.head = TRHead(**cfg.HEAD.KWARGS)

    def classify(self, x):
        # get feature
        feat = self.backbone(x)
        neck = self.neck(feat)
        head = self.head(neck)
        return F.softmax(head, dim=-1)

    def forward(self, data):
        """ only used in training
        """
        image = data['image'].cuda()
        label = data['label'].long().cuda()

        # get feature
        feat = self.backbone(image)
        neck = self.neck(feat)
        head = self.head(neck)

        loss = F.cross_entropy(head, label)

        outputs = {}
        outputs['total_loss'] = loss
        return outputs
