# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sc.models.backbone.alexnet import alexnetlegacy, alexnet
from sc.models.backbone.mobilenetv2 import mobilenetv2
from sc.models.backbone.resnet import resnet18, resnet34, resnet50

BACKBONES = {
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
