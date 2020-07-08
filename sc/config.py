# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "transferability-resnet18"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.SIZE = 224

# __C.TRAIN.OUTPUT_SIZE = 7

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 100

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 128

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 1.0

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 2

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.FREEZE_BACKBONE = False

__C.TRAIN.FREEZE_NECK = False

__C.TRAIN.BACKBONE_TRAIN_LAYERS = []

__C.EVAL = CN()

__C.EVAL.SIZE = 224

__C.EVAL.NUM_WORKERS = 1

__C.EVAL.BATCH_SIZE = 1

__C.PREPROC = CN()
__C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
__C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# __C.DATASET.TEMPLATE = CN()
# __C.DATASET.TEMPLATE.SHIFT = 4.0 / 64.0 #4
# __C.DATASET.TEMPLATE.SCALE = 1.1111
# __C.DATASET.TEMPLATE.ASPECT = 1.1
# __C.DATASET.TEMPLATE.BLUR = 0.0
# __C.DATASET.TEMPLATE.FLIP = 0.0
# __C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.AUG = CN()
__C.DATASET.AUG.ROTATE = 30.0
__C.DATASET.AUG.SCALE = 2.0
__C.DATASET.AUG.ASPECT = 2.5
__C.DATASET.AUG.FLIP = 1.0

__C.DATASET.NAME_DOMAINS = ('VISDA_CLIPART',) # 'VISDA:REAL',)

__C.DATASET.VISDA_CLIPART = CN()
__C.DATASET.VISDA_CLIPART.ROOT = '~/dataset/visda'
__C.DATASET.VISDA_CLIPART.ANNO = '~/dataset/visda/clipart_train.txt'
__C.DATASET.VISDA_CLIPART.NUM_USE = -1
__C.DATASET.VISDA_CLIPART.NUM_INIT_USE = -1

__C.DATASET.VISDA_REAL = CN()
__C.DATASET.VISDA_REAL.ROOT = '~/dataset/visda'
__C.DATASET.VISDA_REAL.ANNO = '~/dataset/visda/real_train.txt'
__C.DATASET.VISDA_REAL.NUM_USE = -1
__C.DATASET.VISDA_REAL.NUM_INIT_USE = -1

__C.DATASET.VISDA_INFOGRAPH = CN()
__C.DATASET.VISDA_INFOGRAPH.ROOT = '~/dataset/visda'
__C.DATASET.VISDA_INFOGRAPH.ANNO = '~/dataset/visda/infograph_train.txt'
__C.DATASET.VISDA_INFOGRAPH.NUM_USE = -1
__C.DATASET.VISDA_INFOGRAPH.NUM_INIT_USE = -1

__C.DATASET.VISDA_SKETCH = CN()
__C.DATASET.VISDA_SKETCH.ROOT = '~/dataset/visda'
__C.DATASET.VISDA_SKETCH.ANNO = '~/dataset/visda/sketch_train.txt'
__C.DATASET.VISDA_SKETCH.NUM_USE = -1
__C.DATASET.VISDA_SKETCH.NUM_INIT_USE = -1

__C.DATASET.VISDA_QUICKDRAW = CN()
__C.DATASET.VISDA_QUICKDRAW.ROOT = '~/dataset/visda'
__C.DATASET.VISDA_QUICKDRAW.ANNO = '~/dataset/visda/quickdraw_train.txt'
__C.DATASET.VISDA_QUICKDRAW.NUM_USE = -1
__C.DATASET.VISDA_QUICKDRAW.NUM_INIT_USE = -1

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'resnet18'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# ------------------------------------------------------------------------ #
# Neck layer options
# ------------------------------------------------------------------------ #
__C.NECK = CN()

__C.NECK.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Head layer options
# ------------------------------------------------------------------------ #
__C.HEAD = CN()

__C.HEAD.KWARGS = CN(new_allowed=True)
