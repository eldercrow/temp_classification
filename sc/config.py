from yacs.config import CfgNode as CN


# config
__C = CN()

cfg = __C

__C.META_ARC = "default-meta"

# __C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.BACKEND = 'ddp'

__C.TRAIN.PRETRAINED = ''
__C.TRAIN.RESUME = ''

__C.TRAIN.LOG_DIR = './logs'
__C.TRAIN.SNAPSHOT_DIR = './snapshot'
__C.TRAIN.PRINT_FREQ = 20
__C.TRAIN.LOG_GRADS = False

__C.TRAIN.EPOCH = 100
__C.TRAIN.START_EPOCH = 0

__C.TRAIN.OPTIMIZER = CN()
__C.TRAIN.OPTIMIZER.NAME = 'SGD'
__C.TRAIN.OPTIMIZER.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_SCHEDULER = CN()
__C.TRAIN.LR_SCHEDULER.NAME = 'ExponentialLR'
__C.TRAIN.LR_SCHEDULER.KWARGS = CN(new_allowed=True)
# __C.TRAIN.MOMENTUM = 0.9
# __C.TRAIN.WEIGHT_DECAY = 0.0001
# __C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 1.0

# __C.TRAIN.LR = CN()

# __C.TRAIN.LR.TYPE = 'log'

# __C.TRAIN.LR.KWARGS = CN(new_allowed=True)

# __C.TRAIN.LR_WARMUP = CN()

# __C.TRAIN.LR_WARMUP.WARMUP = True

# __C.TRAIN.LR_WARMUP.TYPE = 'step'

# __C.TRAIN.LR_WARMUP.EPOCH = 2

# __C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# __C.TRAIN.FREEZE_BACKBONE = False

# __C.TRAIN.FREEZE_NECK = False

# __C.TRAIN.BACKBONE_TRAIN_LAYERS = []

# __C.EVAL = CN()

# __C.EVAL.SIZE = 224

# __C.EVAL.NUM_WORKERS = 1

# __C.EVAL.BATCH_SIZE = 1

# ------------------------------------------------------------------------ #
# Data batching and augmentation options
# ------------------------------------------------------------------------ #
__C.PREPROC = CN()

__C.PREPROC.NUM_WORKERS = 1
__C.PREPROC.IMAGE_HW = (96, 96)
__C.PREPROC.BATCH_SIZE = 128

__C.PREPROC.EVAL_NUM_WORKERS = 1
__C.PREPROC.EVAL_IAMGE_HW = (96, 96)
__C.PREPROC.EVAL_BATCH_SIZE = 1

__C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
__C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]

__C.PREPROC.AUG = CN()
__C.PREPROC.AUG.ROTATE = 30.0
__C.PREPROC.AUG.SCALE = 2.0
__C.PREPROC.AUG.ASPECT = 2.5
__C.PREPROC.AUG.FLIP = 1.0

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

__C.DATASET.VISDA_CLIPART = CN()
__C.DATASET.VISDA_CLIPART.ROOT = '~/dataset/visda'
__C.DATASET.VISDA_CLIPART.ANNO = '~/dataset/visda/clipart_train.txt'
__C.DATASET.VISDA_CLIPART.NUM_USE = -1
__C.DATASET.VISDA_CLIPART.NUM_INIT_USE = -1

# ------------------------------------------------------------------------ #
# Training model options
# ------------------------------------------------------------------------ #
__C.MODEL = CN()
__C.MODEL.NUM_CLASSES = 10

# backbone
__C.MODEL.BACKBONE = CN()
__C.MODEL.BACKBONE.NAME = 'resnet18'
__C.MODEL.BACKBONE.KWARGS = CN(new_allowed=True)