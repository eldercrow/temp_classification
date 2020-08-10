from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import AttributeDict as adict

# config
__C = CN()

cfg = __C

__C.META_ARC = "default-meta"

# __C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.BACKEND = 'dp'

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
__C.TRAIN.GRAD_CLIP = 10.0

# ------------------------------------------------------------------------ #
# Evaluation options
# ------------------------------------------------------------------------ #
__C.EVAL = CN()

# __C.EVAL.PRETRAINED = ''
# __C.EVAL.HYPERPARAMS = ''

# ------------------------------------------------------------------------ #
# Data batching and augmentation options
# ------------------------------------------------------------------------ #
__C.PREPROC = CN()

__C.PREPROC.NUM_WORKERS = 1
__C.PREPROC.IMAGE_HW = (32, 32)
__C.PREPROC.BATCH_SIZE = 256

__C.PREPROC.EVAL_NUM_WORKERS = 1
__C.PREPROC.EVAL_IMAGE_HW = (32, 32)
__C.PREPROC.EVAL_BATCH_SIZE = 256

__C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
__C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]

__C.PREPROC.AUG = CN()
__C.PREPROC.AUG.ROTATE = 20.0
__C.PREPROC.AUG.SCALE = 1.333
__C.PREPROC.AUG.ASPECT = 1.5
__C.PREPROC.AUG.FLIP = 1.0

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN()

# training
__C.DATASET.TRAIN = CN(new_allowed=True)
__C.DATASET.TRAIN.NAME_DOMAINS = ['CIFAR10_SAMPLED']

# CIFAR
__C.DATASET.TRAIN.CIFAR10_ORIG = CN()
__C.DATASET.TRAIN.CIFAR10_ORIG.ROOT = '~/dataset/cifar-10-images'
__C.DATASET.TRAIN.CIFAR10_ORIG.ANNO = '~/dataset/cifar-10-images/train.txt'
__C.DATASET.TRAIN.CIFAR10_ORIG.NUM_USE = -1
__C.DATASET.TRAIN.CIFAR10_ORIG.NUM_INIT_USE = -1

__C.DATASET.TRAIN.CIFAR10_SAMPLED = CN()
__C.DATASET.TRAIN.CIFAR10_SAMPLED.ROOT = '~/dataset/cifar-10-sampled'
__C.DATASET.TRAIN.CIFAR10_SAMPLED.ANNO = '~/dataset/cifar-10-sampled/train.txt'
__C.DATASET.TRAIN.CIFAR10_SAMPLED.NUM_USE = -1
__C.DATASET.TRAIN.CIFAR10_SAMPLED.NUM_INIT_USE = -1

__C.DATASET.TRAIN.CIFAR10_BIASED = CN()
__C.DATASET.TRAIN.CIFAR10_BIASED.ROOT = '~/dataset/cifar-10-biased'
__C.DATASET.TRAIN.CIFAR10_BIASED.ANNO = '~/dataset/cifar-10-biased/train.txt'
__C.DATASET.TRAIN.CIFAR10_BIASED.NUM_USE = -1
__C.DATASET.TRAIN.CIFAR10_BIASED.NUM_INIT_USE = -1

__C.DATASET.TRAIN.CIFAR10_FILTERED = CN()
__C.DATASET.TRAIN.CIFAR10_FILTERED.ROOT = '~/dataset/cifar-10-biased'
__C.DATASET.TRAIN.CIFAR10_FILTERED.ANNO = '~/dataset/cifar-10-biased/train_kept.txt'
__C.DATASET.TRAIN.CIFAR10_FILTERED.NUM_USE = -1
__C.DATASET.TRAIN.CIFAR10_FILTERED.NUM_INIT_USE = -1

# STL10
__C.DATASET.TRAIN.STL10_ORIG = CN()
__C.DATASET.TRAIN.STL10_ORIG.ROOT = '~/dataset/stl-10-images'
__C.DATASET.TRAIN.STL10_ORIG.ANNO = '~/dataset/stl-10-images/train.txt'
__C.DATASET.TRAIN.STL10_ORIG.NUM_USE = -1
__C.DATASET.TRAIN.STL10_ORIG.NUM_INIT_USE = -1

__C.DATASET.TRAIN.STL10_BIASED = CN()
__C.DATASET.TRAIN.STL10_BIASED.ROOT = '~/dataset/stl-10-biased'
__C.DATASET.TRAIN.STL10_BIASED.ANNO = '~/dataset/stl-10-biased/train.txt'
__C.DATASET.TRAIN.STL10_BIASED.NUM_USE = -1
__C.DATASET.TRAIN.STL10_BIASED.NUM_INIT_USE = -1

__C.DATASET.TRAIN.STL10_FILTERED = CN()
__C.DATASET.TRAIN.STL10_FILTERED.ROOT = '~/dataset/stl-10-biased'
__C.DATASET.TRAIN.STL10_FILTERED.ANNO = '~/dataset/stl-10-biased/train_kept.txt'
__C.DATASET.TRAIN.STL10_FILTERED.NUM_USE = -1
__C.DATASET.TRAIN.STL10_FILTERED.NUM_INIT_USE = -1

# LASOT
__C.DATASET.TRAIN.LASOT_ORIG = CN()
__C.DATASET.TRAIN.LASOT_ORIG.ROOT = '~/dataset/lasot/crop'
__C.DATASET.TRAIN.LASOT_ORIG.ANNO = '~/dataset/lasot/crop/train10.txt'
__C.DATASET.TRAIN.LASOT_ORIG.NUM_USE = -1
__C.DATASET.TRAIN.LASOT_ORIG.NUM_INIT_USE = -1

__C.DATASET.TRAIN.LASOT_FILTERED = CN()
__C.DATASET.TRAIN.LASOT_FILTERED.ROOT = '~/dataset/lasot/crop'
__C.DATASET.TRAIN.LASOT_FILTERED.ANNO = '~/dataset/lasot/crop/train10_filtered.txt'
__C.DATASET.TRAIN.LASOT_FILTERED.NUM_USE = -1
__C.DATASET.TRAIN.LASOT_FILTERED.NUM_INIT_USE = -1

# evaluation
__C.DATASET.EVAL = CN(new_allowed=True)
__C.DATASET.EVAL.NAME_DOMAINS = ['CIFAR10_ORIG']

__C.DATASET.EVAL.CIFAR10_ORIG = CN()
__C.DATASET.EVAL.CIFAR10_ORIG.ROOT = '~/dataset/cifar-10-images'
__C.DATASET.EVAL.CIFAR10_ORIG.ANNO = '~/dataset/cifar-10-images/validation.txt'
__C.DATASET.EVAL.CIFAR10_ORIG.NUM_USE = -1
__C.DATASET.EVAL.CIFAR10_ORIG.NUM_INIT_USE = -1

__C.DATASET.EVAL.STL10_ORIG = CN()
__C.DATASET.EVAL.STL10_ORIG.ROOT = '~/dataset/stl-10-images'
__C.DATASET.EVAL.STL10_ORIG.ANNO = '~/dataset/stl-10-images/validation.txt'
__C.DATASET.EVAL.STL10_ORIG.NUM_USE = -1
__C.DATASET.EVAL.STL10_ORIG.NUM_INIT_USE = -1

__C.DATASET.EVAL.LASOT_ORIG = CN()
__C.DATASET.EVAL.LASOT_ORIG.ROOT = '~/dataset/lasot/crop'
__C.DATASET.EVAL.LASOT_ORIG.ANNO = '~/dataset/lasot/crop/val10.txt'
__C.DATASET.EVAL.LASOT_ORIG.NUM_USE = -1
__C.DATASET.EVAL.LASOT_ORIG.NUM_INIT_USE = -1

# ------------------------------------------------------------------------ #
# Training model options
# ------------------------------------------------------------------------ #
__C.MODEL = CN()
__C.MODEL.NUM_CLASSES = 10

# backbone
__C.MODEL.BACKBONE = CN()
__C.MODEL.BACKBONE.NAME = 'resnet18'
__C.MODEL.BACKBONE.KWARGS = CN(new_allowed=True)


def convert_cfg_to_adict(cfg_node, key_list=[]):
    '''
    To access parameters not only attributes but dict keys.
    '''
    _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}

    if not isinstance(cfg_node, CN):
        assert type(cfg_node) in _VALID_TYPES, \
            "Key {} with value {} is not a valid type; valid types: {}".format( \
                ".".join(key_list), type(cfg_node), _VALID_TYPES)
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_adict(v, key_list + [k])
        return adict(cfg_dict)