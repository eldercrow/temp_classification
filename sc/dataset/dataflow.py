import cv2
import numpy as np
import copy
import itertools
import os
from scipy.spatial.transform import Rotation

from dataflow.dataflow import (
        imgaug, BatchData, MultiProcessMapDataZMQ, MapData, MultiThreadMapData, MultiProcessRunnerZMQ)
from dataflow.utils.argtools import log_once
# from tensorpack.utils import logger
from sc.dataset.dataset import TRDataset
from sc.dataset.augmentation import (
        ShiftScaleAugmentor, ResizeAugmentor, ColorJitterAugmentor)
from sc.config import cfg

from torch.utils.data import IterableDataset


class TPIterableDataset(IterableDataset):
    '''
    '''
    def __init__(self, df):
        super(TPIterableDataset, self).__init__()
        self._dataflow = df
        self._dataflow.reset_state()

    def __iter__(self):
        return self._dataflow.__iter__()

    def __len__(self):
        return self._dataflow.__len__()


class MalformedData(BaseException):
    pass


class TrainingDataPreprocessor:
    """
    The mapper to preprocess the input data for training.
    Since the mapping may run in other processes, we write a new class and
    explicitly pass cfg to it, in the spirit of "explicitly pass resources to subprocess".
    """
    def __init__(self, cfg):
        self.cfg = cfg
        pixel_mean = np.array(self.cfg.PREPROC.PIXEL_MEAN[::-1])
        # augmentations:
        #   shift, scale, color, flip
        augmentors = [
            imgaug.RandomChooseAug([
                imgaug.Identity(),
                imgaug.Rotation(self.cfg.DATASET.AUG.ROTATE,
                                (0.499, 0.501),
                                border_value=pixel_mean),
                imgaug.RotationAndCropValid(self.cfg.DATASET.AUG.ROTATE)
            ]),
            imgaug.RandomChooseAug([
                (imgaug.Resize(self.cfg.TRAIN.SIZE), 0.25),
                (ShiftScaleAugmentor(self.cfg.DATASET.AUG.SCALE,
                                     self.cfg.DATASET.AUG.ASPECT,
                                     self.cfg.TRAIN.SIZE,
                                     pixel_mean
                                     ), 0.75)
            ]),
            ColorJitterAugmentor(),
            imgaug.Flip(horiz=True)
        ]
        self.aug = imgaug.AugmentorList(augmentors)

    def __call__(self, datum_dict):
        '''
        datum_dict: dict of {'fn_img', 'zenith', 'horizon'}
            fn_img: full image path
            zenith: [vx, vy]
            horizon: [hx0, hy0, hx1, hy1]
        '''
        fn_img, cid = datum_dict['fn_img'], datum_dict['cid']

        # load images
        image = cv2.imread(fn_img)
        try:
            hh, ww = image.shape[:2]
        except:
            log_once('Could not load {}.'.format(fn_img), 'warn')
            return None

        # apply augmentation
        # to image
        tfms = self.aug.get_transform(image)
        r_img = tfms.apply_image(image)

        r_img = np.transpose(r_img, (2, 0, 1)).astype(np.float32)
        r_img -= np.array(cfg.PREPROC.PIXEL_MEAN[::-1]).reshape((3, 1, 1))

        ret = { \
                'image': r_img,
                'label': cid
                }
                # 'template_box': template_box
        return ret


def get_train_dataflow():
    '''
    training dataflow with data augmentation.
    '''
    ds = TRDataset()
    train_preproc = TrainingDataPreprocessor(cfg)

    if cfg.TRAIN.NUM_WORKERS == 1:
        ds = MapData(ds, train_preproc)
    else:
        ds = MultiProcessMapDataZMQ(ds, cfg.TRAIN.NUM_WORKERS, train_preproc)
    ds = BatchData(ds, cfg.TRAIN.BATCH_SIZE)
    return TPIterableDataset(ds)


def get_eval_dataflow():
    '''
    evaluation dataflow with crop-resize-batch
    '''
    ds = TRDataset(shuffle=False)
    sz = cfg.EVAL.SIZE
    augmentors = [
            imgaug.ResizeShortestEdge(sz+32, interp=cv2.INTER_LINEAR),
            imgaug.CenterCrop((sz, sz)),
        ]
    aug = imgaug.AugmentorList(augmentors)

    #
    def mapf(dp):
        fname, cls = dp['fn_img'], dp['cid']
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = aug.augment(im)
        im = np.transpose(im, (2, 0, 1)).astype(np.float32)
        im -= np.array(cfg.PREPROC.PIXEL_MEAN[::-1]).reshape((3, 1, 1))
        return { 'image': im, 'label': cls }
    #

    if cfg.EVAL.NUM_WORKERS == 1:
        ds = MapData(ds, mapf)
    else:
        ds = MultiThreadMapData(ds, cfg.EVAL.NUM_WORKERS, mapf, buffer_size=2000, strict=True)
    ds = BatchData(ds, cfg.EVAL.BATCH_SIZE, remainder=True)
    if cfg.EVAL.NUM_WORKERS > 1:
        ds = MultiProcessRunnerZMQ(ds, 1)
    return TPIterableDataset(ds)
