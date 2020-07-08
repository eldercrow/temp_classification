from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
import cv2

from sc.config import cfg
from sc.models.model_builder import ModelBuilder


class Classifier(object):
    def __init__(self, model, size):
        self.model = model
        self.size = size
        self.model.eval()

    def _preprocess_img(self, img):
        img = cv2.resize(img, (self.size, self.size))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img -= np.array(cfg.PREPROC.PIXEL_MEAN).reshape((3, 1, 1))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        if cfg.CUDA:
            img = img.cuda()
        return img

    def _preprocess_batch(self, img):
        if cfg.CUDA:
            img = img.cuda()
        return img

    def classify(self, img):
        '''
        '''
        if img.dim() == 3:
            data = self._preprocess_img(img)
        elif img.dim() == 4:
            data = self._preprocess_batch(img)
        else:
            assert False, 'img should have 3 or 4 dimensions.'

        probs = self.model.classify(data)

        cid = torch.argmax(probs, dim=-1)

        return { 'class_id': cid.detach().cpu().numpy() }
