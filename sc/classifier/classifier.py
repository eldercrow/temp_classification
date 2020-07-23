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


class Classifier(torch.nn.Module):
    def __init__(self, model, cfg): #image_hw):
        super(Classifier, self).__init__()

        self.model = model
        # self.model.eval()
        self.device = next(model.parameters()).device

        self.hw = cfg.EVAL_IMAGE_HW #image_hw
        self.pixel_mean = np.array(cfg.PIXEL_MEAN).reshape((3, 1, 1))

    def _preprocess_img(self, img):
        img = cv2.resize(img, (self.hw[1], self.hw[0]))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img -= self.pixel_mean
        # img -= np.array(cfg.PREPROC.PIXEL_MEAN).reshape((3, 1, 1))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)

        # if cfg.CUDA:
        #     img = img.cuda()
        return img.to(self.device)

    def _preprocess_batch(self, img):
        # if cfg.CUDA:
        #     img = img.cuda()
        return img.to(self.device)

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
