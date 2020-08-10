# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import logging

from sc.config import cfg
from sc.models.model_builder import ModelBuilder
from sc.classifier.classifier import Classifier
from torch.utils.data import DataLoader
# from sc.utils.model_load import load_pretrain
from sc.dataset.dataflow import get_eval_dataflow
from sc.dataset.dataset import TRDataset

'''
This function is to show how to use trained models w/o pytorch-lightning.
One can evaluate the performance of a trained classifier using this function.
'''

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='classification eval')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--num-workers', type=int, default=1,
        help='number of cpu workers per gpu node')
args = parser.parse_args()

torch.set_num_threads(1)
device = torch.device('cuda')


def build_data_loader():
    logger.info("build eval dataset")
    # train_dataset
    ds = TRDataset(cfg.DATASET.EVAL)
    eval_dataset = get_eval_dataflow(ds, cfg.PREPROC) #TrkDataset()
    logger.info("build dataset done")

    # let tensorpack handle all the distributed data loading
    eval_loader = DataLoader(eval_dataset,
                             batch_size=None,
                             batch_sampler=None,
                             sampler=None)
    return eval_loader


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['state_dict']
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    model.load_state_dict(pretrained_dict, strict=True)
    return model


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.EVAL.NUM_WORKERS = args.num_workers

    # data loader
    eval_loader = build_data_loader()

    # build classifier
    model = ModelBuilder(cfg.MODEL).to(device)
    classifier = Classifier(model, cfg.PREPROC)

    # load weights
    classifier = load_pretrain(classifier, args.snapshot).eval()

    total = 0
    correct = 0

    for idx, data in enumerate(eval_loader):
        outputs = classifier.classify(data['image'])['class_id']
        labels = data['label'].numpy()

        total += len(labels)
        correct += len(np.where(labels == outputs)[0])

    acc = correct / total

    print('Accuracy = {}'.format(acc))


if __name__ == '__main__':
    main()
