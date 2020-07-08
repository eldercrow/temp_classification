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
from sc.utils.model_load import load_pretrain
from sc.dataset.dataflow import get_eval_dataflow


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


def build_data_loader():
    logger.info("build eval dataset")
    # train_dataset
    eval_dataset = get_eval_dataflow() #TrkDataset()
    logger.info("build dataset done")

    # let tensorpack handle all the distributed data loading
    eval_loader = DataLoader(eval_dataset,
                             batch_size=None,
                             batch_sampler=None,
                             sampler=None)
    return eval_loader


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.EVAL.NUM_WORKERS = args.num_workers

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    eval_loader = build_data_loader()

    # build tracker
    classifier = Classifier(model, cfg.EVAL.SIZE)

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
