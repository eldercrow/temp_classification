import os
import argparse
import multiprocessing
import logging
import json
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger

from sc.config import cfg, convert_cfg_to_adict
from sc.dataset.dataflow import get_train_dataflow, get_eval_dataflow
from sc.models.model_builder import ModelBuilder
from sc.lightning.classification_learner import ClassficationLearner
# from sc.utils.log_helper import init_log, add_file_handler
# from sc.utils.misc import describe, commit
 

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='classification train')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='.yaml config file')
parser.add_argument('--seed', type=int, 
                    help='random seed')
parser.add_argument('--gpus', type=int, default=1,
                    help='how many gpus')
# parser.add_argument('--local_rank', type=int, default=0,
#                     help='compulsory for pytorch launcer')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of cpu workers per gpu node')
parser.add_argument('--evaluate', action='store_true',
                    help='run evaluation instead of training')
args = parser.parse_args()


# def main(args: Namespace) -> None:
def main():
    '''
    '''
    # load cfg
    cfg.merge_from_file(args.cfg)
    cfg.PREPROC.NUM_WORKERS = args.num_workers

    # training model
    model = ClassficationLearner(convert_cfg_to_adict(cfg))

    # setup log
    # if not os.path.exists(cfg.TRAIN.LOG_DIR):
    #     os.makedirs(cfg.TRAIN.LOG_DIR)
    # init_log('global', logging.INFO)
    # if cfg.TRAIN.LOG_DIR:
    #     add_file_handler('global',
    #                         os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
    #                         logging.INFO)

    # logger.info("Version Information: \n{}\n".format(commit()))
    # logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     cudnn.deterministic = True

    # checkpoint 
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.TRAIN.LOG_DIR, #os.getcwd(),
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.TRAIN.BACKEND,
        max_epochs=cfg.TRAIN.EPOCH,
        gradient_clip_val=cfg.TRAIN.GRAD_CLIP,
        # precision=16 if args.use_16bit else 32,
    )

    if args.evaluate:
        model.load_from_checkpoint(
            cfg.EVAL.PRETRAINED,
            convert_cfg_to_adict(cfg),
            hparam_overrides=convert_cfg_to_adict(cfg))
            # convert_cfg_to_adict(cfg))
        trainer.test(model)
        # trainer.load_from_checkpoint(cfg.EVAL.PRETRAINED)
        # model.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()