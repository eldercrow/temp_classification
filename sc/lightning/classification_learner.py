import os
import multiprocessing

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import pytorch_lightning as pl

# from sc.config import cfg
from sc.dataset.dataflow import get_train_dataflow, get_eval_dataflow
from sc.models.model_builder import ModelBuilder


# lightening model for training
class ClassficationLearner(pl.LightningModule):
    '''
    '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ModelBuilder(cfg.MODEL)

    def forward(self, batch):
        '''
        x: batch of images and labels
        returns dict of losses
        '''
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        '''
        batch: dict of {'image', 'label'}
        '''
        images, labels = batch['image'], batch['label']
        loss_dict = self(images)
        return loss_dict

    def configure_optimizers(self):
        '''
        Optimizer and LR scheduler
        '''
        cfg_opt = self.cfg.TRAIN.OPTIMIZER
        cfg_lr = self.cfg.TRAIN.LR_SCHEDULER
        optimizer = getattr(optim, cfg_opt.NAME)(**cfg_opt.KWARGS)
        scheduler = getattr(lr_scheduler, cfg_lr.NAME)(optimizer, **cfg_lr.KWARGS)
        # optimizer = optim.SGD(
        #     self.parameters(),
        #     lr=self.lr,
        #     momentum=self.momentum,
        #     weight_decay=self.weight_decay
        # )
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        '''
        Process a validation batch and compute accuracies.
        '''
        images, labels = batch['image'], batch['label']
        output = self.model.predict(images)
        loss_val = F.cross_entropy(output, labels)
        acc1 = self.__accuracy(output, target, topk=(1,))[0]

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc1': acc1,
        })

        return output

    def validation_epoch_end(self, outputs):
        '''
        Record valudation accuracies.
        '''
        tqdm_dict = {}

        for metric_name in ["val_loss", "val_acc1"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    @classmethod
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train_dataloader(self):
        '''
        '''
        train_dataset = get_train_dataflow(self.cfg.PREPROC)
        train_loader = DataLoader(train_dataset, batch_size=None, batch_sampler=none, sampler=None)
        return train_loader

    def val_dataloader(self):
        '''
        '''
        val_dataset = get_eval_dataflow(self.cfg.PREPROC)
        val_loader = DataLoader(val_dataset, batch_size=None, batch_sampler=none, sampler=None)
        return val_loader