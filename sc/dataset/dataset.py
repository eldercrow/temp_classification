import numpy as np
import os, sys
import json
import pickle
from dataflow.dataflow import *
from dataflow.utils import logger
# from dataflow.dataflow.utils import *
from sc.config import cfg


class SubDataset(object):
    '''
    Inferface for loading a single dataset.
    '''
    def __init__(self, name_domain, root, anno, num_init_use=-1):
        '''
        name_domain: dataset name and domain, e.g. 'VisDA:real', 'VisDA:clipart'
        root: root path to the images
        anno: annotation file name
        '''
        self.name_domain = name_domain
        self.root = root
        self.anno = anno

        # load metadata from .json annotation file
        logger.info("loading {} from {}".format(name_domain, anno))
        if anno.endswith('.txt'):
            ### TODO: revise this later
            # fn_split = os.path.join(root.replace('images', 'split'), 'train.txt')
            # with open(fn_split, 'r') as fh:
            #     train_set = set(fh.read().splitlines())
            ###
            meta_data = {}
            with open(anno, 'r') as fh:
                raw_data = fh.read().splitlines()
            raw_data = [r.split(' ') for r in raw_data]
            for r in raw_data:
                # if r[0] not in train_set:
                #     continue
                k = os.path.splitext(r[0])[0]
                meta_data[k] = { 'fn_img': os.path.join(self.root, r[0]), 'cid': int(r[1]) }
        else:
            raise ValueError()

        if num_init_use > 0:
            keys = np.random.permutation(list(meta_data.keys()))[:num_init_use]
            meta_data = { k: meta_data[k] for k in keys }

        # load annotation
        '''
        metadata:
            {
                image name (w/o extension): class id
            }
        '''
        self.labels = meta_data
        self._images = list(meta_data.keys())
        self.indices = [k for k in self._images]
        logger.info("{} loaded".format(self.name_domain))

    def __len__(self):
        return len(self.indices)

    def get_image_anno(self, index):
        annot = self.labels[index]
        return annot

    def shuffle_resize(self, size, shuffle=True):
        '''
        '''
        if size < 0: # no repeat
            if shuffle:
                pick = np.random.permutation(self._images).tolist()
            else:
                pick = [v for v in self._images]
            size = len(pick)
        else:
            pick = []
            m = 0
            while m < size:
                if shuffle:
                    pick.extend(np.random.permutation(self._images).tolist())
                else:
                    pick.extend([v for v in self._images])
                m = len(pick)
        self.indices = pick[:size]
        # logger.info("shuffle done!")
        # logger.info("dataset length {}".format(self.num))
        # return pick[:self.num]


class TRDataset(RNGDataFlow):
    '''
    Dataset class that merges multiple sub datasets.
    '''
    def __init__(self, shuffle=True):
        '''
        Get and merge all sub datasets
        '''
        self.shuffle = shuffle

        self.all_dataset = []
        # start = 0
        # self.num = 0
        # self.all_nums = []
        for name_domain in cfg.DATASET.NAME_DOMAINS:
            subdata_cfg = getattr(cfg.DATASET, name_domain)
            sub_dataset = SubDataset(
                    name_domain,
                    os.path.expanduser(subdata_cfg.ROOT),
                    os.path.expanduser(subdata_cfg.ANNO),
                    subdata_cfg.NUM_INIT_USE
                )
            sub_dataset.shuffle_resize(subdata_cfg.NUM_USE, shuffle=shuffle)
            # sub_dataset.log()
            self.all_dataset.append(sub_dataset)
            # self.all_nums.append(subdata_cfg.NUM_USE)

    def __len__(self):
        return np.sum([len(d) for d in self.all_dataset])

    def __iter__(self):
        '''
        '''
        # first reset all sub datasets
        for db in self.all_dataset:
            db.shuffle_resize(len(db), shuffle=self.shuffle)

        # merge all
        all_data_list = []
        for i, db in enumerate(self.all_dataset):
            all_data_list.extend([(i, datum) for datum in db.indices])

        if self.shuffle:
            np.random.shuffle(all_data_list)

        # populate one by one
        for (db_idx, index) in all_data_list:
            yield self.all_dataset[db_idx].get_image_anno(index)
