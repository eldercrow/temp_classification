import numpy as np
import os, sys


class CIFAR10Image(object):
    '''
    Inferface for loading a single dataset.
    '''
    def __init__(self, name, root, anno, num_init_use=-1):
        '''
        name: dataset name and domain, e.g. 'VisDA:real', 'VisDA:clipart'
        root: root path to the images
        anno: annotation file name
        '''
        self.name = name
        self.root = root

        # load metadata from .json annotation file
        logger.info("loading {} from {}".format(name, anno))
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
        logger.info("{} loaded".format(self.name))

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
