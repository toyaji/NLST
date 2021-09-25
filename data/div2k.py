import cv2
import numpy as np
from data import common
import torch
from pathlib import Path
from .srdata import SRData


class DIV2K(SRData):
    def __init__(self, dir, scale, name='DIV2K', train=True, patch_size=48, augment=True, **kwargs):

        super(DIV2K, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=1, augment=augment)

    def __len__(self):
        return len(self.hr_pathes)

    def _scan(self):
        if self.train:
            self.hr_pathes = sorted(list(self.dir_hr.glob("*")))[:800]
            self.lr_pathes = sorted(list((self.dir_lr / "X{:1d}".format(self.scale)).glob("*")))[:800]
        else:
            self.hr_pathes = sorted(list(self.dir_hr.glob("*")))[800:]
            self.lr_pathes = sorted(list((self.dir_lr / "X{:1d}".format(self.scale)).glob("*")))[800:]

    def _set_filesystem(self, data_dir):
        super(DIV2K, self)._set_filesystem(data_dir)
        
        self.dir_hr = self.apath / 'DIV2K_train_HR'
        self.dir_lr = self.apath / 'DIV2K_train_LR_bicubic'


    
