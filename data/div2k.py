import cv2
import numpy as np
from data import common
import torch
from pathlib import Path
from torch.utils.data import Dataset


class DIV2K(Dataset):
    def __init__(self, dir, scale_idx, patch_size, train=True, **kargs):
        self.patch_size = patch_size
        self.scale_idx = scale_idx
        self.train = train
        self._set_filesystem(dir)

    def __getitem__(self, idx):
        
        hr = cv2.imread(str(self.hr_pathes[idx]))
        lr =cv2.imread(str(self.lr_pathes[idx]))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)    
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        h, w, _ = hr.shape
        lr = cv2.resize(lr, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        if self.patch_size == -1:
            pass
        else:
            hr, lr = common.get_random_patch(hr, lr, self.patch_size)

        hr, lr = common.augment([hr, lr], hflip=True, rot=True)  
        hr, lr = common.np2Tensor([hr, lr], rgb_range=1)  
        return lr, hr

    def __len__(self):
        return len(self.hr_pathes)

    def _set_filesystem(self, data_dir):
        if isinstance(data_dir, str):
            self.apath = Path(data_dir)
        
        if self.train:
            self.dir_hr = self.apath / 'HR' / 'train_HR'
            self.dir_lr = self.apath / 'LR_bicubic' / 'DIV2K_train_LR_bicubic'
        else:
            self.dir_hr = self.apath / 'HR' / 'valid_HR'
            self.dir_lr = self.apath / 'LR_bicubic' / 'DIV2K_valid_LR_bicubic'

        self.hr_pathes = list(self.dir_hr.glob("*"))
        self.lr_pathes = list((self.dir_lr / "X{:1d}".format(self.scale_idx)).glob("*"))
    
