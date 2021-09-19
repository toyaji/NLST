import cv2
import numpy as np
from data import common
import torch
from pathlib import Path
from torch.utils.data import Dataset


class BSD500(Dataset):
    def __init__(self, dir, scale_idx, patch_size, train=True, test=False, **kargs):
        self.patch_size = patch_size
        self.scale_idx = scale_idx
        self.train = train
        self._set_filesystem(dir)

    def __getitem__(self, idx):
        
        hr = cv2.imread(str(self.hr_pathes[idx]))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)    
        h, w, _ = hr.shape
        lr = cv2.resize(hr, dsize=(0, 0), fx=1/self.scale_idx, fy=1/self.scale_idx, interpolation=cv2.INTER_LINEAR)
        # TODO gaussian noise 필요하면 추가하기
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
        
        assert self.apath.exists(), "Data dir path is wrong!"
        
        if self.train:
            self.dir_hr = self.apath / 'images' / 'train'
        else:
            self.dir_hr = self.apath / 'images' / 'val'

        if not self.train and self.test:
            self.dir_hr = self.apath / 'images' / 'test'

        self.hr_pathes = list(self.dir_hr.glob("*"))