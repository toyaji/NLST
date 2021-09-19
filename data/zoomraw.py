import cv2
import numpy as np
from data import common
import torch
from pathlib import Path
from torch.utils.data import Dataset

#from data import common

class ZoomLZoomRaw(Dataset):
    """
    This data class return data pair which consist of 7 images having different resolution  
    
    """
    def __init__(self, dir, scale_idx, patch_size, train=True):
        super().__init__()
        self.patch_size = patch_size
        self.scale_idx = scale_idx
        self.train = train
        self._set_filesystem(dir)

    def __getitem__(self, idx):
        hr, hr_raw, lr_raw = self._scan(idx)
        hr, lr = common.get_raw_patch(hr, lr_raw, self.patch_size)
        # to reduce computational cost, we can consider reduce the size of inputs
        if self.reduce_size != 1:
            hr = cv2.resize(hr, dsize=(0, 0), fx=self.reduce_size, fy=self.reduce_size, interpolation=cv2.INTER_LINEAR)
            lr = cv2.resize(lr, dsize=(0, 0), fx=self.reduce_size, fy=self.reduce_size, interpolation=cv2.INTER_LINEAR)
        
        hr, lr = common.augment([hr, lr], hflip=True, rot=True)
        hr, lr = common.np2Tensor([hr, lr], rgb_range=1)
        return lr, hr

    def __len__(self):
        return len(self.base_paths)

    def _set_filesystem(self, dir_data):
        if isinstance(dir_data, str):
            self.apath = Path(dir_data)
        # check for path exist
        assert self.apath.exists(), "Data dir path is wrong!"

        if self.train:
            self.base_paths = sorted(list((self.apath / "train").glob("*")))
        else:
            self.base_paths = sorted(list((self.apath / "test").glob("*")))
    
    def _scan(self, idx):
        jpg_path = self.base_paths[idx] / "00001.JPG"
        raw_path = self.base_paths[idx] / "00001.ARW"

        hr = cv2.imread(str(jpg_path))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        # load hr and lr bayer
        bayer = common.get_bayer(raw_path)
        hr_raw = common.get_4ch(bayer)
        h, w = hr_raw.shape[:2]

        lr_raw = cv2.resize(hr_raw, dsize=(0,0), fx=1/self.scale_idx, fy=1/self.scale_idx, interpolation=cv2.INTER_LINEAR)
        lr_raw = cv2.resize(lr_raw, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        return hr, hr_raw, lr_raw

    def _get_focalscale(self, idx):
        ref_paths = self.base_paths[idx].glob("*.JPG")
        focals = [common.readFocal_pil(p) for p in ref_paths]
        focal_scale = np.array(focals) / 240
        return focal_scale

