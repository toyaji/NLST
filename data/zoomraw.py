import cv2
import numpy as np
from data import common
from pathlib import Path
from .srdata import SRData

#from data import common

class ZoomRaw(SRData):
    def __init__(self, dir, scale, name='ZoomRaw', train=True, patch_size=48, augment=True, **kwargs):

        super(ZoomRaw, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=255, augment=augment)

    def _set_filesystem(self, data_dir):
        if isinstance(data_dir, str):
            self.apath = Path(data_dir) / "SRRAW"
        elif isinstance(data_dir, Path):
            self.apath = data_dir / "SRRAW"

        assert self.apath.exists(), "Given base data dir path is wrong!: {}".format(self.apath)

        if self.train:
            self.dir_hr = self.apath / 'train' / 'raw_HR_binary'
            self.dir_lr = self.apath / 'train' / 'raw_LR_binary'
        else:
            self.dir_hr = self.apath / 'test' / 'raw_HR_binary'
            self.dir_lr = self.apath / 'test' / 'raw_LR_binary'     

        assert self.dir_hr.exists(), "HR input data path does not exist!"
        assert self.dir_lr.exists(), "LR input data path does not exist!"

        self.ext = ("npy", "npy")

    def _load_file(self, idx):
        f_hr = self.hr_pathes[idx]
        f_lr = self.lr_pathes[idx]
        filename = f_hr.name

        hr = np.load(f_hr)
        lr = np.load(f_lr)

        return lr, hr, filename

