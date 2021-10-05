import cv2
import numpy as np
from pathlib import Path
from .srdata import SRData
from . import common

class ZoomRaw2RGB(SRData):
    def __init__(self, dir, scale, name='ZoomRaw2RGB', train=True, patch_size=48, augment=True, **kwargs):

        super(ZoomRaw2RGB, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=1, augment=augment)

    def _set_filesystem(self, data_dir):
        if isinstance(data_dir, str):
            self.apath = Path(data_dir) / "SRRAW"
        elif isinstance(data_dir, Path):
            self.apath = data_dir / "SRRAW"

        assert self.apath.exists(), "Given base data dir path is wrong!: {}".format(self.apath)

        if self.train:
            self.dir_hr = self.apath / 'train' / 'rgb_HR'
            self.dir_lr = self.apath / 'train' / 'raw_LR_binary'
        else:
            self.dir_hr = self.apath / 'test' / 'rgb_HR'
            self.dir_lr = self.apath / 'test' / 'raw_LR_binary'     

        assert self.dir_hr.exists(), "HR input data path does not exist!"
        assert self.dir_lr.exists(), "LR input data path does not exist!"

        self.ext = ("jpg", "npy")

    def _load_file(self, idx):
        f_hr = self.hr_pathes[idx]
        f_lr = self.lr_pathes[idx]
        filename = f_hr.name
        hr = cv2.imread(str(f_hr))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)    
        lr = np.load(f_lr)
        lr = lr[:, :, :3]
        return lr*255, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale * 2
        if self.patch_size > 0:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.patch_size,
                scale=scale,
                input_large=self.input_large
            )
            if self.augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr
   