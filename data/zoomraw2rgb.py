import cv2
import numpy as np
import rawpy
from pathlib import Path
from .srdata import SRData
from . import common

class ZoomRaw2RGB(SRData):
    def __init__(self, dir, scale, name='ZoomRaw2RGB', train=True, patch_size=48, rgb_range=1, augment=True, **kwargs):

        super(ZoomRaw2RGB, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=rgb_range, augment=augment)

    def _set_filesystem(self, data_dir):
        if isinstance(data_dir, str):
            self.apath = Path(data_dir) / "SRRAW"
        elif isinstance(data_dir, Path):
            self.apath = data_dir / "SRRAW"

        assert self.apath.exists(), "Given base data dir path is wrong!: {}".format(self.apath)

        if self.train:
            self.dir_hr = self.apath / 'train' / 'raw_HR'
        else:
            self.dir_hr = self.apath / 'test' / 'raw_HR'    

        assert self.dir_hr.exists(), "HR input data path does not exist!"

        self.ext = ("arw", "arw")

    def _scan(self):
        self.hr_pathes = sorted(list(self.dir_hr.glob("*." + self.ext[0])))

    def _load_file(self, idx):
        f_hr = self.hr_pathes[idx]
        filename = f_hr.name
        black_lv=512; white_lv=16383
        with rawpy.imread(str(f_hr)) as r:
            hr_bayer = r.raw_image_visible.astype(np.float32)
            hr_bayer = (hr_bayer - black_lv) / (white_lv - black_lv)
            #wb = common.compute_wb(r)
            hr_rgb = r.postprocess(no_auto_bright=False, use_camera_wb=True, output_bps=8)
            hr_raw = common.get_4ch(hr_bayer)
        
            h, w = hr_raw.shape[:2]
            lr_raw = cv2.resize(hr_raw, (w // self.scale, h // self.scale), interpolation=cv2.INTER_LINEAR)
            hr_rgb = hr_rgb / 255
        return lr_raw, hr_rgb, filename

    def get_patch(self, lr, hr):
        scale = self.scale * 2
        if self.patch_size > 0:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.patch_size,    # raw bayer has x2 smaller size
                scale=scale,
                input_large=self.input_large
            )
            if self.augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr
   