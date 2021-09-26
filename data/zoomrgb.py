from pathlib import Path
from .srdata import SRData


class ZoomRGB(SRData):
    def __init__(self, dir, scale, name='ZoomRGB', train=True, patch_size=48, augment=True, **kwargs):

        super(ZoomRGB, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=1, augment=augment)

    def _set_filesystem(self, data_dir):
        if isinstance(data_dir, str):
            self.apath = Path(data_dir) / "SRRAW"
        elif isinstance(data_dir, Path):
            self.apath = data_dir / "SRRAW"

        assert self.apath.exists(), "Given base data dir path is wrong!: {}".format(self.apath)

        if self.train:
            self.dir_hr = self.apath / 'train' / 'rgb_HR'
            self.dir_lr = self.apath / 'train' / 'rgb_LR_bicubic'
        else:
            self.dir_hr = self.apath / 'test' / 'rgb_HR'
            self.dir_lr = self.apath / 'test' / 'rgb_LR_bicubic'     

        assert self.dir_hr.exists(), "HR input data path does not exist!"
        assert self.dir_lr.exists(), "LR input data path does not exist!"

        self.ext = ("jpg", "jpg")