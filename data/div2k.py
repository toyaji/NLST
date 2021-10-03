from .srdata import SRData
from . import common

class DIV2K(SRData):
    def __init__(self, dir, scale, name='DIV2K', train=True, patch_size=48, augment=True, **kwargs):

        super(DIV2K, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=1, augment=augment)

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

        assert self.dir_hr.exists(), "HR input data path does not exist!"
        assert self.dir_lr.exists(), "LR input data path does not exist!"

    def get_patch(self, lr, hr):
        scale = self.scale
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


    
