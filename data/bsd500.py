from .srdata import SRData


class BSD500(SRData):
    def __init__(self, dir, scale, name='BSD500', train=True, patch_size=48, augment=True, **kwargs):

        super(BSD500, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=1, augment=augment)

    def _scan(self):
        if self.train:
            self.hr_pathes = sorted(list(self.dir_hr.glob("*")))
            self.lr_pathes = sorted(list((self.dir_lr / "X{:1d}".format(self.scale)).glob("*")))
        else:
            self.hr_pathes = sorted(list(self.dir_hr.glob("*")))
            self.lr_pathes = sorted(list((self.dir_lr / "X{:1d}".format(self.scale)).glob("*")))

    def _set_filesystem(self, data_dir):
        super(BSD500, self)._set_filesystem(data_dir)
        
        if self.train:
            self.dir_hr = self.apath / 'train' / 'HR'
            self.dir_lr = self.apath / 'train' / 'LR_bicubic'
        else:
            self.dir_hr = self.apath / 'test' / 'HR'
            self.dir_lr = self.apath / 'test' / 'LR_bicubic'     

        assert self.dir_hr.exists(), "HR input data path does not exist!"
        assert self.dir_lr.exists(), "LR input data path does not exist!"

        self.ext = ("jpg", "jpg")