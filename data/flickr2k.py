from .srdata import SRData
from . import common

class Flickr2K(SRData):
    def __init__(self, dir, scale, name='Flickr2K', train=True, patch_size=48, rgb_range=1, augment=True, **kwargs):

        super(Flickr2K, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=patch_size, 
                                    n_colors=3, rgb_range=rgb_range, augment=augment)

    def _scan(self):
        if self.train:
            self.hr_pathes = sorted(list(self.dir_hr.glob("*")))[:2500]
            self.lr_pathes = sorted(list((self.dir_lr / "X{:1d}".format(self.scale)).glob("*")))[:2500]
        else:
            self.hr_pathes = sorted(list(self.dir_hr.glob("*")))[2500:]
            self.lr_pathes = sorted(list((self.dir_lr / "X{:1d}".format(self.scale)).glob("*")))[2500:] 