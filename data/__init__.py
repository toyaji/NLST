from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from typing import Optional

from .div2k import DIV2K
from .zoomdata import ZoomLZoomData

class LitDataset(LightningDataModule):
    def __init__(self,
                 dir,
                 data='zoom',
                 patch_size=64,
                 batch_size=4,
                 scale_idx=2,
                 shuffle=True,
                 num_workers=4, 
                 train_transforms=None, 
                 val_transforms=None, 
                 test_transforms=None):
        super().__init__(train_transforms=train_transforms, 
                         val_transforms=val_transforms, 
                         test_transforms=test_transforms)

        self.data = data
        self.dir = dir
        self.scale_idx = scale_idx
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        #self.save_hyperparameters()

    def setup(self, stage: Optional[str]=None) -> None:
        if self.data == 'div':
            train_set = DIV2K(self.dir, self.scale_idx, self.patch_size)
            self.test_set = DIV2K(self.dir, self.scale_idx, self.patch_size, train=False)
        elif self.data == 'zoom':
            train_set = ZoomLZoomData(self.dir, 'cropped', [1, self.scale_idx], self.patch_size, img_ext='JPG')
            self.test_set = ZoomLZoomData(self.dir, 'cropped', [1, self.scale_idx], self.patch_size, img_ext='JPG', train=False)
        
        length = [round(len(train_set)*0.8), round(len(train_set)*0.2)]
        self.train_set, self.val_set = random_split(train_set, length)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, self.shuffle, num_workers=self.num_workers)