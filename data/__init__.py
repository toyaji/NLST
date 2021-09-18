from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, ConcatDataset
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
                 scale_idx=[2, 3, 4],
                 shuffle=True,
                 num_workers=4, 
                 **kwargs):
        super().__init__()

        self.data = data
        self.dir = dir
        self.scale_idx = scale_idx
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # settings for zoom data
        self.kwargs = kwargs

    def setup(self, stage: Optional[str]=None) -> None:
        if self.data == 'div':
            train_sets = [DIV2K(self.dir, i, self.patch_size) for i in self.scale_idx]
            val_sets = [DIV2K(self.dir, i, self.patch_size, train=False) for i in self.scale_idx]
        elif self.data == 'zoom':
            train_sets = [ZoomLZoomData(self.dir, (1, i), self.patch_size, **self.kwargs) for i in self.scale_idx]
            val_sets = [ZoomLZoomData(self.dir, (1, i), self.patch_size, train=False, **self.kwargs) for i in self.scale_idx]

        self.train_set = ConcatDataset(train_sets)
        self.val_set = ConcatDataset(val_sets)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    # TODO benchmark test 하도록 여 부분 수정하기
    def test_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, self.shuffle, num_workers=self.num_workers)