from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset

from typing import Optional
from importlib import import_module

class LitDataset(LightningDataModule):
    def __init__(self,
                 args,
                 data='zoom',
                 batch_size=4,
                 shuffle=True,
                 num_workers=4,
                 test_only=False,
                 **kwargs):
                 
        super().__init__()

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.test_only = test_only
        self.test_data = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']

        # args for dataset
        self.args = args

    def setup(self, stage: Optional[str]=None) -> None:
        if not self.test_only:
            datasets = []
            for d in self.data:
                m = import_module('data.' + d.lower())
                datasets.append(getattr(m, d)(**self.args, name=d))

        testsets = []
        for d in self.test_data:
            m = import_module('data.benchmark')
            testsets.append(getattr(m, 'Benchmark')(**self.args, train=False, name=d))

        self.train_set = ConcatDataset(datasets)
        self.val_set = ConcatDataset(testsets)
        self.test_set = ConcatDataset(testsets)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, self.shuffle, num_workers=self.num_workers)