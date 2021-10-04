from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset

from typing import Optional
from importlib import import_module

class LitDataset(LightningDataModule):
    def __init__(self,
                 args,
                 train_data=['DIV2K'],
                 test_data=['BSD100'],
                 batch_size=4,
                 shuffle=True,
                 num_workers=4,
                 **kwargs):
                 
        super().__init__()

        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # args for dataset
        self.args = args

    def setup(self, stage: Optional[str]=None) -> None:

        if stage == "fit":
            trainsets = []
            for d in self.train_data:
                m = import_module('data.' + d.lower())
                trainsets.append(getattr(m, d)(**self.args, name=d))
            valsets = []
            for d in self.train_data:
                m = import_module('data.' + d.lower())
                valsets.append(getattr(m, d)(**self.args, train=False, name=d))

            self.train_set = ConcatDataset(trainsets)
            self.val_set = ConcatDataset(valsets)           

        if stage == "test":
            testsets = []
            for d in self.test_data:
                m = import_module('data.benchmark')
                testsets.append(getattr(m, 'Benchmark')(**self.args, train=False, name=d))

            self.test_set = testsets

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, self.shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        testloaders = [DataLoader(data, 1, self.shuffle, num_workers=self.num_workers) for data in self.test_set]
        return testloaders