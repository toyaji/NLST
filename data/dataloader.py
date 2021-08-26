from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter


class MSDataLoader(DataLoader):
    # FIXME dataloader 를 param 받도록 수정해야함.
    def __init__(
        self, dataset, batch_size, shuffle: bool, num_workers: int, 
        sampler=None, batch_sampler=None, collate_fn=default_collate, 
        pin_memory: bool = False, drop_last: bool = True):

        super(MSDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, 
            sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, 
            collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, )

    def __iter__(self):
        return _MultiProcessingDataLoaderIter(self)