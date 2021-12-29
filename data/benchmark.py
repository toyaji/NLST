from pathlib import Path
from .srdata import SRData


class Benchmark(SRData):
    # 'train=False' is base setting for benchmark. This is make dataloader do not crop the patch
    def __init__(self, dir, scale, name, train=False, patch_size=-1, rgb_range=1, augment=False, **kwargs):

        super(Benchmark, self).__init__(dir=dir, scale=scale, name=name, train=train, patch_size=-1, 
                                    n_colors=3, rgb_range=rgb_range, augment=augment)

    def _set_filesystem(self, data_dir):
        data_dir = Path(data_dir) / 'benchmark'

        super(Benchmark, self)._set_filesystem(data_dir)