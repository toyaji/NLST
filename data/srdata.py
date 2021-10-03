import cv2
from pathlib import Path
from torch.utils.data import Dataset

from . import common


class SRData(Dataset):        
    def __init__(self, dir, scale, name='', train=True, patch_size=48, n_colors=3, 
                 rgb_range=1, augment=True, benchmark=False, input_large=False, **kwargs):

        super().__init__()
        self.name = name
        self.scale = scale
        self.train = train
        self.patch_size = patch_size
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.augment = augment
        self.benchmark = benchmark
        self.input_large = input_large
        
        self._set_filesystem(dir)
        self._scan()

    def _scan(self):
        # FIXME pathlib glob library don't distingish between capital and lower in Window but not in Linux.
        self.hr_pathes = sorted(list(self.dir_hr.glob("*." + self.ext[0])))
        self.lr_pathes = sorted(list((self.dir_lr / "x{:1d}".format(self.scale)).glob("*." + self.ext[1])))
        
        assert len(self.hr_pathes) > 0 or len(self.lr_pathes) > 0, "Can't read the data properly! Check the dir or data."

    def _set_filesystem(self, data_dir):
        if isinstance(data_dir, str):
            self.apath = Path(data_dir) / self.name
        elif isinstance(data_dir, Path):
            self.apath = data_dir / self.name

        assert self.apath.exists(), "Given base data dir path is wrong!: {}".format(self.apath)
        
        self.dir_hr = self.apath / 'HR' 
        self.dir_lr = self.apath / 'LR_bicubic'
        self.ext = ('png', 'png')

        # TODO 이거 어떻게 쓰는건지... 확인 필요
        if self.input_large: self.dir_lr += 'L'

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.rgb_range)
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        return len(self.hr_pathes)

    def _load_file(self, idx):
        f_hr = self.hr_pathes[idx]
        f_lr = self.lr_pathes[idx]
        filename = f_hr.name
        hr = cv2.imread(str(f_hr))
        lr = cv2.imread(str(f_lr))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)    
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale
        if self.train:
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