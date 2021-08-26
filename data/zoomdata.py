import cv2
import numpy as np
import common
import torch
from pathlib import Path
from torch.utils.data import Dataset

#from data import common

class ZoomLZoomData(Dataset):
    """
    This data class return data pair which consist of 7 images having different resolution  
    
    """
    def __init__(self, args, train: bool = True) -> None:
        super().__init__()
        self.patch_size = args.patch_size
        self.train = train
        self._set_filesystem(args.dir_data)

    def __getitem__(self, idx):
        lrs = self._scan(idx)
        hr = lrs.pop(0)
        hrs, lrs = common.get_random_patches(hr, lrs, self.patch_size)
        hrs = torch.stack(common.np2Tensor(hrs, 255)) # size -> (6, 3, H, W)
        lrs = torch.stack(common.np2Tensor(lrs, 255))
        return hrs, lrs

    def __len__(self):
        return len(self.base_paths)

    def _set_filesystem(self, dir_data):
        if isinstance(dir_data, str):
            self.apath = Path(dir_data)
        if self.train:
            self.base_paths = sorted(list((self.apath / "train").glob("*")))
        else:
            self.base_paths = sorted(list((self.apath / "test").glob("*")))
    
    def _scan(self, idx):
        imgs_path = (self.base_paths[idx] / "aligned").glob("*")
        lrs = [cv2.imread(str(p)) for p in imgs_path]
        return lrs

    def _get_focalscale(self, idx):
        ref_paths = self.base_paths[idx].glob("*.JPG")
        focals = [common.readFocal_pil(p) for p in ref_paths]
        focal_scale = np.array(focals) / 240
        return focal_scale

