import cv2
import numpy as np
from data import common
import torch
from pathlib import Path
from torch.utils.data import Dataset

#from data import common

class ZoomLZoomData(Dataset):
    """
    This data class return data pair which consist of 7 images having different resolution  
    
    """
    def __init__(self, data_dir, from_dir, scale_idx, patch_size, img_ext, train: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.scale_idx = scale_idx
        self.get_from_dir = from_dir
        self.img_ext = img_ext
        self.train = train
        self._set_filesystem(data_dir)

    def __getitem__(self, idx):
        lrs = self._scan(idx, self.scale_idx)
        hr = lrs.pop(0)
        # get patch or just give full size image
        if self.patch_size == -1:
            hrs = [hr] * len(lrs)
        else:
            hrs, lrs = common.get_random_patches(hr, lrs, self.patch_size)
        hrs = torch.stack(common.np2Tensor(hrs, 1)) # size -> (6, 3, H, W)
        lrs = torch.stack(common.np2Tensor(lrs, 1))
        return hrs, lrs

    def __len__(self):
        return len(self.base_paths)

    def _set_filesystem(self, data_dir):
        if isinstance(data_dir, str):
            self.apath = Path(data_dir)
        if self.train:
            self.base_paths = sorted(list((self.apath / "train").glob("*")))
        else:
            self.base_paths = sorted(list((self.apath / "test").glob("*")))
    
    def _scan(self, idx, scale_idx=[1, 2]):
        base_path = self.base_paths[idx] / self.get_from_dir
        imgs_path = [base_path / "{:05d}.{}".format(i, self.img_ext) for i in scale_idx]
        lrs = [cv2.imread(str(path)) for path in imgs_path]
        lrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in lrs]
        return lrs

    def _get_focalscale(self, idx):
        ref_paths = self.base_paths[idx].glob("*.JPG")
        focals = [common.readFocal_pil(p) for p in ref_paths]
        focal_scale = np.array(focals) / 240
        return focal_scale

