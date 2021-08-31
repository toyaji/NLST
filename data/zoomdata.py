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
    def __init__(self, args, train: bool = True) -> None:
        super().__init__()
        self.patch_size = args.patch_size
        self.scale_idx = args.scale_idx
        self.train = train
        self._set_filesystem(args.dir_data)

    def __getitem__(self, idx):
        lrs = self._scan(idx, self.scale_idx)
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
    
    def _scan(self, idx, scale_idx=[1, 2]):
        # FIXME 일단 align 된 JPG 로 하고 그 다음에 JPG align 하면서 되나 보고, 최종 ARW 로 align 하면서 되나 보기
        base_path = self.base_paths[idx] / "aligned"
        imgs_path = [base_path / "{:05d}.JPG".format(i) for i in scale_idx]
        lrs = [cv2.imread(str(path)) for path in imgs_path]
        return lrs

    def _get_focalscale(self, idx):
        ref_paths = self.base_paths[idx].glob("*.JPG")
        focals = [common.readFocal_pil(p) for p in ref_paths]
        focal_scale = np.array(focals) / 240
        return focal_scale

