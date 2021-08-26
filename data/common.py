import random

import numpy as np
import skimage.color as sc
from PIL import Image
from pathlib import Path
import torch
from torch._C import ErrorReport


def get_identical_patches(imgs, patch_size):
    """Get patches of same fov from all scales of images"""
    ih, iw = imgs[0].shape[:2]
    tp = patch_size
    ix = np.random.randint(0, iw - patch_size)
    iy = np.random.randint(0, ih - patch_size)
    imgs = []
    for i in range(len(imgs)):
        imgs.append(imgs[i][iy:iy + tp, ix:ix + tp, :])
    return imgs

def get_random_patches(hr, lrs, patch_size):
    """Get patches of different random fov for each scale of image"""
    def _get_random_patch(hr, lr, patch_size):
        ih, iw = hr.shape[:2]
        tp = patch_size
        ix = np.random.randint(0, iw - patch_size)
        iy = np.random.randint(0, ih - patch_size)
        hr = hr[iy:iy + tp, ix:ix + tp, :]
        lr = lr[iy:iy + tp, ix:ix + tp, :]
        return hr, lr
    
    hrs = []
    for i, lr in enumerate(lrs):
        h, l = _get_random_patch(hr, lr, patch_size)
        hrs.append(h)
        lrs[i] = l
    return hrs, lrs

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise != '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]

def readFocal_pil(image_path, focal_code=37386):
    if isinstance(image_path, Path):
        image_path = str(image_path)
    try:
        img = Image.open(image_path)
    except:
        raise ErrorReport("Can't open image!")
    exif_data = img._getexif()
    img.close()
    return float(exif_data[focal_code])