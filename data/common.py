import random
import rawpy
import numpy as np
import skimage.color as sc
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn



def get_patch(*args, patch_size=96, scale=2, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def get_random_patch(hr, lr, patch_size):
    # some input images have little bit different size so we need to consider that
    ih1, iw1 = hr.shape[:2]
    ih2, iw2  =lr.shape[:2]
    ih = min(ih1, ih2)
    iw = min(iw1, iw2)

    # get patch by random crop
    tp = patch_size
    ix = np.random.randint(0, iw - patch_size)
    iy = np.random.randint(0, ih - patch_size)
    hr = hr[iy:iy + tp, ix:ix + tp, :]
    lr = lr[iy:iy + tp, ix:ix + tp, :]
    return hr, lr
        
def get_random_patches(hr, lrs, patch_size):
    """Get patches of different random fov for each scale of image"""   
    hrs = []
    for i, lr in enumerate(lrs):
        h, l = get_random_patch(hr, lr, patch_size)
        hrs.append(h)
        lrs[i] = l
    return hrs, lrs

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()

        # normalization for rgb range
        if rgb_range == 255:
            tensor.mul_(1 / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

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

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

def readFocal_pil(image_path, focal_code=37386):
    if isinstance(image_path, Path):
        image_path = str(image_path)
    try:
        img = Image.open(image_path)
    except:
        print(image_path)
        return None
    exif_data = img._getexif()
    img.close()
    return float(exif_data[focal_code])

def crop_fov(image, ratio, buffer=1.):
    width, height = image.shape[:2]
    new_width = width * ratio * buffer
    new_height = height * ratio * buffer
    left = np.ceil((width - new_width)/2.)
    top = np.ceil((height - new_height)/2.)
    right = np.floor((width + new_width)/2.)
    bottom = np.floor((height + new_height)/2.)
    # print("Cropping boundary: ", top, bottom, left, right)
    cropped = image[int(left):int(right), int(top):int(bottom), ...]
    return cropped

# zoom-learn-zoom/utils.py
def get_bayer(path, black_lv=512, white_lv=16383):
    if isinstance(path, str) or isinstance(path, Path):
        raw = rawpy.imread(str(path))
    else:
        raw = path
    bayer = raw.raw_image_visible.astype(np.float32)
    bayer = (bayer - black_lv) / (white_lv - black_lv)  # subtract the black level
    return bayer

def get_4ch(bayer):
    h, w = bayer.shape[:2]
    rgba = np.zeros((h // 2, w // 2, 4), dtype=np.float32)
    rgba[:, :, 0] = bayer[0::2, 0::2]  # R
    rgba[:, :, 1] = bayer[1::2, 0::2]  # G1
    rgba[:, :, 2] = bayer[1::2, 1::2]  # B
    rgba[:, :, 3] = bayer[0::2, 1::2]  # G2

    return rgba

def get_1ch(raw):
    h, w = raw.shape[:2]
    bayer = np.zeros((h * 2, w * 2), dtype=raw.dtype)
    bayer[0::2, 0::2] = raw[..., 0]
    bayer[1::2, 0::2] = raw[..., 1]
    bayer[1::2, 1::2] = raw[..., 2]
    bayer[0::2, 1::2] = raw[..., 3]

    return bayer