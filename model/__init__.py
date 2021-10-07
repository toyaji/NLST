import os
import math
from pathlib import Path
import cv2
import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.metrics.functional import ssim as _ssim
from importlib import import_module


CHOP_SIZE = {"HAN" : 160000, "CSNLN" : 2500}

class LitModel(pl.LightningModule):
    def __init__(self, model_params, opt_params, test_data) -> None:
        super().__init__()

        # set opt params
        self.name = model_params.net
        self.scale = model_params.scale
        self.lr = opt_params.learning_rate
        self.weight_decay = opt_params.weight_decay
        self.patience = opt_params.patience
        self.factor = opt_params.factor
        self.test_data = test_data
        self.chop_size = CHOP_SIZE[self.name]

        # load the model
        module = import_module('model.' + self.name.lower())
        self.model = getattr(module, self.name)(model_params)

        # pretrain set
        if model_params.pretrain and hasattr(self.model, 'load_state_dict'):
            try:
                path = 'pretrain/{name}/{name}_X{scale}.pt'.format(name=self.name, scale=self.scale)
                dicts = torch.load(path)
                self.model.load_state_dict(dicts)
            finally: 
                print("Loading pretrained model got error. It will start without pretrained weight.")
                pass
        
        # save hprams for log
        self.save_hyperparameters(model_params)
        self.save_hyperparameters(opt_params)

    def forward(self, x):
        return self.model(x)

    def forward_chop(self, x, shave=10, min_size=160000):
        # TODO min_size 받는 부분 고민 좀 필요함...
        # it work for only batch size of 1
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, 1):
                sr_batch = self.model(lr_list[i])
                sr_list.extend(sr_batch)

        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def configure_optimizers(self):
        optimazier = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimazier, factor=self.factor, patience=self.patience),
            'monitor': "valid/loss",
            'name': 'leraning_rate'
        }
        return [optimazier], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        sr = self(x)
        loss = F.mse_loss(sr, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        sr = self(x)
        loss = F.mse_loss(sr, y)
        psnr = self._psnr(sr, y, self.scale, 1)
        ssim = _ssim(sr, y)
        self.log('valid/loss', loss, prog_bar=True)
        self.log('valid/psnr', psnr, prog_bar=True)
        self.log('valid/ssim', ssim, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, filename = batch
        dataset_name = self.test_data[dataloader_idx]
        sr = self.forward_chop(x, min_size=self.chop_size)
        self._img_save(sr.clone().detach(), filename[0], dataset_name)
        sr = self._quantize(sr, 1)
        psnr = self._psnr(sr, y, self.scale, 1)
        ssim = _ssim(sr, y)
        
        self.log('test/{}/psnr'.format(dataset_name), psnr, prog_bar=True)
        self.log('test/{}/ssim'.format(dataset_name), ssim, prog_bar=True)
        return psnr, ssim

    @staticmethod
    def _img_save(sr, filename, dataset_name):
        if isinstance(sr, Tensor):
            img = sr.mul(255).cpu().numpy()

        base = Path("results")
        base.mkdir(exist_ok=True)
        save_path = base / dataset_name
        save_path.mkdir(exist_ok=True)

        cv2.imwrite(str(save_path / "{}".format(filename)), img[0].transpose(1,2,0))

    @staticmethod
    def _psnr(sr, hr, scale, rgb_range):
        # TODO make into custom metrics class 
        diff = (sr - hr) / rgb_range
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)

        valid = diff[..., shave:-shave, shave:-shave]
        mse = valid.pow(2).mean()

        return -10 * math.log10(mse)

    @staticmethod
    def _quantize(img, rgb_range):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)