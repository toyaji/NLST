import math

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import PSNR, SSIM
from torchvision.transforms import CenterCrop
from torchvision.transforms.transforms import Scale

from importlib import import_module


class LitModel(pl.LightningModule):
    def __init__(self, model_params, opt_params) -> None:
        super().__init__()

        # set opt params
        self.scale = model_params.scale
        self.lr = opt_params.learning_rate
        self.weight_decay = opt_params.weight_decay
        self.patience = opt_params.patience
        self.factor = opt_params.factor
        self.b = 16    # boundary for psnr, ssim

        # load the model
        module = import_module('model.' + model_params.net.lower())
        self.model = getattr(module, model_params.net)(model_params)

        # set metrices to evaluate performence
        self.val_psnr = PSNR(data_range=1, reduction='sum')
        self.test_psnr = PSNR(data_range=1, reduction='sum')
        self.val_ssim = SSIM(reduction='sum')
        self.test_ssim = SSIM(reduction='sum')
        
        # save hprams for log
        self.save_hyperparameters(model_params)
        self.save_hyperparameters(opt_params)

    def forward(self, x):
        return self.model(x)

    def forward_chop(self, x, shave=10, min_size=160000):
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
            'monitor': "valid_loss",
            'name': 'leraning_rate'
        }
        return [optimazier], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        sr = self(x)
        loss = F.mse_loss(sr, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        sr = self(x)
        loss = F.mse_loss(sr, y)
        psnr = self.calc_psnr(sr, y, self.scale, 1)
        psnr2 = self.val_psnr(sr, y)
        ssim = self.val_ssim(sr, y)
        self.log('valid_loss', loss, prog_bar=True, logger=True)
        self.log('valid_psnr1', psnr, prog_bar=True, logger=True)
        self.log('valid_psnr2', psnr2, prog_bar=True, logger=True)
        self.log('valid_ssim', ssim, prog_bar=True, logger=True)
        return loss, psnr, ssim

    def test_step(self, batch, batch_idx):
        # TODO 평가 채널 하나로 어케 하는지 참고해서 반영하기
        x, y, _ = batch
        sr = self.forward_chop(x)
        psnr = self.calc_psnr(sr, y, self.scale, 1)
        ssim = self.test_ssim(sr, y)
        self.log('test_psnr', psnr, prog_bar=True, logger=True)
        self.log('test_ssim', ssim, prog_bar=True, logger=True)
        return psnr, ssim

    @staticmethod
    def calc_psnr(sr, hr, scale, rgb_range):
        diff = (sr - hr) / rgb_range
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)

        valid = diff[..., shave:-shave, shave:-shave]
        mse = valid.pow(2).mean()

        return -10 * math.log10(mse)