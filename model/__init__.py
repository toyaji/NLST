import os
import math
from pathlib import Path
import cv2
import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR
from torchmetrics import SSIM, PSNR
from importlib import import_module

# You can adjust following chop size fit to your GPU memory.
CHOP_SIZE = {"HAN" : 300000, "CSNLN" : 25000, "NLST" : 10000, "SCAN": 200000}

class LitModel(pl.LightningModule):
    def __init__(self, model_params, opt_params, sch_params, data_params) -> None:
        super().__init__()

        # set opt params
        self.name = model_params.net
        self.scale = model_params.scale
        self.rgb_range = model_params.rgb_range
        self.chop_size = CHOP_SIZE[self.name] if model_params.chop_size is None else model_params.chop_size

        self.opt_params = opt_params
        self.sch_params = sch_params

        self.test_data = data_params.test_data
        self.save_test_img = data_params.save_test_img

        # load the model
        module = import_module('model.' + self.name.lower())
        self.model = getattr(module, self.name)(model_params)

        # pretrain set
        if model_params.pretrain and hasattr(self.model, 'load_state_dict'):
            try:
                path = 'pretrain/{name}/{name}_X{scale}.pt'.format(name=self.name, scale=self.scale)
                dicts = torch.load(path)
                msg = self.model.load_state_dict(dicts)
                print("Loading pretrained stat dict: ", msg)
            except Exception as e: 
                print(e, "Loading pretrained stat dict: Cannot find the pretrained file or faile to load.")
                pass
        else:
            print("Training starts from the scratch without pretrain state dict.")
        
        # save hprams for log
        self.save_hyperparameters(model_params)
        self.save_hyperparameters(opt_params)

        # loss
        if model_params.loss == 'l1':
            self.loss = F.l1_loss
        elif model_params.loss == 'l2':
            self.loss = F.mse_loss
        elif model_params.loss == 'cosine':
            self.loss = F.cosine_similarity

        # metrics
        self.val_ssim = SSIM()
        self.test_ssim = SSIM()

    def forward(self, x):
        return self.model(x)

    def forward_chop(self, x, auto_shave=True, shave_feed=16, shave=(10, 10), min_size=160000):
        # it work for only batch size of 1
        scale = self.scale
        h_shave, w_shave = shave
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2

        if auto_shave:
            h_shave = (round(h_half / shave_feed) + 1) * shave_feed - h_half
            w_shave = (round(w_half / shave_feed) + 1) * shave_feed - w_half

        h_size, w_size = h_half + h_shave, w_half + w_shave

        #print("Size of sliced input: ", h_shave, w_shave)

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            lr_batch = torch.cat(lr_list)
            sr_batch = self.model(lr_batch)
        else:
            sr_batch = torch.cat([
                self.forward_chop(patch, min_size=min_size) \
                for patch in lr_list
            ], dim=0)

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] = sr_batch[0:b, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_batch[b:b*2, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] = sr_batch[b*2:b*3, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_batch[b*3:b*4, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def configure_optimizers(self):
        if self.opt_params.name == 'adam':
            optimizer = Adam(self.parameters(), lr=self.opt_params.learning_rate)
        elif self.opt_params.name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.opt_params.learning_rate, 
                               weight_decay=self.opt_params.weight_decay)
        else:
            optimizer = SGD(self.parameters(), lr=self.opt_params.learning_rate, 
                             momentum=self.opt_params.momentum, weight_decay=self.opt_params.weight_decay)

        if self.sch_params.name == 'multistep':
            scheduler = MultiStepLR(optimizer, milestones=self.sch_params.multistep, gamma=self.sch_params.factor)
        elif self.sch_params.name == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.sch_params.period, eta_min=self.sch_params.min_lr)
        elif self.sch_params.name == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, factor=self.sch_params.factor, patience=self.sch_params.patience,
                                           cooldown=self.sch_params.cooldown, min_lr=self.sch_params.min_lr),
        
        lr_scheduler = {
            'scheduler': scheduler,
            'monitor': "valid/loss",
            'name': 'leraning_rate'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        sr = self(x)
        loss = self.loss(sr, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        sr = self(x)
        loss = self.loss(sr, y)
        sr = self._quantize(sr, self.rgb_range)
        psnr = self._psnr(sr, y, self.scale, self.rgb_range)
        ssim = self.val_ssim(sr, y)

        self.log('valid/loss', loss, prog_bar=True)
        self.log('valid/psnr', psnr, prog_bar=True)
        self.log('valid/ssim', ssim, prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:
        # to prevent memory leak 
        self.val_ssim.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, filename = batch
        dataset_name = self.test_data[dataloader_idx]
        sr = self.forward_chop(x, min_size=self.chop_size)
        sr = self._quantize(sr, self.rgb_range)
        psnr = self._psnr(sr, y, self.scale, self.rgb_range)
        ssim = self.test_ssim(sr, y)

        self.log('test/{}/psnr'.format(dataset_name), psnr, prog_bar=True)
        self.log('test/{}/ssim'.format(dataset_name), ssim, prog_bar=True)

        if self.save_test_img:
            self._img_save(sr.clone().detach(), filename[0], dataset_name)

        return psnr, ssim

    def test_epoch_end(self, outputs) -> None:
        # to prevent memory leak 
        self.test_ssim.reset()

    @staticmethod
    def _img_save(sr, filename, dataset_name):
        if isinstance(sr, Tensor):
            sr = sr.mul(255).cpu().numpy()

        base = Path("results")
        base.mkdir(exist_ok=True)
        save_path = base / dataset_name
        save_path.mkdir(exist_ok=True)

        sr = sr.transpose(0, 2, 3, 1)
        for img in sr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path / filename), img)

    @staticmethod
    def _psnr(sr, hr, scale, rgb_range):
        # This psnr function is old legacy from other researches. 
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