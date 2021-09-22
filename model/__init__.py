from model.NLRN import NLRN
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import PSNR, SSIM

from model.NLST import NLST
from model.NLRN_corr import NLRN as NLRN_corr


class LitModel(pl.LightningModule):
    def __init__(self, model_params, opt_params) -> None:
        super().__init__()

        # set opt params
        self.lr = opt_params.learning_rate
        self.weight_decay = opt_params.weight_decay
        self.patience = opt_params.patience
        self.factor = opt_params.factor

        # load the model
        if model_params.net == 'NLST':
            self.model = NLST(**model_params)
        elif model_params.net == 'NLRN':
            self.model = NLRN_corr(**model_params)

        # set metrices to evaluate performence
        self.train_psnr = PSNR()
        self.train_ssim = SSIM()
        self.valid_psnr = PSNR()
        self.valid_ssim = SSIM()
        self.test_psnr = PSNR()
        self.test_ssim = SSIM()
        
        # save hprams for log
        self.save_hyperparameters(model_params)
        self.save_hyperparameters(opt_params)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimazier = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimazier, factor=self.factor, patience=self.patience),
            'monitor': "valid_loss",
            'name': 'leraning_rate'
        }
        return [optimazier], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        # my dataloader get 6 pairs of image crops per iteration
        if len(x.size()) > 4:
            b, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            y = y.view(-1, c, h, w)

        sr = self.model(x)
        loss = F.mse_loss(sr, y)
        self.train_psnr(sr, y)
        self.train_ssim(sr, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_psnr', self.train_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_ssim', self.train_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # my dataloader get 6 pairs of image crops per iteration
        if len(x.size()) > 4:
            b, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            y = y.view(-1, c, h, w)

        sr = self.model(x)
        loss = F.mse_loss(sr, y)
        self.valid_psnr(sr, y)
        self.valid_ssim(sr, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_psnr', self.valid_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_ssim', self.valid_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # TODO benchmark 에다가 해볼 수 있게 변형
        x, y = batch
        # my dataloader get 6 pairs of image crops per iteration
        if len(x.size()) > 4:
            b, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            y = y.view(-1, c, h, w)

        sr = self.model(x)
        loss = F.mse_loss(sr, y)
        self.test_psnr(sr, y)
        self.test_ssim(sr, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_psnr', self.test_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_ssim', self.test_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss