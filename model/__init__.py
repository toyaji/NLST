import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import PSNR, SSIM

from model.NLST import NLST


class LitModel(pl.LightningModule):
    def __init__(self, model_params, opt_params) -> None:
        super().__init__()

        # set opt params
        self.lr = opt_params.learning_rate
        self.weight_decay = opt_params.weight_decay

        # load the model
        self.model = NLST(**model_params)

        # set metrices to evaluate performence
        # TODO 다른 metrics 추가해야함... 모듈 만들던지 해서
        self.train_psnr = PSNR()
        self.train_ssim = SSIM()
        self.valid_psnr = PSNR()
        self.valid_ssim = SSIM()
        
        # save hprams for log
        self.save_hyperparameters(model_params)
        self.save_hyperparameters(opt_params)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # TODO params 분리되 되는듯... 여기다가 앞에 CNN gep 붙이는거 붙여되 되겠네
        optimazier = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimazier, patience=7),
            'monitor': "val_loss",
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