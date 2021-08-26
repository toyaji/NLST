import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.image import PSNR, SSIM

from model.NLST import NLST
from data.dataloader import MSDataLoader


class LitModel(pl.LightningModule):
    def __init__(self, model_params, loader_params) -> None:
        super().__init__()
        
        self.model = NLST(model_params)
        self.loader_params = loader_params

        self.train_metrics = MetricCollection([PSNR(), SSIM()])
        self.valid_metrics = MetricCollection([PSNR(), SSIM()])

    def configure_optimizers(self):
        # TODO params 분리되 되는듯... 여기다가 앞에 붙이는거 붙여되 되겠네
        optimazier = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimazier

    def training_step(self, batch, batch_idx):
        x, y = batch
        # my dataloader get 6 pairs of image crops per iteration
        if len(x.size()) > 4:
            b, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            y = y.view(-1, c, h, w)

        sr = self.model(x)
        loss = F.mse_loss(sr, y)
        self.train_metrics(sr, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_metrics', self.train_metrics, on_epoch=True, prog_bar=True, logger=True)
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
        self.valid_metrics(sr, y)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_metrics', self.train_metrics, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def set_dataset(self, train_set, val_set, test_set):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        # TODO dataloader 를 param 받도록 수정해야함.
        dataloader = MSDataLoader(self.train_set, self.loader_params)
        return dataloader

    def val_dataloader(self):
        dataloader = MSDataLoader(self.val_set, self.loader_params)
        return dataloader

    def test_dataloader(self):
        dataloader = MSDataLoader(self.test_set, self.loader_params)
        return dataloader


