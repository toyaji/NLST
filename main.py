import warnings
from pathlib import Path

from torch.utils.data.dataset import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import LitModel
from data.zoomdata import ZoomLZoomData

warnings.filterwarnings('ignore')


def main(config):

    # dataset settting
    base = Path('./data/SRRAW/')
    assert base.exists()

    train_set = ZoomLZoomData(config.dataset, train=True)
    test_set = ZoomLZoomData(config.dataset, train=False)

    length = [round(len(train_set)*0.8), round(len(train_set)*0.2)]
    train_set, val_set = random_split(train_set, length)

    # load pytorch lightning model
    model = LitModel(config.model, config.dataloader)
    model.set_dataset(train_set, val_set, test_set)

    # instantiate trainer
    logger = TensorBoardLogger('logs/', 'SRraw_by_Paul')
    trainer = Trainer(gpus=1,
                      max_epochs=2,
                      progress_bar_refresh_rate=1, 
                      logger=logger, log_every_n_steps=1)
    
    # start training!
    trainer.fit(model)


if __name__ == "__main__":
    main()