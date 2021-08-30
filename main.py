import warnings
from pathlib import Path

from torch.utils.data.dataset import random_split
from torchviz import make_dot
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import LitModel
from data.zoomdata import ZoomLZoomData

warnings.filterwarnings('ignore')


def main(config):

    # dataset settting
    train_set = ZoomLZoomData(config.dataset, train=True)
    test_set = ZoomLZoomData(config.dataset, train=False)

    length = [round(len(train_set)*0.8), round(len(train_set)*0.2)]
    train_set, val_set = random_split(train_set, length)

    # load pytorch lightning model
    model = LitModel(config.model, config.dataloader)
    model.set_dataset(train_set, val_set, test_set)

    # instantiate trainer
    logger = TensorBoardLogger('logs/', 'first_attemps')
    trainer = Trainer(gpus=1,
                      max_epochs=2,
                      progress_bar_refresh_rate=1, 
                      logger=logger, 
                      log_every_n_steps=1,
                      gradient_clip_val=0.5,
                      )
    
    # start training!
    trainer.fit(model)

    
if __name__ == "__main__":
    from options import load_config
    config = load_config("config/first_test.yaml")
    main(config)