import warnings
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model import LitModel
from data import LitDataset

warnings.filterwarnings('ignore')


def main(config):

    # datamodule settting
    dm = LitDataset(**config.dataset)

    # load pytorch lightning model
    model = LitModel(config.model, config.optimizer, config.dataset)

    # instantiate trainer
    logger = TensorBoardLogger('logs/', **config.log)

    if config.log.log_graph:
        logger.log_graph(model, torch.zeros(1, 3, 64, 64).cuda())

    checkpoint_callback = ModelCheckpoint(monitor="valid/loss", save_top_k=config.callback.save_top_k)
    early_stop_callback = EarlyStopping(monitor="valid/loss", patience=config.callback.earlly_stop_patience)
    lr_callback = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(logger=logger, 
                      callbacks=[checkpoint_callback, early_stop_callback, lr_callback], 
                      **config.trainer
                      )
    
    # start training!
    if not config.dataset.test_only:
        trainer.fit(model, dm)
    
    trainer.test(model, datamodule=dm)

    
if __name__ == "__main__":
    from options import load_config_from_args
    config = load_config_from_args()
    main(config)