import warnings
from pathlib import Path

import torch
from torch.utils.data.dataset import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from model import LitModel
from data import LitDataset

warnings.filterwarnings('ignore')


def main(config):

    # datamodule settting
    dm = LitDataset(**config.dataset)

    # load pytorch lightning model - TODO 요 부분 argparser 로 모델명 받게하기
    model = LitModel(config.model, config.optimizer)

    # instantiate trainer
    logger = TensorBoardLogger('logs/', log_graph=True, **config.log)
    logger.log_graph(model, torch.zeros(1, 3, 64, 64).cuda())

    # callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=3)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=20)
    
    # profiling for memory usage
    #profiler=PyTorchProfiler(sort_by_key="cuda_memory_usage")

    trainer = Trainer(logger=logger, 
                      callbacks=[checkpoint_callback, early_stop_callback], 
                      **config.trainer)
    
    # start training!
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)

    
if __name__ == "__main__":
    from options import load_config
    config = load_config("config/base_template.yaml")
    main(config)