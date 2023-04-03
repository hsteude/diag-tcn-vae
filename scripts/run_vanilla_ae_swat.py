from diag_vae.vanilla_tcn_ae import VanillaTcnAE
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from diag_vae.swat_data_module import SwatDataModule
import diag_vae.constants as const


if __name__ == "__main__":
    SEQ_LEN = 1000
    LATEND_DIM = 10

    dm = SwatDataModule(
        data_path_train=const.SWAT_TRAIN_PATH,
        data_path_val=const.SWAT_VAL_PATH,
        seq_len=SEQ_LEN,
        cols=const.SWAT_SENSOR_COLS,
        symbols_dct=const.SWAT_SYMBOLS_MAP,
        dl_workers=8,
    )
    model = VanillaTcnAE(
        in_dim=len(const.SWAT_SENSOR_COLS),
        latent_dim=LATEND_DIM,
        seq_len=SEQ_LEN,
    )
    logger = TensorBoardLogger("logs", name="vanilla_tcn_ae", default_hp_metric=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 
    trainer = pl.Trainer(
        max_epochs=500, accelerator=accelerator, devices=devices, logger=logger
    )
    trainer.fit(model, datamodule=dm)
