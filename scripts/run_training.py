from diag_vae.ae import Conv1dAE
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from diag_vae.datamodule import DiagTsDataModule


if __name__ == "__main__":
    SEQ_LEN_X = 1000
    SEQ_LEN_Y = 100
    dm = DiagTsDataModule(
        data_path="data/trainin_data.csv",
        seq_len_x=SEQ_LEN_X,
        seq_len_y=SEQ_LEN_Y,
        cols=[f"sig_{i+1}" for i in range(3)],
    )
    model = Conv1dAE(
        in_dim=3,
        latent_dim=200,
        seq_len_x=SEQ_LEN_X,
        seq_len_y=SEQ_LEN_Y,
    )
    logger = TensorBoardLogger("logs", name="tcn_ae", default_hp_metric=True)
    accelerator = "gpu" if torch.cuda.is_available() else 0
    devices = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=500, accelerator=accelerator, devices=devices, logger=logger
    )
    trainer.fit(model, datamodule=dm)
