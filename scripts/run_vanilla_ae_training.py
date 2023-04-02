from diag_vae.vanilla_tcn_ae import VanillaTcnAE
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from diag_vae.datamodule import DiagTsDataModule


if __name__ == "__main__":
    SEQ_LEN = 1000
    COLS = [f"sig_{i+1}" for i in range(6)]
    COLS_COMP_A = COLS[0:3] 
    COLS_COMP_B = COLS[3:6] 
    LATEND_DIM = 10
    dm = DiagTsDataModule(
        data_path="data/trainin_data.csv",
        seq_len=SEQ_LEN,
        cols=COLS,
    )
    model = VanillaTcnAE(
        in_dim=len(COLS),
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
