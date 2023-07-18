from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tep_data_module import TEPDataModule
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from diag_tcn_vae import DiagTcnVAE
from typing import Dict, List


def run_training(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    scaler: StandardScaler,
    batch_size: int,
    num_workers: int,
    subsystems_map: Dict[str, List],
    input_cols: List[str],
    model_output_path: str,
    kernel_size: int = 15,
    dropout: float = 0.1,
    max_epochs: int = 3,
    num_signals: int = 52,
    learning_rate: float = 1e-3,
    logs_path: str = "logs",
    early_stopping_patience: int = 5,
) -> None:
    dm = TEPDataModule(
        df_train=df_train,
        df_val=df_val,
        batch_size=batch_size,
        subsystems_map=subsystems_map,
        input_cols=input_cols,
        scaler=scaler,
        num_workers=num_workers,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    torch.set_float32_matmul_precision("medium")

    model = DiagTcnVAE(
        kernel_size=kernel_size,
        enc_tcn1_in_dims=[num_signals, 50, 40, 30, 20],
        dec_tcn2_out_dims=[10, 15, 20, 40, num_signals],
        lr=learning_rate,
        seq_len=480,
        component_output_dims=[len(subsystems_map[k]) for k in subsystems_map.keys()],
    )
    tb_logger = TensorBoardLogger(
        logs_path,
        name="tep-diag-tcn-vae",
        default_hp_metric=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        strict=True,
    )

    callbacks = [early_stopping_patience]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.save_checkpoint(model_output_path)
