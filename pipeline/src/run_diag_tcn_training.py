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
    data_modeule: pl.LightningDataModule,
    batch_size: int,
    num_workers: int,
    subsystems_map: Dict[str, List],
    input_cols: List[str],
    model_output_path: str,
    kernel_size: int = 15,
    max_epochs: int = 3,
    learning_rate: float = 1e-3,
    logs_path: str = "logs",
    early_stopping_patience: int = 5,
    latent_dim: int = 5,
    dropout: float = 0.5,
    model_name: str = 'diag-tcn-vae'
) -> None:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    torch.set_float32_matmul_precision("medium")

    model = DiagTcnVAE(
        kernel_size=kernel_size,
        enc_tcn1_in_dims=[len(input_cols), 50, 40, 30, 20],
        dec_tcn2_out_dims=[10, 15, 20, 40, len(input_cols)],
        lr=learning_rate,
        seq_len=480,
        component_output_dims=[
            len(subsystems_map[k]) for k in sorted(subsystems_map.keys())
        ],
        latent_dim=latent_dim,
        dropout=dropout,
    )
    tb_logger = TensorBoardLogger(
        logs_path,
        name=model_name,
        default_hp_metric=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        strict=True,
    )

    callbacks = [early_stop_callback]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model=model, datamodule=data_modeule)
    trainer.save_checkpoint(model_output_path)
