from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from diag_vae.swat_data_module import SwatDataModule
from loguru import logger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import importlib
import os
from pytorch_lightning.callbacks import Callback
import numpy as np

class BetaIncreaseCallBack(Callback):
    def __init__(self, initial_beta, beta_start, beta_max, increase_after_n_epochs, number_steps):
        super().__init__()
        self.idx = 0
        self.num_epochs = increase_after_n_epochs
        self.betas = [initial_beta]*10 + list(
            np.linspace(beta_start, beta_max, number_steps)
        )

    def on_train_epoch_start(self, trainer, pl_modelu):
        if self.idx < len(self.betas):
            pl_modelu.beta = self.betas[self.idx]
            if trainer.current_epoch % self.num_epochs == 0:
                if self.idx <= len(self.betas):
                    logger.info(f"Set Beta to {self.betas[self.idx]}")
                    self.idx += 1

def run_training(
    model_module: str,
    model_class_name: str,
    model_args: dict,
    data_module_args: dict,
    log_dir: str = "logs",
    num_devices: int = 1,
    max_epochs: int = 500,
    early_stopping_patience=50,
    checkpoint_dir: str = "",
    beta_increase_callback = None
) -> None:
    dm = SwatDataModule(**data_module_args)
    importlib.import_module(model_module)
    ModelClass = getattr(
        importlib.import_module(model_module),
        model_class_name,
    )
    if checkpoint_dir:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{os.listdir(checkpoint_dir)[0]}",
        )
        model = ModelClass(**model_args).load_from_checkpoint(checkpoint_path)
    else:
        model = ModelClass(**model_args)
    logger = TensorBoardLogger(log_dir, name=model_class_name, default_hp_metric=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = num_devices

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        strict=True,
    )
    model_callback = ModelCheckpoint(monitor="val_loss", save_top_k=2)

    callbacks = [early_stop_callback, model_callback]
    if beta_increase_callback:
        callbacks = callbacks+[beta_increase_callback]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
    )
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    import diag_vae.constants as const
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The name of the model, currently we have VanillaTcnAE and DiagTcnAE",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to where the latest checkpoints are, in case you want"
        "to continue training",
    )
    args = parser.parse_args()

    LATEND_DIM = 5
    SEQ_LEN = 500
    KERNEL_SIZE = 13
    BATCH_SIZE = 128
    MAX_EPOCHS = 10000
    NUM_TRAIN_SAMPLES = 50_000
    NUM_VAL_SAMPLES = 5_000
    EARLY_STOPPING_PATIENCE = 50
    # diag tcn vae
    if args.model == "DiagTcnVAE":
        model_args = dict(
            enc_tcn1_in_dims=[51, 40, 20, 15, 10],
            enc_tcn1_out_dims=[40, 20, 15, 10, 10],
            enc_tcn2_in_dims=[10, 8, 6, 5, 3],
            enc_tcn2_out_dims=[8, 6, 5, 3, 1],
            component_output_dims=[
                len(const.SWAT_SYMBOLS_MAP[k]) for k in const.SWAT_SYMBOLS_MAP.keys()
            ],
            latent_dim=LATEND_DIM,
            kernel_size=KERNEL_SIZE,
            seq_len=SEQ_LEN,
            lr=1e-3,
            beta=.0001,
        )

        dm_args = dict(
            data_path_train=const.SWAT_TRAIN_PATH,
            data_path_val=const.SWAT_VAL_PATH,
            seq_len_x=SEQ_LEN,
            cols=const.SWAT_SENSOR_COLS,
            symbols_dct=const.SWAT_SYMBOLS_MAP,
            batch_size=BATCH_SIZE,
            dl_workers=12,
            num_train_samples=NUM_TRAIN_SAMPLES,
            num_val_samples=NUM_VAL_SAMPLES,
        )
        
        beta_increase_callback = BetaIncreaseCallBack(
            initial_beta=0.0001, 
            beta_start=model_args['beta'], 
            beta_max=model_args['beta'],
            increase_after_n_epochs=1,
            number_steps=10,
         )
        

        run_training(
            model_module="diag_vae.diag_tcn_vae",
            model_class_name="DiagTcnVAE",
            model_args=model_args,
            data_module_args=dm_args,
            log_dir="logs",
            num_devices=1,
            max_epochs=MAX_EPOCHS,
            checkpoint_dir=args.checkpoint_path,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            beta_increase_callback=beta_increase_callback
        )

        
    # vanila tcn ae
    if args.model == "VanillaTcnAE":
        model_args = dict(
            enc_tcn1_in_dims=[51, 100, 50, 40, 30],
            enc_tcn1_out_dims=[100, 50, 40, 30, 20],
            enc_tcn2_in_dims=[20, 10, 8, 6, 3],
            enc_tcn2_out_dims=[10, 8, 6, 3, 1],
            latent_dim=LATEND_DIM,
            kernel_size=KERNEL_SIZE,
            seq_len=SEQ_LEN,
            lr=1e-3,
        )

        dm_args = dict(
            data_path_train=const.SWAT_TRAIN_PATH,
            data_path_val=const.SWAT_VAL_PATH,
            seq_len_x=SEQ_LEN,
            cols=const.SWAT_SENSOR_COLS,
            symbols_dct=const.SWAT_SYMBOLS_MAP,
            batch_size=BATCH_SIZE,
            dl_workers=12,
            num_train_samples=NUM_TRAIN_SAMPLES,
            num_val_samples=NUM_VAL_SAMPLES,
        )
        run_training(
            model_module="diag_vae.vanilla_tcn_ae",
            model_class_name="VanillaTcnAE",
            model_args=model_args,
            data_module_args=dm_args,
            log_dir="logs",
            num_devices=1,
            max_epochs=MAX_EPOCHS,
            checkpoint_dir=args.checkpoint_path,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )

    # diag tcn
    if args.model == "DiagTcnAE":
        model_args = dict(
            enc_tcn1_in_dims=[51, 40, 20, 15, 10],
            enc_tcn1_out_dims=[40, 20, 15, 10, 10],
            enc_tcn2_in_dims=[10, 8, 6, 5, 3],
            enc_tcn2_out_dims=[8, 6, 5, 3, 1],
            component_output_dims=[
                len(const.SWAT_SYMBOLS_MAP[k]) for k in const.SWAT_SYMBOLS_MAP.keys()
            ],
            latent_dim=LATEND_DIM,
            kernel_size=KERNEL_SIZE,
            seq_len=SEQ_LEN,
            lr=1e-3,
        )

        dm_args = dict(
            data_path_train=const.SWAT_TRAIN_PATH,
            data_path_val=const.SWAT_VAL_PATH,
            seq_len_x=SEQ_LEN,
            cols=const.SWAT_SENSOR_COLS,
            symbols_dct=const.SWAT_SYMBOLS_MAP,
            batch_size=BATCH_SIZE,
            dl_workers=12,
            num_train_samples=NUM_TRAIN_SAMPLES,
            num_val_samples=NUM_VAL_SAMPLES,
        )

        run_training(
            model_module="diag_vae.diag_tcn_ae",
            model_class_name="DiagTcnAE",
            model_args=model_args,
            data_module_args=dm_args,
            log_dir="logs",
            num_devices=1,
            max_epochs=MAX_EPOCHS,
            checkpoint_dir=args.checkpoint_path,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )

    # diag tcn predictor
    if args.model == "DiagTcnAePredictor":
        model_args = dict(
            enc_tcn1_in_dims=[
                51,
                50,
                20,
            ],  # 30, 20],
            enc_tcn1_out_dims=[50, 20, 10],  # , 20, 10],
            enc_tcn2_in_dims=[10, 6, 3],  # , 4, 3],
            enc_tcn2_out_dims=[6, 3, 1],  # , 3, 1],
            component_output_dims=[
                len(const.SWAT_SYMBOLS_MAP[k]) for k in const.SWAT_SYMBOLS_MAP.keys()
            ],
            latent_dim=LATEND_DIM,
            kernel_size=KERNEL_SIZE,
            seq_len_x=SEQ_LEN,
            seq_len_y=SEQ_LEN_Y,
            lr=1e-3,
        )

        dm_args = dict(
            data_path_train=const.SWAT_TRAIN_PATH,
            data_path_val=const.SWAT_TRAIN_PATH,
            seq_len_x=SEQ_LEN,
            seq_len_y=SEQ_LEN_Y,
            cols=const.SWAT_SENSOR_COLS,
            symbols_dct=const.SWAT_SYMBOLS_MAP,
            batch_size=BATCH_SIZE,
            dl_workers=12,
        )
        run_training(
            model_module="diag_vae.diag_tcn_ae_predictor",
            model_class_name="DiagTcnAePredictor",
            model_args=model_args,
            data_module_args=dm_args,
            log_dir="logs",
            num_devices=1,
            max_epochs=MAX_EPOCHS,
            checkpoint_dir=args.checkpoint_path,
        )
