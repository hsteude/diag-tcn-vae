from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from diag_vae.swat_data_module import SwatDataModule
import importlib


def run_training(
    model_module: str,
    model_class_name: str,
    model_args: dict,
    data_module_args: dict,
    log_dir: str = "logs",
    num_devices: int = 1,
    max_epochs: int = 500,
) -> None:
    dm = SwatDataModule(**data_module_args)
    importlib.import_module(model_module)
    ModelClass = getattr(
        importlib.import_module(model_module),
        model_class_name,
    )
    model = ModelClass(**model_args)
    logger = TensorBoardLogger(log_dir, name=model_class_name, default_hp_metric=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = num_devices
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    import diag_vae.constants as const
    import argparse
    breakpoint()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The name of the model, currently we have VanillaTcnAE and DiagTcnAE",
    )
    args = parser.parse_args()

    LATEND_DIM = 100
    SEQ_LEN = 1000
    KERNEL_SIZE = 5

    # vanila tcn ae
    if args.model == "VanillaTcnAE":
        model_args = dict(
            enc_tcn1_in_dims=[51, 100, 60, 40, 20],
            enc_tcn1_out_dims=[100, 60, 40, 20, 10],
            enc_tcn2_in_dims=[10, 6, 5, 4, 3],
            enc_tcn2_out_dims=[6, 5, 4, 3, 1],
            latent_dim=LATEND_DIM,
            kernel_size=KERNEL_SIZE,
            seq_len=SEQ_LEN,
            lr=1e-3,
        )

        dm_args = dict(
            data_path_train=const.SWAT_TRAIN_PATH,
            data_path_val=const.SWAT_TRAIN_PATH,
            seq_len=SEQ_LEN,
            cols=const.SWAT_SENSOR_COLS,
            symbols_dct=const.SWAT_SYMBOLS_MAP,
            batch_size=32,
            dl_workers=12,
        )
        run_training(
            model_module="diag_vae.vanilla_tcn_ae",
            model_class_name="VanillaTcnAE",
            model_args=model_args,
            data_module_args=dm_args,
            log_dir="logs",
            num_devices=1,
            max_epochs=500,
        )

    # diag tcn
    if args.model == "DiagTcnAE":
        model_args = dict(
            enc_tcn1_in_dims=[51, 50, 40, 30, 20],
            enc_tcn1_out_dims=[50, 40, 30, 20, 10],
            enc_tcn2_in_dims=[10, 6, 5, 4, 3],
            enc_tcn2_out_dims=[6, 5, 4, 3, 1],
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
            data_path_val=const.SWAT_TRAIN_PATH,
            seq_len=SEQ_LEN,
            cols=const.SWAT_SENSOR_COLS,
            symbols_dct=const.SWAT_SYMBOLS_MAP,
            batch_size=32,
            dl_workers=12,
        )
        run_training(
            model_module="diag_vae.diag_tcn_ae",
            model_class_name="DiagTcnAE",
            model_args=model_args,
            data_module_args=dm_args,
            log_dir="logs",
            num_devices=1,
            max_epochs=500,
        )
