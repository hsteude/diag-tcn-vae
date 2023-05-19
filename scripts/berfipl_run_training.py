from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from diag_vae.berfipl_datamodule import BERFIPLDataModule

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import importlib
import os
from loguru import logger


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
    model_name: str = None,
) -> None:
    dm = BERFIPLDataModule(**data_module_args)
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
    model_name = model_name if model_name else model_module
    logger = TensorBoardLogger(log_dir, name=model_name, default_hp_metric=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = num_devices

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        strict=True,
    )
    model_callback = ModelCheckpoint(monitor="val_loss", save_top_k=2)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[early_stop_callback, model_callback],
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

    LATEND_DIM = 10
    SEQ_LEN = 500
    KERNEL_SIZE = 13
    BATCH_SIZE = 32
    MAX_EPOCHS = 25
    EARLY_STOPPING_PATIENCE = 20


    for symb in list(const.BERFIPL_SIGNALS_MAP.keys()):
        logger.info(f"Training model for comp {symb}")

        model_args = dict(
            enc_tcn1_in_dims=[len(const.BERFIPL_SIGNALS_MAP[symb]), 50, 40, 30, 20],
            enc_tcn1_out_dims=[50, 40, 30, 20, 10],
            enc_tcn2_in_dims=[10, 6, 5, 4, 3],
            enc_tcn2_out_dims=[6, 5, 4, 3, 1],
            latent_dim=LATEND_DIM,
            kernel_size=KERNEL_SIZE,
            seq_len=SEQ_LEN,
            lr=1e-3,
        )

        dm_args = dict(
            data_path_normal=const.BERFIPL_RAW_DATA_PATH_NORMAL,
            cols=const.BERFIPL_SIGNALS_MAP[symb],
            symbols_dct=const.BERFIPL_SIGNALS_MAP,
            batch_size=BATCH_SIZE,
            seq_len_x=SEQ_LEN,
            dl_workers=60,
            diag_ls=False,
        )
        logger.info(f"Using these signals {const.BERFIPL_SIGNALS_MAP[symb]}")

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
            model_name=f'VanillaTcnComp_{symb}',
        )

    # diag tcn
    # if args.model == "DiagTcnAE":

    # model_args = dict(
    #     enc_tcn1_in_dims=[154, 40, 20, 15, 10],
    #     enc_tcn1_out_dims=[40, 20, 15, 10, 10],
    #     enc_tcn2_in_dims=[10, 8, 6, 5, 3],
    #     enc_tcn2_out_dims=[8, 6, 5, 3, 1],
    #     dec_tcn1_in_dims=[1, 3, 5, 6, 6],
    #     dec_tcn1_out_dims=[3, 5, 6, 6, 6],
    #     dec_tcn2_in_dims=[6, 6, 6, 6, 6],
    #     dec_tcn2_out_dims=[6, 6, 6, 6, 9999999], #one shorter bcause of diag-vae code
    #     component_output_dims=[
    #         len(const.BERFIPL_SIGNALS_MAP[k]) for k in const.BERFIPL_SIGNALS_MAP.keys()
    #     ],
    #     latent_dim=LATEND_DIM,
    #     kernel_size=KERNEL_SIZE,
    #     seq_len=SEQ_LEN,
    #     lr=1e-3,
    # )
    #
    # dm_args = dict(
    #     data_path_normal=const.BERFIPL_RAW_DATA_PATH_NORMAL,
    #     cols=const.BERFIPL_COLS,
    #     symbols_dct=const.BERFIPL_SIGNALS_MAP,
    #     batch_size=BATCH_SIZE,
    #     seq_len_x=SEQ_LEN,
    #     dl_workers=60,
    # )
    #
    # run_training(
    #     model_module="diag_vae.diag_tcn_ae",
    #     model_class_name="DiagTcnAE",
    #     model_args=model_args,
    #     data_module_args=dm_args,
    #     log_dir="logs",
    #     num_devices=1,
    #     max_epochs=MAX_EPOCHS,
    #     checkpoint_dir=args.checkpoint_path,
    #     early_stopping_patience=EARLY_STOPPING_PATIENCE,
    # )
