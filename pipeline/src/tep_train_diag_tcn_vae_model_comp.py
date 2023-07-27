from run_diag_tcn_training import run_training
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact
from typing import Dict, List
import pandas as pd
import joblib
from tep_data_module import TEPDataModule


@dsl.component(
    base_image="python:3.9",
    target_image="gitlab.kiss.space.unibw-hamburg.de:4567/kiss/diag-tcn-vae/diag-tcn-trainer:v12",
    packages_to_install=[
        "pytorch_lightning",
        "torch",
        "loguru",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "tensorboard",
    ],
)
def train_diag_model_tep(
    df_train: Input[Dataset],
    df_val: Input[Dataset],
    scaler: Input[Artifact],
    batch_size: int,
    num_workers: int,
    subsystems_map: Dict[str, List],
    input_cols: List[str],
    trained_model: Output[Model],
    logs_dir: Output[Artifact],
    kernel_size: int = 15,
    max_epochs: int = 3,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 10,
    latent_dim: int = 5,
    dropout: float = 0.5,
) -> None:
    train_df = pd.read_parquet(df_train.path)
    val_df = pd.read_parquet(df_val.path)
    scaler_obj = joblib.load(scaler.path)
    
    dm = TEPDataModule(
        df_train=train_df,
        df_val=val_df,
        batch_size=batch_size,
        subsystems_map=subsystems_map,
        input_cols=input_cols,
        scaler=scaler_obj,
        num_workers=num_workers,
    )
    trained_model = run_training(
        data_modeule=dm,
        subsystems_map=subsystems_map,
        input_cols=input_cols,
        batch_size=batch_size,
        num_workers=num_workers,
        model_output_path=trained_model.path,
        max_epochs=max_epochs,
        logs_path=logs_dir.path,
        kernel_size=kernel_size,
        early_stopping_patience=early_stopping_patience,
        latent_dim=latent_dim,
        dropout=dropout,
        learing_rate=learning_rate,
    )
