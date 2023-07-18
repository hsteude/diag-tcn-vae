from tep_run_diag_tcn_training  import run_training
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact
from typing import Dict, List
import pandas as pd
import joblib


@dsl.component(base_image='python:3.9',
               target_image='gitlab.kiss.space.unibw-hamburg.de:4567/kiss/diag-tcn-vae/diag-tcn-trainer:v11',
               packages_to_install=[
                'pytorch_lightning',
                'torch',
                'loguru',
                'numpy',
                'pandas',
                'scikit-learn',
                'pyarrow',
                'tensorboard'
                   ]
               )
def train_model(
    df_train: Input[Dataset],
    df_val: Input[Dataset],
    scaler: Input[Artifact],
    batch_size: int,
    num_workers: int,
    subsystems_map: Dict[str, List],
    input_cols: List[str],
    trained_model: Output[Model],
    logs: Output[Artifact],
    kernel_size: int = 15,
    dropout: float = 0.1,
    max_epochs: int = 3,
    num_signals: int = 52,
    learning_rate: float = 1e-3,
) -> None:
    train_df = pd.read_parquet(df_train.path)
    val_df = pd.read_parquet(df_val.path)
    scaler_obj = joblib.load(scaler.path)
    trained_model = run_training(
        df_train=train_df,
        subsystems_map=subsystems_map,
        input_cols=input_cols,
        df_val=val_df,
        scaler=scaler_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        model_output_path=trained_model.path,
        max_epochs=max_epochs,
        logs_path=logs.path,
        kernel_size=kernel_size
    )

