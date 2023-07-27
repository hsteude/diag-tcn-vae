import pandas as pd
import s3fs
import os
from pipeline.src.tep_data_module import TEPDataset
import pipeline.src.constants as const
from pipeline.src.tep_run_diag_tcn_training import run_training
import joblib

scaler = joblib.load("./data/tmp/scaler")
df_train = pd.read_parquet("./data/tmp/df_train.parquet")
df_val = pd.read_parquet("./data/tmp/df_val.parquet")

run_training(
    df_train=df_train,
    subsystems_map=const.SUBSYSTEM_MAP,
    input_cols=const.DATA_COLS,
    df_val=df_val,
    scaler=scaler,
    batch_size=128,
    num_workers=20,
    max_epochs=2,
    model_output_path='../../data/tmp/output_model',
    logs_path='s3://hs-bucket/logs'
)
