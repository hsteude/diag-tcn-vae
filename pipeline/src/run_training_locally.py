import pandas as pd
import os
from pipeline.src.tep_data_module import TEPDataset
import pipeline.src.constants as const

DF_TRAIN_PATH = "minio://mlpipeline/v2/artifacts/diag-tcn-tep-pipeline/1dc2ab4a-2066-4836-8cea-f5d2a1e1b8e1/create-train-and-val-dataset/df_training"
DF_VAL_PATH = "minio://mlpipeline/v2/artifacts/diag-tcn-tep-pipeline/1dc2ab4a-2066-4836-8cea-f5d2a1e1b8e1/create-train-and-val-dataset/df_validation"
DF_TEST_PATH = "minio://mlpipeline/v2/artifacts/diag-tcn-tep-pipeline/4022b3b2-0255-4683-8ef0-f5f3deb569ec/create-test-dataset/df_test"
SALER_PATH = "minio://mlpipeline/v2/artifacts/diag-tcn-tep-pipeline/ebbca4f7-a912-4090-af57-b26837ecc4a8/fit-scaler/scaler"

storage_options = {
    "key": os.environ["AWS_ACCESS_KEY_ID"],
    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    "client_kwargs": {"endpoint_url": f'http://{os.environ["S3_ENDPOINT"]}'},
}
df_train = pd.read_parquet(
    DF_TRAIN_PATH.replace("minio://", "s3://"), storage_options=storage_options
)
df_val = pd.read_parquet(
    DF_VAL_PATH.replace("minio://", "s3://"), storage_options=storage_options
)
df_test = pd.read_parquet(
    DF_TEST_PATH.replace("minio://", "s3://"), storage_options=storage_options
)

ds_train = TEPDataset(dataframe=df_train, input_cols=const.DATA_COLS)
ds_val = TEPDataset(dataframe=df_val, input_cols=const.DATA_COLS)
ds_test = TEPDataset(dataframe=df_test, input_cols=const.DATA_COLS)





