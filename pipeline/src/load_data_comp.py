# imports
from kfp import dsl
from kfp.dsl import Output, Dataset


@dsl.component(
    packages_to_install=["pandas==1.5.3", "s3fs", "pyarrow", "pyreadr"],
    base_image="python:3.9",
)
def laod_data(
    raw_data_dir: str,
    df_fault_free_training: Output[Dataset],
    df_fault_free_testing: Output[Dataset],
    df_faulty_training: Output[Dataset],
    df_faulty_testing: Output[Dataset],
):
    import pandas as pd
    import s3fs
    import os
    import pyreadr

    s3 = s3fs.S3FileSystem(
        anon=False,
        use_ssl=False,
        client_kwargs={
            "endpoint_url": f'http://{os.environ["S3_ENDPOINT"]}',
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        },
    )

    # download files
    for path in s3.ls(raw_data_dir):
        s3.get(path, "./data/raw/TEP-KAGGLE/")
    file_names = os.listdir("./data/raw/TEP-KAGGLE/")

    df_dct = {}
    for f in file_names:
        result = pyreadr.read_r(f"./data/raw/TEP-KAGGLE/{f}")
        df_name = list(result.keys())[0]
        df_dct.update({df_name: pd.DataFrame(result[df_name])})

    faulty_training_df = df_dct["faulty_training"]
    faulty_testing_df = df_dct["faulty_testing"]
    fault_free_training_df = df_dct["fault_free_training"]
    fault_free_testing_df = df_dct["fault_free_testing"]

    for df, output in zip(
        [
            faulty_training_df,
            faulty_testing_df,
            fault_free_training_df,
            fault_free_testing_df,
        ],
        [
            df_faulty_training,
            df_faulty_testing,
            df_fault_free_training,
            df_fault_free_testing,
        ],
    ):
        df.to_parquet(output.path)
