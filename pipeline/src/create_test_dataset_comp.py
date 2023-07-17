# imports
from kfp import dsl
from kfp.dsl import Input, Output, Dataset


@dsl.component(
    packages_to_install=["pandas==1.5.3", "s3fs", "pyarrow"],
    base_image="python:3.9",
)
def create_test_dataset(
    df_faulty_training: Input[Dataset],
    df_faulty_testing: Input[Dataset],
    df_test: Output[Dataset],
):
    import pandas as pd

    # read data
    faulty_testing_df = pd.read_parquet(df_faulty_testing.path)
    faulty_training_df = pd.read_parquet(df_faulty_training.path)

    # select correct samples from faulty training df
    faulty_training_df = faulty_training_df[
        faulty_training_df["sample"] > 20
    ].reset_index(drop=True)
    faulty_training_df["simulationRun"] = [
        i for i in range(1, 500 * 20 + 1) for _ in range(480)
    ]

    # split correct samples from faulty testing
    faulty_testing_df = faulty_testing_df[
        (faulty_testing_df["sample"] > 160)
        & (faulty_testing_df["sample"] <= (160 + 480))
    ].reset_index(drop=True)
    faulty_testing_df["simulationRun"] = [
        i+(500 * 20) for i in range(1, 500 * 20 + 1) for _ in range(480)
    ]

    # merge them together
    faulty_df = pd.concat((faulty_training_df, faulty_testing_df), axis=0).reset_index(
        drop=True
    )

    # write out
    faulty_df.to_parquet(df_test.path)
