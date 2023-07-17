# imports
from kfp import dsl
from kfp.dsl import Input, Output, Dataset


@dsl.component(
    packages_to_install=["pandas==1.5.3", "s3fs", "pyarrow"],
    base_image="python:3.9",
)
def create_train_and_val_dataset(
    number_validation_samples: int,
    df_fault_free_training: Input[Dataset],
    df_fault_free_testing: Input[Dataset],
    df_train: Output[Dataset],
    df_val: Output[Dataset],
):
    import pandas as pd

    # read data
    fault_free_testing_df = pd.read_parquet(df_fault_free_testing.path)
    fault_free_training_df = pd.read_parquet(df_fault_free_training.path)

    # split simulation runs of testing df
    fault_free_testing_df["sample"] = list(range(1, 481)) * 500 * 2
    fault_free_testing_df["simulationRun"] = [
        i + 500 for i in range(1, 1001) for _ in range(480)
    ]

    # remove samples between 480 and 500 from trainig simulation runs
    fault_free_training_df = fault_free_training_df[
        fault_free_training_df["sample"] < 481
    ].reset_index(drop=True)

    # merge them together
    fault_free_df = pd.concat(
        (fault_free_training_df, fault_free_testing_df), axis=0
    ).reset_index(drop=True)

    # split train and val
    # just take the last simulation runs, this has all been generated at random
    max_train_simulation_run = fault_free_df.simulationRun.max() - number_validation_samples
    train_df = fault_free_df[fault_free_df.simulationRun <= max_train_simulation_run]
    val_df = fault_free_df[fault_free_df.simulationRun > max_train_simulation_run]

    # write out
    train_df.to_parquet(df_train.path)
    val_df.to_parquet(df_val.path)
