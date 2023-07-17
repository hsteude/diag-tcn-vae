# imports
from kfp.components import load_component_from_file
from kfp import dsl
from kfp.client import Client
import os
from src.load_data_comp import laod_data
from src.create_train_and_val_dataset_comp import create_train_and_val_dataset
from src.create_test_dataset_comp import create_test_dataset
from src.fit_scaler_comp import fit_scaler
import src.constants as const




# define pipeline
@dsl.pipeline
def diag_tcn_tep_pipeline(number_validation_samples: int = 150):
    load_data_task = laod_data(raw_data_dir=const.RAW_DATA_DIR)
    load_data_task.set_env_variable(
        "AWS_SECRET_ACCESS_KEY", os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    load_data_task.set_env_variable(
        "AWS_ACCESS_KEY_ID", os.environ["AWS_ACCESS_KEY_ID"]
    )
    load_data_task.set_env_variable("S3_ENDPOINT", os.environ["S3_ENDPOINT"])
    create_train_val_ds_task = create_train_and_val_dataset(
        number_validation_samples=number_validation_samples,
        df_fault_free_training=load_data_task.outputs["df_fault_free_training"],
        df_fault_free_testing=load_data_task.outputs["df_fault_free_testing"],
    )
    create_test_ds_task = create_test_dataset(
        df_faulty_training=load_data_task.outputs["df_faulty_training"],
        df_faulty_testing=load_data_task.outputs["df_faulty_testing"],
    )
    fit_scaler_task = fit_scaler(df_train=create_train_val_ds_task.outputs['df_train'])


# compile and run pipeline
client = Client()
client.create_run_from_pipeline_func(
    diag_tcn_tep_pipeline, arguments={}, experiment_name="diag-tcn-tep"
)
