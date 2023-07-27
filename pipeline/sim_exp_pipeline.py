# imports
from kfp.components import load_component_from_file
from kfp import dsl
from kfp.client import Client
import os
from src.load_data_comp import laod_data
from src.create_train_and_val_dataset_comp import create_train_and_val_dataset
from src.create_test_dataset_comp import create_test_dataset
from src.fit_scaler_comp import fit_scaler
from src.visualize_results_comp import visualize_predictions
import src.constants as const

from src.lightweight_comps_sim_exp import simulate_data, split_data

# define pipeline
@dsl.pipeline
def diag_tcn_simulated_pipeline():
    simulate_data_task = simulate_data(
        min_phase_length=1500,
        max_phase_length=2500,
        length=500,
    )
    split_data_task = split_data(
            simulate_data=simulate_data_task.outputs['simulation_data'])


# compile and run pipeline
client = Client()
client.create_run_from_pipeline_func(
    diag_tcn_simulated_pipeline,
    arguments={},
    experiment_name="diag_tcn_sim",
    enable_caching=True,
)


