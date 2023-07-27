from run_diag_tcn_training import run_training
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact
from typing import Dict, List
import pandas as pd
import joblib
from diag_tcn_vae import DiagTcnVAE
from tep_data_module import TEPDataset
import constants as const
import torch
import numpy as np


@dsl.component(
    base_image="python:3.9",
    target_image="gitlab.kiss.space.unibw-hamburg.de:4567/kiss/diag-tcn-vae/tep_compute_results:v04",
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
def compute_results(
    df: Input[Dataset],
    model: Input[Model],
    scaler: Input[Artifact],
    predictions: Output[Dataset],
    number_of_samples: int = 100,
) -> None:
    df = pd.read_parquet(df.path)
    scaler_obj = joblib.load(scaler.path)

    df = pd.DataFrame(
        scaler_obj.transform(df.values), columns=df.columns
    )
    ds = TEPDataset(
        dataframe=df,
        input_cols=const.DATA_COLS,
        subsystems_map=const.SUBSYSTEM_MAP,
    )

    model_obj = DiagTcnVAE.load_from_checkpoint(model.path)

    # run prediction loop
    y_ls = []
    scores_ls = []
    for i in range(number_of_samples):
        x, x_comp_ls, y = ds[i]
        with torch.no_grad():
            scores, _, _ = model_obj.predict(
                torch.Tensor(x),
                [torch.Tensor(x).reshape(1, *x.shape) for x in x_comp_ls],
            )
        scores_ls.append(np.array(scores).reshape(-1))
        y_ls.append(y[0])

    # save results in dataframe
    results_df = pd.DataFrame(scores_ls, columns=sorted(const.SUBSYSTEM_MAP.keys()))
    results_df['faultNumber'] = y_ls

    results_df.to_parquet(predictions.path)
