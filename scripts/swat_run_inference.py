import pandas as pd


import diag_vae.constants as const
import pandas as pd
from diag_vae.vanilla_tcn_ae import VanillaTcnAE
from diag_vae.diag_tcn_vae import DiagTcnVAE
import torch
import tqdm
import os
from diag_vae.swat_data_module import SwatDataModule
import numpy as np


def run(
    model_version_vanilla: int = 0,
    model_version_diag: int = 0,
    split: str = "test",
    latent_dim: int = 5,
    number_comps: int = 6,
):
    # load models from checkoints
    loggs_dir = f"./logs/VanillaTcnAE/version_{model_version_vanilla}/checkpoints/"
    checkpoint = os.path.join(loggs_dir, f"{os.listdir(loggs_dir)[0]}")
    ae_model = VanillaTcnAE.load_from_checkpoint(checkpoint)

    loggs_dir = f"./logs/DiagTcnVAE/version_{model_version_diag}/checkpoints/"
    checkpoint = os.path.join(loggs_dir, f"{os.listdir(loggs_dir)[0]}")
    diag_vae_model = DiagTcnVAE.load_from_checkpoint(checkpoint)

    # laod data modules
    dm = SwatDataModule(
        data_path_val=const.SWAT_VAL_PATH,
        data_path_train=const.SWAT_TRAIN_PATH,
        seq_len_x=500,
        seq_len_y=100,
        cols=const.SWAT_SENSOR_COLS,
        symbols_dct=const.SWAT_SYMBOLS_MAP,
        batch_size=100,
        dl_workers=8,
    )
    dl_val = dm.val_dataloader()
    dl_train = dm.train_dataloader()
    dl_test = dm.test_dataloader()

    dl = dl_train if split == "train" else dl_test

    ae_model_mse_ls_ls = []
    diag_vae_model_mse_ls_ls = []
    counter = 0
    for batch in tqdm.tqdm(iter(dl)):
        with torch.no_grad():
            ae_model_mse_ls_ls.append(ae_model.predict_step(batch, batch_idx=None))
            diag_vae_model_mse_ls_ls.append(
                diag_vae_model.predict_step(batch, batch_idx=None)
            )
            counter += 1

    ae_mse_df = pd.DataFrame(
        {
            f"ae_mse_comp{i+1}": np.concatenate([l[0][i] for l in ae_model_mse_ls_ls])
            for i in range(number_comps)
        }
    )
    ae_mse_df.index = dl.dataset.df.index[0 : len(ae_mse_df)]

    ae_z_df = pd.DataFrame(
        np.concatenate([l[1] for l in ae_model_mse_ls_ls]),
        columns=[f"ae_z{i+1}" for i in range(latent_dim)],
    )
    ae_z_df.index = dl.dataset.df.index[0 : len(ae_mse_df)]

    diag_vae_mse_df = pd.DataFrame(
        {
            f"diag_vae_mse_comp{i+1}": np.concatenate(
                [l[0][i] for l in diag_vae_model_mse_ls_ls]
            )
            for i in range(number_comps)
        }
    )
    diag_vae_mse_df.index = dl.dataset.df.index[0 : len(diag_vae_mse_df)]

    diag_vae_z_df = pd.DataFrame(
        np.concatenate([l[1] for l in ae_model_mse_ls_ls]),
        columns=[f"diag_vae_z{i+1}" for i in range(latent_dim)],
    )
    diag_vae_z_df.index = dl.dataset.df.index[0 : len(diag_vae_mse_df)]
    result_df = pd.concat(
        [ae_mse_df, ae_z_df, diag_vae_mse_df, diag_vae_z_df],
        axis=1,
        join="outer",
    )
    result_df.to_parquet(
        const.RESULTS_SWAT_TRAIN_PATH
        if split == "train"
        else const.RESULTS_SWAT_TEST_PATH
    )


if __name__ == "__main__":
    run(split="train")
    run(split="test")
