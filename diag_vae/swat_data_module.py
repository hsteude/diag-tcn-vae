import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SwatDataset(Dataset):
    def __init__(
        self,
        cols: list[str],
        symbols_dct: dict,
        data_path: str,
        scale: bool,
        seq_len_x: int,
        seq_len_y: int = 200,
    ):
        df = pd.read_parquet(data_path)

        df = df[cols]
        if scale:
            self.scaler = StandardScaler()
            df = pd.DataFrame(
                self.scaler.fit_transform(df.values),
                columns=df.columns,
                index=df.index,
            )
        self.df = df
        self.x = torch.from_numpy(self.df.values.astype(np.float32))
        self.comp_ls = [
            torch.from_numpy(self.df[symbols_dct[comp]].values.astype(np.float32))
            for comp in symbols_dct.keys()
        ]
        self.length = len(self.df) - seq_len_x - seq_len_y
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            self.x[index : index + self.seq_len_x, :].T,
            [comp[index : index + self.seq_len_x, :].T for comp in self.comp_ls],
            [
                comp[
                    index + self.seq_len_x : index + self.seq_len_x + self.seq_len_y, :
                ].T
                for comp in self.comp_ls
            ],
        )


class SwatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path_train: str,
        data_path_val: str,
        cols: list[str],
        symbols_dct: dict,
        batch_size: int = 32,
        seq_len_x: int = 1000,
        seq_len_y: int = 250,
        dl_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len_x = seq_len_x
        self.cols = cols
        self.num_workers = dl_workers

        self.train_ds, self.val_ds = [
            SwatDataset(
                data_path=path,
                seq_len_x=seq_len_x,
                seq_len_y=seq_len_y,
                cols=cols,
                symbols_dct=symbols_dct,
                scale=True,
            )
            for path in (data_path_train, data_path_val)
        ]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    import diag_vae.constants as const

    ds = SwatDataset(
        data_path=const.SWAT_VAL_PATH,
        seq_len_x=1000,
        cols=const.SWAT_SENSOR_COLS,
        symbols_dct=const.SWAT_SYMBOLS_MAP,
        scale=True,
    )
    ds.__getitem__(0)

    dm = SwatDataModule(
        data_path_train=const.SWAT_TRAIN_PATH,
        data_path_val=const.SWAT_VAL_PATH,
        seq_len_x=1000,
        cols=const.SWAT_SENSOR_COLS,
        symbols_dct=const.SWAT_SYMBOLS_MAP,
        dl_workers=8,
    )
