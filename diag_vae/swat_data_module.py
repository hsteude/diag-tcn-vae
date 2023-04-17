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
        train_data_path: str,
        val_data_path: str,
        seq_len_x: int,
        seq_len_y: int = 200,
        split: str = "train",
        val_ts_start: str = "2015-12-28 18:00:00",
        val_ts_end: str = "2015-12-29 06:00:00",
    ):
        df_train = pd.read_parquet(train_data_path)
        df_val = pd.read_parquet(val_data_path)

        df_train = df_train[cols]
        df_val = df_val[cols]
        self.scaler = StandardScaler()
        df_train_sc = pd.DataFrame(
            self.scaler.fit_transform(df_train.values),
            columns=df_train.columns,
            index=df_train.index,
        )
        df_train = df_train
        df_train_sc = df_train_sc
        df_val_sc = pd.DataFrame(
            self.scaler.transform(df_val.values),
            columns=df_val.columns,
            index=df_val.index,
        )

        if split == "train":
            self.df = df_train_sc['2015-12-22 16:30:00':'2015-12-27 23:59:59']
        elif split == "val":
            self.df = df_train_sc['2015-12-28 00:00:00':'2015-12-28 09:59:55']
        elif split == "test":
            self.df = df_val_sc

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
        val_ts_start: str = "2015-12-28 18:00:00",
        val_ts_end: str = "2015-12-29 06:00:00",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len_x = seq_len_x
        self.cols = cols
        self.num_workers = dl_workers

        self.train_ds, self.val_ds, self.test_ds = [
            SwatDataset(
                train_data_path=data_path_train,
                val_data_path=data_path_val,
                seq_len_x=seq_len_x,
                seq_len_y=seq_len_y,
                cols=cols,
                symbols_dct=symbols_dct,
                split=split,
                val_ts_start=val_ts_start,
                val_ts_end=val_ts_end,
            )
            for split in ["train", "val", "test"]
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

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
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
