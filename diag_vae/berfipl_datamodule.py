import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random


class BERFIPLDataSet(Dataset):
    def __init__(
        self,
        cols: list[str],
        symbols_dct: dict,
        data_path: str,
        seq_len_x: int,
        split: str,
        val_split: float = .7,
    ):
        df = pd.read_csv(data_path)
        df = df[cols]
        max_train_idx = int(val_split*len(df))
        df_train = df[:max_train_idx]
        df_val = df[max_train_idx:]

        self.scaler = StandardScaler()
        df_train_sc = pd.DataFrame(
            self.scaler.fit_transform(df_train.values),
            columns=df_train.columns,
            index=df_train.index,
        )
        df_val_sc = pd.DataFrame(
            self.scaler.transform(df_val.values),
            columns=df_val.columns,
            index=df_val.index,
        )
        if split == 'train':
            self.df_sc = df_train_sc

        if split == 'val':
            self.df_sc = df_val_sc

        self.len = len(self.df_sc.index) - seq_len_x

        self.x = torch.from_numpy(self.df_sc.values.astype(np.float32))
        self.comp_ls = [
            torch.from_numpy(self.df_sc[symbols_dct[comp]].values.astype(np.float32))
            for comp in symbols_dct.keys()
        ]
        self.seq_len_x = seq_len_x

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            self.x[index : index + self.seq_len_x, :].T,
            [comp[index : index + self.seq_len_x, :].T for comp in self.comp_ls],
            torch.zeros(1)
        )


class BERFIPLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path_normal: str,
        cols: list[str],
        symbols_dct: dict,
        batch_size: int = 32,
        seq_len_x: int = 1000,
        dl_workers: int = 8,
        data_path_test_c: str = None,
        data_path_test_l: str = None,
        data_path_test_lc: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len_x = seq_len_x
        self.cols = cols
        self.num_workers = dl_workers

        self.train_ds, self.val_ds = [BERFIPLDataSet(
            cols=cols,
            symbols_dct=symbols_dct,
            data_path=data_path_normal,
            seq_len_x=seq_len_x,
            split=split
        ) for split in ('train', 'val')]


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

    # ds = BERFIPLDataSet(
    #         cols=const.BERFIPL_COLS,
    #         symbols_dct=const.BERFIPL_SIGNALS_MAP,
    #         data_path=const.BERFIPL_RAW_DATA_PATH_NORMAL,           seq_len_x=1000,
    #         split='trian'
    # )
    #
    # ds.__getitem__(0)

    dm = BERFIPLDataModule(
        data_path_normal=  const.BERFIPL_RAW_DATA_PATH_NORMAL,
        cols = const.BERFIPL_COLS,
        symbols_dct=const.BERFIPL_SIGNALS_MAP,
        batch_size=32,
        seq_len_x=1000,
        dl_workers=60
    )
    dlt = dm.train_dataloader()
    dlv = dm.val_dataloader()
    batch = next(iter(dlt))
    batch = next(iter(dlv))
    breakpoint()
