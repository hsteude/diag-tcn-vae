import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import random
import torch


class SimDataSet(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        subsystems_map: Dict[str, List],
        input_cols: List,
        number_of_samples: int,
        seq_len: int,
    ):
        self.df = dataframe
        self.seq_len = seq_len
        self.length = number_of_samples
        self.input_cols = input_cols
        self.subsystems_map = subsystems_map
        start_idx_ls = list(range(len(dataframe) - seq_len))
        random.seed(42)
        self.start_idx_ls = random.sample(start_idx_ls, k=number_of_samples)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = self.start_idx_ls[index]
        return (
                self.df[idx:idx+self.seq_len][self.input_cols]
            .values.astype(np.float32)
            .T,
            [
                self.df[idx:idx+self.seq_len][
                    self.subsystems_map[subsys]
                ]
                .values.astype(np.float32)
                .T
                for subsys in sorted(self.subsystems_map.keys())
            ],
            torch.Tensor()
        )


class SIMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        batch_size: int,
        input_cols: List[str],
        subsystems_map: Dict[str, List],
        number_of_train_samples: int,
        number_of_val_samples: int,
        seq_len: int,
        num_workers: int = 20,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ds_train = SimDataSet(
            dataframe=df_train,
            input_cols=input_cols,
            subsystems_map=subsystems_map,
            number_of_samples=number_of_val_samples,
            seq_len=seq_len
        )
        self.ds_val = SimDataSet(
            dataframe=df_val,
            input_cols=input_cols,
            subsystems_map=subsystems_map,
            number_of_samples=number_of_train_samples,
            seq_len=seq_len
        )

    def scale_df(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_values = self.scaler.transform(df.values)
        return pd.DataFrame(scaled_values, columns=df.columns)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

