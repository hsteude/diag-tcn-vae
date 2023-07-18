import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, List, Optional


class TEPDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_cols: List[str],
        subsystems_map: Dict[str, List],
    ):
        self.df = dataframe
        self.sim_runs = list(self.df.simulationRun.unique())
        self.length = len(self.sim_runs)
        self.input_cols = input_cols
        self.subsystems_map = subsystems_map

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            self.df[self.df.simulationRun == self.sim_runs[index]][
                self.input_cols
            ].values.astype(np.float32).T,
            [
                self.df[
                    self.df.simulationRun == self.sim_runs[index]
                ][self.subsystems_map[subsys]].values.astype(np.float32).T
                for subsys in self.subsystems_map.keys()
            ],
            self.df[self.df.simulationRun == self.sim_runs[index]][
                "faultNumber"
            ].values.astype(np.float32),
        )


class TEPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        batch_size: int,
        input_cols: List[str],
        subsystems_map: Dict[str, List],
        scaler: StandardScaler,
        num_workers: int = 20,
        df_test: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = scaler
        df_train_sc = self.scale_df(df_train)
        self.ds_train = TEPDataset(
            dataframe=df_train_sc,
            input_cols=input_cols,
            subsystems_map=subsystems_map,
        )
        df_val_sc = self.scale_df(df_val)
        self.ds_val = TEPDataset(
            dataframe=df_val_sc,
            input_cols=input_cols,
            subsystems_map=subsystems_map,
        )
        if df_test:
            df_test_sc = self.scale_df(df_test)
            self.ds_test = TEPDataset(
                dataframe=df_test_sc,
                input_cols=input_cols,
                subsystems_map=subsystems_map,
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

