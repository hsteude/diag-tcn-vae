import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from loguru import logger
from typing import Dict, List


class TEPDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_cols: List[str]
    ):
        self.df = dataframe
        self.sim_runs =list(self.df.simulationRun.unique())
        self.length = len(self.sim_runs)
        self.input_cols  = input_cols

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            self.df[self.df.simulationRun == self.sim_runs[index]][self.input_cols].values.astype(np.float32)
        )

class TEPDataModule(pl.LightningDataModule): 
    def __init__(self,
                 df_train: pd.DataFrame,
                 df_val: pd.DataFrame,
                 df_test: pd.DataFrame,
                 batch_size: int,
                 ) -> None:
        super().__init__()
        self.batch_size = batch_size



# class StressCurveDM(pl.LightningDataModule):
#     def __init__(
#         self,
#         x_data_path: str,
#         y_data_path: str,
#         cat_data_path: str,
#         num_k_fold_splits: int = 5,
#         batch_size : int = 8,
#         num_dl_workers: int = 8,
#         storage_options: Dict = None,
#         train_all: bool = False
#     ):
#         super().__init__()
#         self.train_all = train_all
#         self.batch_size = batch_size
#         self.num_workers = num_dl_workers
#         self.num_splits = num_k_fold_splits
#         self.storage_options = storage_options
#         df_x, df_y, df_cat = self.read_data(x_data_path, y_data_path, cat_data_path)
#         self.X = torch.from_numpy(df_x.values.astype(np.float32))
#         self.y = torch.from_numpy(df_y.values.astype(np.float32))
#
#         self.rskf = RepeatedStratifiedKFold(n_splits=self.num_splits, n_repeats=1, 
#                        random_state=42)
#         self.splits = list(self.rskf.split(self.X, df_cat.cathegory.values))
#
#     def read_data(self, x_data_path, y_data_path, cat_data_path):
#         logger.info(f'Received data inputs of type {type(x_data_path)}')
#         if type(x_data_path) == str:
#             logger.info(f'String path for x_data_path looks like this: {x_data_path}')
#             if 'minio' in x_data_path:
#                 x_data_path, y_data_path, cat_data_path = \
#                         [path.replace('/minio/', 's3://') for path in (x_data_path, y_data_path, cat_data_path)]
#
#             df_x = pd.read_parquet(x_data_path, storage_options=self.storage_options)
#             df_y = pd.read_parquet(y_data_path, storage_options=self.storage_options)
#             df_cat = pd.read_parquet(cat_data_path, storage_options=self.storage_options)
#         else:
#             df_x = pd.read_parquet(x_data_path.path)
#             df_y = pd.read_parquet(y_data_path.path)
#             df_cat = pd.read_parquet(cat_data_path.path)
#         return df_x, df_y, df_cat
#
#     def train_dataloader(self, fold=0):
#         if self.train_all:
#             return DataLoader(
#                 StressCurveDS(
#                     self.X,
#                     self.y),
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=self.num_workers,
#             )
#         else:
#             return DataLoader(
#                 StressCurveDS(
#                     self.X[self.splits[fold][0], :],
#                     self.y[self.splits[fold][0], :]),
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=self.num_workers,
#             )
#
#     def val_dataloader(self, fold=0):
#         if self.train_all:
#             return None
#         else:
#             return DataLoader(
#                 StressCurveDS(
#                     self.X[self.splits[fold][1], :],
#                     self.y[self.splits[fold][1], :]),
#                 batch_size=self.batch_size,
#                 shuffle=False,
#                 num_workers=self.num_workers,
#             )

