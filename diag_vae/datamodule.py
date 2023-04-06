import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DiagTsDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        data_path: str,
        cols: list,
        comp_a_cols: list,
        comp_b_cols: list,
    ):
        df = pd.read_csv(data_path)
        self.df = df[cols]
        self.x = torch.from_numpy(self.df.values.astype(np.float32))
        self.x_a = torch.from_numpy(self.df[comp_a_cols].values.astype(np.float32))
        self.x_b = torch.from_numpy(self.df[comp_b_cols].values.astype(np.float32))
        self.length = len(self.df) - 2 * seq_len
        self.seq_len = seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            self.x[index : index + self.seq_len, :].T,
            self.x_a[index : index + self.seq_len, :].T,
            self.x_b[index : index + self.seq_len, :].T,
        )


class DiagTsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "path/to/dir",
        batch_size: int = 32,
        seq_len: int = 1000,
        cols: list[str] = ["sig_1", "sig_2", "sig_3", "sig_4", "sig_5", "sig_6"],
        comp_a_cols = ["sig_1", "sig_2", "sig_3"],
        comp_b_cols = ["sig_4", "sig_5", "sig_6"],
        dl_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_path
        self.batch_size = batch_size
        self.seq_len_x = seq_len
        self.cols = cols
        self.num_workers = dl_workers
        self.dataset = DiagTsDataset(
            data_path=self.data_dir,
            seq_len=self.seq_len_x,
            cols=self.cols,
            comp_a_cols=comp_a_cols,
            comp_b_cols=comp_b_cols,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
