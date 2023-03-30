import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DiagTsDataset(Dataset):
    def __init__(
        self,
        seq_len_x: int,
        seq_len_y: int,
        data_path: str,
        cols: list[str],
    ):
        df = pd.read_csv(data_path)
        self.df = df[cols]
        self.x = torch.from_numpy(self.df.values.astype(np.float32))
        self.length = len(self.df) - 2 * seq_len_x
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            self.x[index : index + self.seq_len_x, :].T,
            self.x[
                index + self.seq_len_x : index + self.seq_len_x + self.seq_len_y, :
            ].T,
        )


class DiagTsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "path/to/dir",
        batch_size: int = 32,
        seq_len_x: int = 1000,
        seq_len_y: int = 100,
        cols: list[str] = ["sig_1", "sig_2", "sig_3"],
        dl_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_path
        self.batch_size = batch_size
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.cols = cols
        self.num_workers = dl_workers
        self.dataset = DiagTsDataset(
            data_path=self.data_dir,
            seq_len_x=self.seq_len_x,
            seq_len_y=self.seq_len_y,
            cols=self.cols,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
