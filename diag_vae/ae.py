from diag_vae.tcn_modules import Encoder, Decoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Conv1dAE(pl.LightningModule):
    def __init__(
        self,
        in_dim=1,
        latent_dim: int = 10,
        seq_len_x: int = 500,
        seq_len_y: int = 100,
        lr: float = 1e-3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()
        self.lr = lr

        ENC_TCN1_IN_DIMS = [in_dim, 50, 40, 30, 20]
        ENC_TCN1_OUT_DIMS = [50, 40, 30, 20, 10]
        ENC_TCN2_IN_DIMS = [10, 6, 5, 4, 3]
        ENC_TCN2_OUT_DIMS = [6, 5, 4, 3, 1]
        DEC_TCN1_IN_DIMS = ENC_TCN2_OUT_DIMS[::-1]
        DEC_TCN1_OUT_DIMS = ENC_TCN2_IN_DIMS[::-1]
        DEC_TCN2_IN_DIMS = ENC_TCN1_OUT_DIMS[::-1]
        DEC_TCN2_OUT_DIMS = ENC_TCN1_IN_DIMS[::-1]
        KERNEL_SIZE = 15
        self.encoder = Encoder(
            tcn1_in_dims=ENC_TCN1_IN_DIMS,
            tcn1_out_dims=ENC_TCN1_OUT_DIMS,
            tcn2_in_dims=ENC_TCN2_IN_DIMS,
            tcn2_out_dims=ENC_TCN2_OUT_DIMS,
            kernel_size=KERNEL_SIZE,
            latent_dim=latent_dim,
            seq_len=seq_len_x,
        )
        self.x_decoder, self.y_decoder = [Decoder(
            tcn1_in_dims=DEC_TCN1_IN_DIMS,
            tcn1_out_dims=DEC_TCN1_OUT_DIMS,
            tcn2_in_dims=DEC_TCN2_IN_DIMS,
            tcn2_out_dims=DEC_TCN2_OUT_DIMS,
            kernel_size=KERNEL_SIZE,
            latent_dim=latent_dim,
            seq_len=seq_len,
        ) for seq_len in (seq_len_x, seq_len_y)]
        

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.x_decoder(z), self.y_decoder(z)

    def forward(self, x):
        return self.encode(x)

    @staticmethod
    def loss_function(x, x_hat, y, y_hat):
        return nn.MSELoss()(x, x_hat) + nn.MSELoss()(y, y_hat)

    def shared_eval(self, x, y):
        z = self.encode(x)
        x_hat, y_hat = self.decode(z)
        loss = self.loss_function(x, x_hat, y, y_hat)
        return z, x_hat, y_hat, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, _, _, loss = self.shared_eval(x, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers for pytorch lightning."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=25,
            min_lr=1e-5,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}
        ]
