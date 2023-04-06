from diag_vae.tcn_modules import Encoder, Decoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class VanillaTcnAE(pl.LightningModule):
    def __init__(
        self,
        enc_tcn1_in_dims: list = [51, 50, 40, 30, 20],
        enc_tcn1_out_dims: list = [50, 40, 30, 20, 10],
        enc_tcn2_in_dims: list = [10, 6, 5, 4, 3],
        enc_tcn2_out_dims: list = [6, 5, 4, 3, 1],
        latent_dim: int = 10,
        kernel_size: int = 15,
        seq_len: int = 500,
        lr: float = 1e-3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()
        self.lr = lr


        dec_tcn1_in_dims = enc_tcn2_out_dims[::-1]
        dec_tcn1_out_dims = enc_tcn2_in_dims[::-1]
        dec_tcn2_in_dims = enc_tcn1_out_dims[::-1]
        dec_tcn2_out_dims = enc_tcn1_in_dims[::-1]

        self.encoder = Encoder(
            tcn1_in_dims=enc_tcn1_in_dims,
            tcn1_out_dims=enc_tcn1_out_dims,
            tcn2_in_dims=enc_tcn2_in_dims,
            tcn2_out_dims=enc_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len,
        )
        self.decoder= Decoder(
            tcn1_in_dims=dec_tcn1_in_dims,
            tcn1_out_dims=dec_tcn1_out_dims,
            tcn2_in_dims=dec_tcn2_in_dims,
            tcn2_out_dims=dec_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len,
        ) 
        

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.encode(x)

    @staticmethod
    def loss_function(x, x_hat):
        return nn.MSELoss()(x, x_hat)

    def shared_eval(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = self.loss_function(x, x_hat)
        return z, x_hat, loss

    def training_step(self, batch, batch_idx):
        x, _  = batch
        _, _, loss = self.shared_eval(x)
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
            min_lr=1e-6,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}
        ]
