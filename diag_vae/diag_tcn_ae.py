from diag_vae.tcn_modules import Encoder, Decoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DiagTcnAE(pl.LightningModule):
    def __init__(
        self,
        enc_tcn1_in_dims: list = [154, 50, 40, 30, 20],
        enc_tcn1_out_dims: list = [50, 40, 30, 20, 10],
        enc_tcn2_in_dims: list = [10, 6, 5, 4, 3],
        enc_tcn2_out_dims: list = [6, 5, 4, 3, 1],
        dec_tcn1_in_dims: list = [1, 3, 5, 6, 8],
        dec_tcn1_out_dims: list = [3, 5, 6, 8, 10],
        dec_tcn2_in_dims: list = [10, 10, 15, 20, 40],
        dec_tcn2_out_dims: list = [10, 15, 20, 40, 154],
        latent_dim: int = 10,
        seq_len: int = 500,
        kernel_size: int = 15,
        component_output_dims: list = [3, 3],
        lr: float = 1e-3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()
        self.lr = lr


        self.encoder = Encoder(
            tcn1_in_dims=enc_tcn1_in_dims,
            tcn1_out_dims=enc_tcn1_out_dims,
            tcn2_in_dims=enc_tcn2_in_dims,
            tcn2_out_dims=enc_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len,
        )
        self.decoder_ls = nn.ModuleList([])
        for out_dim in component_output_dims:
            tcn_output_dims = dec_tcn2_out_dims
            tcn_output_dims[-1] = out_dim
            self.decoder_ls.append(
                Decoder(
                    tcn1_in_dims=dec_tcn1_in_dims,
                    tcn1_out_dims=dec_tcn1_out_dims,
                    tcn2_in_dims=dec_tcn2_in_dims,
                    tcn2_out_dims=tcn_output_dims,
                    kernel_size=kernel_size,
                    latent_dim=latent_dim,
                    tcn1_seq_len=400,
                    tcn2_seq_len=seq_len,
                )
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return [deco(z) for deco in self.decoder_ls]

    def forward(self, x):
        return self.encode(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, x_ls, _ = batch
        z = self.encode(x)
        x_hat_ls = self.decode(z)

        comp_mse_ls = [
            torch.mean(torch.mean((x - x_hat) ** 2, dim=2), dim=1)
            for x, x_hat in zip(x_ls, x_hat_ls)
        ]
        return comp_mse_ls

    @staticmethod
    def loss_function(x_ls, x_hat_ls):
        # breakpoint()
        x = torch.cat(x_ls, dim=1)
        x_hat = torch.cat(x_hat_ls, dim=1)
        loss = nn.MSELoss()(x, x_hat)
        # loss_ls = [nn.MSELoss()(x, x_hat) for x, x_hat in zip(x_ls, x_hat_ls)]
        # loss = torch.mean(torch.stack(loss_ls))
        return loss

    def shared_eval(self, x, x_comp_ls):
        z = self.encode(x)
        x_hat_ls = self.decode(z)
        loss = self.loss_function(x_hat_ls, x_comp_ls)
        return z, x_hat_ls, loss

    def training_step(self, batch, batch_idx):
        x, x_comp_ls, _ = batch
        _, _, loss = self.shared_eval(x, x_comp_ls)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_comp_ls, _ = batch
        _, _, loss = self.shared_eval(x, x_comp_ls)
        self.log("val_loss", loss)
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
            patience=10,
            min_lr=1e-5,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}
        ]
