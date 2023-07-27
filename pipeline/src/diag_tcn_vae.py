from tcn_modules import Decoder, VaeEncoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List


class DiagTcnVAE(pl.LightningModule):
    def __init__(
        self,
        enc_tcn1_in_dims: List[int] = [154, 50, 40, 30, 20],
        enc_tcn1_out_dims: List[int] = [50, 40, 30, 20, 10],
        enc_tcn2_in_dims: List[int] = [10, 6, 5, 4, 3],
        enc_tcn2_out_dims: List[int] = [6, 5, 4, 3, 1],
        dec_tcn1_in_dims: List[int] = [1, 3, 5, 6, 8],
        dec_tcn1_out_dims: List[int] = [3, 5, 6, 8, 10],
        dec_tcn2_in_dims: List[int] = [10, 10, 15, 20, 40],
        dec_tcn2_out_dims: List[int] = [10, 15, 20, 40, 154],
        beta: float = 0,
        latent_dim: int = 10,
        seq_len: int = 500,
        kernel_size: int = 15,
        component_output_dims: list = [3, 3],
        lr: float = 1e-3,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.latent_dim = latent_dim

        self.save_hyperparameters()
        self.lr = lr
        self.component_output_dims = component_output_dims

        self.encoder = VaeEncoder(
            tcn1_in_dims=enc_tcn1_in_dims,
            tcn1_out_dims=enc_tcn1_out_dims,
            tcn2_in_dims=enc_tcn2_in_dims,
            tcn2_out_dims=enc_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
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
                    dropout=dropout
                )
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return [deco(z) for deco in self.decoder_ls]

    def forward(self, x):
        return self.encode(x)

    def loss_function(self, x_ls, x_hat_ls, pzx):
        # initiating pz here since we ran into
        # problems when we did it in the init
        pz = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(self.device),
            covariance_matrix=torch.eye(self.latent_dim).to(self.device),
        )

        kl = torch.distributions.kl.kl_divergence(pzx, pz)
        kl_batch = torch.mean(kl)
        x = torch.cat(x_ls, dim=1)
        x_hat = torch.cat(x_hat_ls, dim=1)
        recon_loss = nn.MSELoss()(x, x_hat)
        loss = recon_loss + self.beta * kl_batch
        return loss, recon_loss, kl_batch

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        s = e * std + mu
        return s

    def predict(self, x, x_ls):
        # x, x_ls, _ = batch
        mu_z, log_var_z = self.encode(x)
        # pzx_sigma = torch.cat(
        #     [torch.diag(torch.exp(log_var_z[i, :])) for i in range(log_var_z.shape[0])]
        # ).view(-1, self.latent_dim, self.latent_dim)
        # pzx = torch.distributions.MultivariateNormal(
        #     loc=mu_z, covariance_matrix=pzx_sigma
        # )
        z = self.sample_gaussian(mu=mu_z, logvar=log_var_z)
        x_hat_ls = self.decode(z)

        comp_mse_ls = [
            torch.mean(torch.mean((x - x_hat) ** 2, dim=2), dim=1)
            for x, x_hat in zip(x_ls, x_hat_ls)
        ]
        return comp_mse_ls, z, x_hat_ls

    def shared_eval(self, x, x_comp_ls):
        mu_z, log_var_z = self.encode(x)
        pzx_sigma = torch.cat(
            [torch.diag(torch.exp(log_var_z[i, :])) for i in range(log_var_z.shape[0])]
        ).view(-1, self.latent_dim, self.latent_dim)
        pzx = torch.distributions.MultivariateNormal(
            loc=mu_z, covariance_matrix=pzx_sigma
        )
        z = self.sample_gaussian(mu=mu_z, logvar=log_var_z)
        x_hat_ls = self.decode(z)
        loss, recon_loss, kl_batch = self.loss_function(x_hat_ls, x_comp_ls, pzx)

        return z, x_hat_ls, loss, recon_loss, kl_batch

    def training_step(self, batch, batch_idx):
        x, x_comp_ls, _ = batch
        _, _, loss, recon_loss, kl_batch = self.shared_eval(x, x_comp_ls)
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl", kl_batch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_comp_ls, _ = batch
        _, _, loss, recon_loss, kl_batch = self.shared_eval(x, x_comp_ls)
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl", kl_batch)
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
