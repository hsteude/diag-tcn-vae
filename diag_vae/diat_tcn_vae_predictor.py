from diag_vae.tcn_modules import VaeEncoder, Decoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DiagTcnAePredictor(pl.LightningModule):
    def __init__(
        self,
        enc_tcn1_in_dims: list = [51, 50, 40, 30, 20],
        enc_tcn1_out_dims: list = [50, 40, 30, 20, 10],
        enc_tcn2_in_dims: list = [10, 6, 5, 4, 3],
        enc_tcn2_out_dims: list = [6, 5, 4, 3, 1],
        latent_dim: int = 10,
        seq_len_x: int = 1000,
        seq_len_y: int = 250,
        kernel_size: int = 15,
        component_output_dims: list = [3, 3],
        lr: float = 1e-3,
        beta: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()
        self.lr = lr
        self.latent_dim = latent_dim

        dec_tcn1_in_dims = enc_tcn2_out_dims[::-1]
        dec_tcn1_out_dims = enc_tcn2_in_dims[::-1]
        dec_tcn2_in_dims = enc_tcn1_out_dims[::-1]
        dec_tcn2_out_dims = enc_tcn1_in_dims[::-1]

        self.encoder = VaeEncoder(
            tcn1_in_dims=enc_tcn1_in_dims,
            tcn1_out_dims=enc_tcn1_out_dims,
            tcn2_in_dims=enc_tcn2_in_dims,
            tcn2_out_dims=enc_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len_x,
        )

        self.x_decoder = Decoder(
            tcn1_in_dims=dec_tcn1_in_dims,
            tcn1_out_dims=dec_tcn1_out_dims,
            tcn2_in_dims=dec_tcn2_in_dims,
            tcn2_out_dims=dec_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len_x,
        )

        self.y_decoder_ls = nn.ModuleList([])
        for out_dim in component_output_dims:
            tcn_output_dims = dec_tcn2_out_dims
            tcn_output_dims[-1] = out_dim
            self.y_decoder_ls.append(
                Decoder(
                    tcn1_in_dims=dec_tcn1_in_dims,
                    tcn1_out_dims=dec_tcn1_out_dims,
                    tcn2_in_dims=dec_tcn2_in_dims,
                    tcn2_out_dims=tcn_output_dims,
                    kernel_size=kernel_size,
                    latent_dim=latent_dim,
                    seq_len=seq_len_y,
                )
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.x_decoder(z), [deco(z) for deco in self.y_decoder_ls]

    def forward(self, x):
        return self.encode(x)

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        s = e * std + mu
        return s

    def loss_function(self, pzx, x_hat, x, y_ls, y_hat_ls):
        # initiating pz here since we ran into
        # problems when we did it in the init
        pz = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(self.device),
            covariance_matrix=torch.eye(self.latent_dim).to(self.device),
        )

        kl = torch.distributions.kl.kl_divergence(pzx, pz)
        y_loss_ls = [nn.MSELoss()(y_hat, y) for y, y_hat in zip(y_ls, y_hat_ls)]
        x_loss = nn.MSELoss()(x, x_hat)
        loss = torch.mean(torch.stack(y_loss_ls)) + x_loss + self.beta * kl
        return loss, x_loss, y_loss_ls

    def shared_eval(self, x, y_ls):
        mu_z, log_var_z = self.encode(x)
        pzx_sigma = torch.cat(
            [torch.diag(torch.exp(log_var_z[i, :])) for i in range(log_var_z.shape[0])]
        ).view(-1, self.latent_dim, self.latent_dim)
        pzx = torch.distributions.MultivariateNormal(
            loc=mu_z, covariance_matrix=pzx_sigma
        )
        z = self.sample_gaussian(mu=mu_z, logvar=log_var_z)

        x_hat, y_hat_ls = self.decode(z)
        loss, x_loss, y_loss_ls = self.loss_function(pzx, x_hat, x, y_hat_ls, y_ls)
        return z, x_loss, y_loss_ls, loss

    def training_step(self, batch, batch_idx):
        x, _, y_ls = batch
        _, x_loss, y_loss_ls, loss = self.shared_eval(x, y_ls)
        self.log("train_loss", loss)
        self.log("x_loss", x_loss)
        for i, y_loss in enumerate(y_loss_ls):
            self.log(f"y_loss_comp_{i+1}", y_loss)
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
