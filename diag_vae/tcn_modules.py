import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from pytorch_lightning import seed_everything
seed_everything(42)


class Conv1DResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding,
                 stride, dilation):
        super(Conv1DResidualBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            in_dim, out_dim, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.activattion = nn.ReLU()
        self.conv2 = weight_norm(nn.Conv1d(
            out_dim, out_dim, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.residual = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activattion(out)
        out = self.conv2(out)
        res = self.residual(x)
        out = self.activattion(x+res)
        return out


class TemporalConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding,
                 stride, dilation) -> None:
        super(TemporalConvLayer, self).__init__()
        self.conv = weight_norm(nn.Conv1d(
            in_dim, out_dim, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return self.activation(out)


class TCN(nn.Module):
    def __init__(self, in_dims: list[int], out_dims: list[int],
                 kernel_size: int):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(in_dims)):
            dilation = 2**i
            layers.append(TemporalConvLayer(
                in_dim=in_dims[i],
                kernel_size=kernel_size,
                out_dim=out_dims[i],
                padding='same',
                dilation=dilation,
                stride=1
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
            self, tcn1_in_dims: list[int],  tcn1_out_dims: list[int],
            tcn2_in_dims: list[int], tcn2_out_dims: list[int],
            kernel_size: int = 15,
            latent_dim: int = 10, seq_len: int = 1000) -> None:
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc_in = nn.Linear(in_features=latent_dim,
                               out_features=int(.25*seq_len*tcn1_in_dims[0]))

        # tcn1
        self.upsampler1 = torch.nn.Upsample(scale_factor=2,  mode='nearest')
        self.tcn1 = TCN(in_dims=tcn1_in_dims, out_dims=tcn1_out_dims,
                        kernel_size=kernel_size)

        # tcn1
        self.upsampler2 = torch.nn.Upsample(scale_factor=2,  mode='nearest')
        self.tcn2 = TCN(in_dims=tcn2_in_dims, out_dims=tcn2_out_dims,
                        kernel_size=kernel_size)

    def forward(self, z):
        # fc in
        out = self.fc_in(z).reshape(-1, 1, self.fc_in.out_features)
        out = self.upsampler1(out)
        out = self.tcn1(out)
        out = self.upsampler2(out)
        out = self.tcn2(out)
        return out


class Encoder(nn.Module):
    def __init__(
            self, tcn1_in_dims: list[int],  tcn1_out_dims: list[int],
            tcn2_in_dims: list[int], tcn2_out_dims: list[int],
            kernel_size: int = 15,
            latent_dim: int = 10, seq_len: int = 1000) -> None:

        super(Encoder, self).__init__()

        # TCN1
        self.tcn1 = TCN(in_dims=tcn1_in_dims, out_dims=tcn1_out_dims,
                        kernel_size=kernel_size)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)

        # TCN2
        self.tcn2 = TCN(in_dims=tcn2_in_dims, out_dims=tcn2_out_dims,
                        kernel_size=kernel_size)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc_out = nn.Linear(in_features=int(.25*seq_len*tcn2_out_dims[-1]),
                                out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn1(x)
        out = self.max_pool1(out)
        out = self.tcn2(out)
        out = self.max_pool2(out)
        out = out.flatten(start_dim=1)
        out = self.fc_out(out)
        return out


