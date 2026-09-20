"""
Apolo Zenith 1.9 — Variational Autoencoders (Imagem 2D e Vídeo 3D).

Comprimem pixels para o espaço latente usado pelos DiTs de flow matching
e decodificam latentes de volta para pixels.

STATUS: IMPLEMENTED (arquitetura) / REQUIRES TRAINING (pesos aleatórios).
"""

import torch
import torch.nn as nn
from typing import Tuple


def _conv_block(cin: int, cout: int, down: bool = False, dims: int = 2) -> nn.Sequential:
    conv = {2: nn.Conv2d, 3: nn.Conv3d}[dims]
    stride = 2 if down else 1
    return nn.Sequential(
        conv(cin, cout, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
        conv(cout, cout, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
    )


class ImageVAE(nn.Module):
    """VAE convolucional para imagens; compressão espacial configurável."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 32, downsample_factor: int = 4):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        n = int(downsample_factor).bit_length() - 1  # log2

        enc_layers = [_conv_block(in_channels, base_channels, down=False, dims=2)]
        c = base_channels
        for _ in range(n):
            enc_layers.append(_conv_block(c, c * 2, down=True, dims=2))
            c *= 2
        self.encoder = nn.Sequential(*enc_layers)
        self.to_mu = nn.Conv2d(c, latent_channels, 1)
        self.to_logvar = nn.Conv2d(c, latent_channels, 1)

        dec_layers = []
        for _ in range(n):
            dec_layers.append(nn.Sequential(
                nn.ConvTranspose2d(c, c // 2, kernel_size=2, stride=2),
                nn.GroupNorm(min(8, c // 2), c // 2),
                nn.SiLU(),
            ))
            c //= 2
        self.from_latent = nn.Conv2d(latent_channels, base_channels * (2 ** n), 1)
        self.decoder = nn.Sequential(*dec_layers)
        self.to_rgb = nn.Conv2d(c, in_channels, 3, padding=1)

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-20, 20)
        if sample:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        return torch.sigmoid(self.to_rgb(self.decoder(h)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class VideoVAE(nn.Module):
    """VAE 3D (temporal + espacial) para vídeos; latente com shape [B, C, T/ft, H/fs, W/fs]."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 24,
        spatial_factor: int = 4,
        temporal_factor: int = 4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        ts = int(temporal_factor).bit_length() - 1
        ss = int(spatial_factor).bit_length() - 1

        c = base_channels
        enc = [nn.Sequential(
            nn.Conv3d(in_channels, c, 3, padding=1), nn.GroupNorm(min(8, c), c), nn.SiLU()
        )]
        for i in range(max(ts, ss)):
            stride = (2 if i < ts else 1, 2 if i < ss else 1, 2 if i < ss else 1)
            enc.append(nn.Sequential(
                nn.Conv3d(c, c * 2, 3, stride=stride, padding=1),
                nn.GroupNorm(min(8, c * 2), c * 2), nn.SiLU(),
            ))
            c *= 2
        self.encoder = nn.Sequential(*enc)
        self.to_mu = nn.Conv3d(c, latent_channels, 1)
        self.to_logvar = nn.Conv3d(c, latent_channels, 1)

        self.from_latent = nn.Conv3d(latent_channels, c, 1)
        dec = []
        for i in range(max(ts, ss)):
            stride = (2 if max(ts, ss) - 1 - i < ts else 1,
                      2 if max(ts, ss) - 1 - i < ss else 1,
                      2 if max(ts, ss) - 1 - i < ss else 1)
            pad = tuple(1 if s == 2 else 0 for s in stride)
            dec.append(nn.Sequential(
                nn.ConvTranspose3d(c, c // 2, kernel_size=(3, 3, 3), stride=stride,
                                   padding=1, output_padding=pad),
                nn.GroupNorm(min(8, c // 2), c // 2), nn.SiLU(),
            ))
            c //= 2
        self.decoder = nn.Sequential(*dec)
        self.to_rgb = nn.Conv3d(c, in_channels, 3, padding=1)

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """x: (B, C, T, H, W) → z: (B, latent_channels, T/tf, H/sf, W/sf)"""
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-20, 20)
        if sample:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.to_rgb(self.decoder(self.from_latent(z))))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z
