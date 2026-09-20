"""
Apolo Zenith 1.9 — Super-Resolução Generativa.

Módulo de refino aprendido que recebe uma imagem de baixa resolução e gera
detalhes de alta frequência (não é interpolação bicúbica): o upsample espacial
é seguido de uma rede conv residual condicionada que *sintetiza* detalhe novo a
partir de um ruído latente (processo generativo). Encadeável em estágios (×2 cada).

Inclui processamento por tiles com sobreposição para resoluções extremas (8K)
sem estourar memória.

STATUS: IMPLEMENTED (arquitetura) / REQUIRES TRAINING (pesos aleatórios).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class GenerativeSuperResolution(nn.Module):
    """Estágio ×2 de super-resolução generativa com injeção de ruído latente."""

    def __init__(self, in_channels: int = 3, width: int = 48, num_blocks: int = 4):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels + 1, width, 3, padding=1)  # +1 = canal de ruído
        self.blocks = nn.Sequential(*[_ResBlock(width) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(width, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if noise is None:
            noise = torch.randn(B, 1, up.shape[2], up.shape[3], device=x.device, dtype=x.dtype)
        h = self.in_proj(torch.cat([up, noise], dim=1))
        detail = self.out_proj(self.blocks(h))
        return (up + detail).clamp(0.0, 1.0)


def _weights(h: int, w: int, overlap: int, device, dtype) -> torch.Tensor:
    """Pesos de blending (rampa linear) para fusão de tiles sobrepostos."""
    wy = torch.ones(h, device=device, dtype=dtype)
    wx = torch.ones(w, device=device, dtype=dtype)
    ramp_y = torch.linspace(0.0, 1.0, overlap + 2, device=device, dtype=dtype)[1:-1]
    ramp_x = torch.linspace(0.0, 1.0, overlap + 2, device=device, dtype=dtype)[1:-1]
    wy[:overlap] = torch.minimum(wy[:overlap], ramp_y)
    wy[-overlap:] = torch.minimum(wy[-overlap:], ramp_y.flip(0))
    wx[:overlap] = torch.minimum(wx[:overlap], ramp_x)
    wx[-overlap:] = torch.minimum(wx[-overlap:], ramp_x.flip(0))
    return wy[:, None] * wx[None, :]


@torch.no_grad()
def tiled_apply(
    module: nn.Module,
    x: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 32,
    **kwargs,
) -> torch.Tensor:
    """
    Aplica um módulo de imagem (×2) por tiles com sobreposição, fundindo com
    blending linear para reduzir seams. Usado no pipeline 8K.
    """
    B, C, H, W = x.shape
    if H <= tile_size and W <= tile_size:
        return module(x, **kwargs)

    out_h, out_w = H * 2, W * 2
    acc = torch.zeros(B, C, out_h, out_w, device=x.device, dtype=x.dtype)
    wsum = torch.zeros(1, 1, out_h, out_w, device=x.device, dtype=x.dtype)
    step = tile_size - overlap

    def _starts(length: int) -> list:
        starts = list(range(0, max(length - tile_size, 0) + 1, step))
        if starts[-1] != length - tile_size:
            starts.append(max(length - tile_size, 0))
        return sorted(set(starts))

    for y0 in _starts(H):
        for x0 in _starts(W):
            y1, x1 = y0 + tile_size, x0 + tile_size
            y0c, x0c = max(0, min(y0, H - tile_size)), max(0, min(x0, W - tile_size))
            y1, x1 = y0c + tile_size, x0c + tile_size
            tile = x[:, :, y0c:y1, x0c:x1]
            out_tile = module(tile, **kwargs)
            th, tw = out_tile.shape[2], out_tile.shape[3]
            ov = max(1, min(overlap * 2, th // 4, tw // 4))
            w = _weights(th, tw, ov, x.device, x.dtype)
            oy, ox = y0c * 2, x0c * 2
            acc[:, :, oy:oy + th, ox:ox + tw] += out_tile * w
            wsum[:, :, oy:oy + th, ox:ox + tw] += w

    return acc / wsum.clamp_min(1e-6)
