"""Embedding sinusoidal de timestep compartilhado (imagem/vídeo/SR)."""

import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        args = timesteps[:, None].float() * freqs[None, :]
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))
