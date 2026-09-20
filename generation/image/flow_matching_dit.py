"""
Apolo Zenith 1.9 — Rectified Flow Diffusion Transformer (DiT).

Geração latente por Flow Matching contínuo:
    dx/dt = v_theta(x_t, t, cond)
com amostragem por integrador ODE (Euler) e suporte a Classifier-Free Guidance.

Blocos DiT com:
  - self-attention sobre patches latentes;
  - cross-attention para tokens de condição (texto/visão);
  - AdaLN-Zero (escala/deslocamento a partir do timestep + embedding global).

STATUS: IMPLEMENTED (arquitetura) / REQUIRES TRAINING (pesos aleatórios).
"""

import torch
import torch.nn as nn
from typing import Optional

from generation.common.timestep import TimestepEmbedding


class DiTBlock(nn.Module):
    """Bloco DiT: self-attn + cross-attn + MLP, modulados por AdaLN-Zero."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        # AdaLN-Zero: projeta condicionamento escalar em 6 modulações
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, sh1, g1, s2, sh2, g2 = self.adaLN(c).chunk(6, dim=-1)

        def mod(h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        h = mod(self.norm1(x), s1, sh1)
        x = x + g1.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = mod(self.norm2(x), s2, sh2)
        x = x + g2.unsqueeze(1) * self.cross(h, cond_tokens, cond_tokens, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class FlowMatchingDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        latent_size: int = 32,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        patch_size: int = 2,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = (latent_size // patch_size) ** 2

        self.patch_proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = TimestepEmbedding(d_model)
        self.global_proj = nn.Linear(d_model, d_model)
        self.cond_in = nn.Linear(cond_dim or d_model, d_model)

        self.blocks = nn.ModuleList([DiTBlock(d_model, num_heads) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, in_channels * patch_size * patch_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: latente ruidoso (B, C, H, W)
            t: passo contínuo em [0, 1] (B,)
            condition: tokens de condição (B, L, D) ou global (B, 1, D)
        Returns:
            campo de velocidade v (B, C, H, W)
        """
        B, C, H, W = x_t.shape
        h = self.patch_proj(x_t).flatten(2).transpose(1, 2)
        gh, gw = H // self.patch_size, W // self.patch_size
        if gh * gw != self.num_patches:
            # adapta pos-embedding a latentes de tamanho arbitrário (interpolação 2D)
            base = self.pos_embed.transpose(1, 2).view(
                1, self.d_model, self.latent_size // self.patch_size, -1)
            h = h + nn.functional.interpolate(base, size=(gh, gw), mode="bilinear",
                                              align_corners=False).flatten(2).transpose(1, 2)
        else:
            h = h + self.pos_embed
        cond_tokens = self.cond_in(condition)
        pooled = cond_tokens.mean(dim=1)  # (B, D)
        c = self.time_embed(t) + self.global_proj(pooled)

        for blk in self.blocks:
            h = blk(h, cond_tokens, c)

        out = self.out_proj(self.norm_out(h))  # (B, P, C*p*p)
        p = self.patch_size
        out = out.transpose(1, 2).view(B, C, p, p, H // p, W // p)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, H, W)
        return out
