"""
Apolo Zenith 1.9 — Zenith Video DiT (Diffusion Transformer espaço-temporal).

Opera sobre o latente de vídeo [B, C, T, H, W] produzido pelo VideoVAE.
Cada bloco aplica, na ordem:
  1. Spatial Self-Attention (dentro de cada frame);
  2. Temporal Self- / Cross-Frame Attention (ao longo do tempo, por posição);
  3. Cross-Attention com tokens de condição (texto / visão / referências);
  4. MLP.
Condicionamento AdaLN-Zero a partir de timestep + embeddings globais de
texto, câmera (por frame), movimento, física e identidade.

STATUS: IMPLEMENTED (arquitetura) / REQUIRES TRAINING (pesos aleatórios).
"""

from typing import Optional

import torch
import torch.nn as nn

from generation.common.timestep import TimestepEmbedding


class _Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, q: torch.Tensor, kv: Optional[torch.Tensor] = None):
        kv = q if kv is None else kv
        return self.attn(q, kv, kv, need_weights=False)[0]


class VideoDiTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm_s = nn.LayerNorm(d_model, elementwise_affine=False)
        self.spatial = _Attention(d_model, num_heads)
        self.norm_t = nn.LayerNorm(d_model, elementwise_affine=False)
        self.temporal = _Attention(d_model, num_heads)
        self.norm_c = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross = _Attention(d_model, num_heads)
        self.norm_m = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 10))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, S, D)
        cond_tokens: torch.Tensor,  # (B, L, D)
        c_frame: torch.Tensor,    # (B, T, D) — AdaLN por frame
    ) -> torch.Tensor:
        B, T, S, D = x.shape
        params = self.adaLN(c_frame)  # (B, T, 10D)
        mods = params.view(B, T, 10, D)

        def mod(h, i):  # aplica escala/shift do par (2i, 2i+1)
            scale, shift = mods[:, :, 2 * i], mods[:, :, 2 * i + 1]
            return h * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)

        # 1) spatial self-attention por frame: (B*T, S, D)
        h = x.reshape(B * T, S, D)
        hs = mod(self.norm_s(x), 0).reshape(B * T, S, D)
        gate_s = mods[:, :, 4].reshape(B * T, 1, D)
        h = h + gate_s * self.spatial(hs)
        x = h.view(B, T, S, D)

        # 2) temporal cross-frame attention por posição: (B*S, T, D)
        h = x.permute(0, 2, 1, 3).reshape(B * S, T, D)
        ht = mod(self.norm_t(x), 1).permute(0, 2, 1, 3).reshape(B * S, T, D)
        gate_t = mods[:, :, 5].unsqueeze(2).expand(B, T, S, D).permute(0, 2, 1, 3).reshape(B * S, T, D)
        h = h + gate_t * self.temporal(ht)
        x = h.view(B, S, T, D).permute(0, 2, 1, 3)

        # 3) cross-attention com tokens de condição
        h = x.reshape(B * T, S, D)
        hc = mod(self.norm_c(x), 2).reshape(B * T, S, D)
        cond_rep = cond_tokens.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, cond_tokens.size(1), D)
        gate_c = mods[:, :, 6].reshape(B * T, 1, D)
        h = h + gate_c * self.cross(hc, cond_rep)
        x = h.view(B, T, S, D)

        # 4) MLP
        gate_m = mods[:, :, 7].unsqueeze(2)
        x = x + gate_m * self.mlp(mod(self.norm_m(x), 3))
        return x


class ZenithVideoDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        max_frames: int = 32,
        latent_size: int = 16,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        patch_size: int = 2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_spatial = (latent_size // patch_size) ** 2

        self.patch_proj = nn.Conv3d(
            latent_channels, d_model,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_spatial, d_model))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        self.time_embed = TimestepEmbedding(d_model)
        # projeções dos condicionamentos globais para o vetor AdaLN por frame
        self.text_proj = nn.Linear(d_model, d_model)
        self.motion_proj = nn.Linear(d_model, d_model)
        self.identity_proj = nn.Linear(d_model, d_model)
        self.cond_in = nn.Linear(d_model, d_model)
        # projeta o campo de movimento (2 canais) para tokens espaciais
        self.motion_field_proj = nn.Conv2d(2, d_model, kernel_size=patch_size, stride=patch_size)

        self.blocks = nn.ModuleList(
            [VideoDiTBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_channels * patch_size * patch_size)

    def forward(
        self,
        x_t: torch.Tensor,                    # (B, C, T, H, W)
        t: torch.Tensor,                      # (B,)
        cond_tokens: torch.Tensor,            # (B, L, D)
        cond_global: torch.Tensor,            # (B, D) texto/global pooled
        camera_seq: Optional[torch.Tensor] = None,   # (B, T, D)
        motion_embed: Optional[torch.Tensor] = None,  # (B, 1, D)
        motion_field: Optional[torch.Tensor] = None,  # (B, 2, T, H, W)
        identity_embed: Optional[torch.Tensor] = None,  # (B, 1, D)
    ) -> torch.Tensor:
        B, C, T, H, W = x_t.shape
        p = self.patch_size
        h = self.patch_proj(x_t)                       # (B, D, T, gh, gw)
        gh, gw = h.shape[3], h.shape[4]
        h = h.permute(0, 2, 3, 4, 1).reshape(B, T, gh * gw, self.d_model)

        if gh * gw == self.num_spatial:
            h = h + self.spatial_pos
        else:
            base = self.spatial_pos.squeeze(1).transpose(1, 2).view(
                1, self.d_model, self.latent_size // p, -1)
            pos = nn.functional.interpolate(base, size=(gh, gw), mode="bilinear",
                                            align_corners=False)
            h = h + pos.flatten(2).transpose(1, 2).unsqueeze(0)

        if motion_field is not None:
            mf = motion_field.permute(0, 2, 1, 3, 4).reshape(B * T, 2, H, W)
            mtok = self.motion_field_proj(mf).flatten(2).transpose(1, 2)  # (B*T, S, D)
            # motion field pode estar em resolução latente própria
            if mtok.shape[1] == gh * gw:
                h = h + mtok.view(B, T, gh * gw, self.d_model)

        # vetor AdaLN por frame
        c = self.time_embed(t).unsqueeze(1).expand(B, T, self.d_model).clone()
        c = c + self.text_proj(cond_global).unsqueeze(1)
        if camera_seq is not None:
            c = c + camera_seq[:, :T]
        if motion_embed is not None:
            c = c + self.motion_proj(motion_embed.squeeze(1)).unsqueeze(1)
        if identity_embed is not None:
            c = c + self.identity_proj(identity_embed.squeeze(1)).unsqueeze(1)

        cond_tokens = self.cond_in(cond_tokens)
        for blk in self.blocks:
            h = blk(h, cond_tokens, c)

        out = self.out_proj(self.norm_out(h))          # (B, T, S, C*p*p)
        out = out.view(B, T, gh, gw, self.latent_channels, p, p)
        out = out.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        out = out.view(B, self.latent_channels, T, gh * p, gw * p)
        return out
