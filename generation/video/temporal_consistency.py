"""
Apolo Zenith 1.9 — Temporal Consistency & Cross-Frame Attention Module.
Elimina flickering, desvio de identidade (identity drift) e inconsistências físicas
através de atenção temporal cruzada e condicionamento de quadros limites (First/Last Frame).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TemporalConsistencyModule(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        video_latents: torch.Tensor,
        first_frame_latent: Optional[torch.Tensor] = None,
        last_frame_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aplica atenção ao longo do eixo temporal para estabilização de movimento.
        Args:
            video_latents: Tensor com formato (batch_size, num_frames, spatial_tokens, embed_dim)
        """
        B, T, S, D = video_latents.shape
        
        # Injeta condicionamento explícito nos quadros terminais
        if first_frame_latent is not None:
            video_latents[:, 0, :, :] = video_latents[:, 0, :, :] + first_frame_latent
        if last_frame_latent is not None:
            video_latents[:, -1, :, :] = video_latents[:, -1, :, :] + last_frame_latent
            
        # Reshape para processar sequências temporais para cada posição espacial
        # (B * S, T, D)
        h = video_latents.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        
        normed = self.norm(h)
        attn_out, _ = self.temporal_attn(normed, normed, normed)
        h = h + attn_out
        
        # Retorna para (B, T, S, D)
        out = h.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()
        return out
