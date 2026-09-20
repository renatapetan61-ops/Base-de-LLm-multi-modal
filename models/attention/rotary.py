"""
Apolo Zenith 1.9 — Rotary Position Embeddings (RoPE).
Fornece suporte para rotação complexa de coordenadas 1D, 2D (espaço de imagem)
e 3D (espaço-temporal de vídeo).
"""

import torch
import torch.nn as nn
from typing import Tuple

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) canônico com escalonamento de frequência.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Inverses frequencies: theta_i = base ^ (-2(i-1)/dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device = None, dtype: torch.dtype = torch.float32):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotaciona a metade dos canais para aplicação complexa."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica RoPE nos tensores de Query e Key.
    Formato esperado: (batch_size, num_heads, seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(1) # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
