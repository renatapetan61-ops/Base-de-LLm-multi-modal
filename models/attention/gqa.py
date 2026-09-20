"""
Apolo Zenith 1.9 — Grouped-Query Attention (GQA).
Reduz drasticamente o consumo de memória de KV-cache mantendo a expressividade
através de compartilhamento de cabeças de chave e valor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models.attention.rotary import RotaryEmbedding, apply_rotary_pos_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repete heads de KV quando num_heads > num_kv_heads."""
    if n_rep == 1:
        return x
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

class ZenithAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)
        current_kv = (k, v)
        
        # Expande K e V para o número de Q heads (GQA)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        
        # Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        else:
            # Máscara causal padrão para decodificação autoregressiva
            causal_mask = torch.triu(torch.full((seq_len, k.shape[-2]), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)
            
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        output = self.o_proj(attn_output)
        return output, current_kv
