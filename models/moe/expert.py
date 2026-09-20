"""
Apolo Zenith 1.9 — SwiGLU Expert Feed-Forward Network.
Implementa a transformação não-linear SwiGLU:
FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(W_gate(x)) * W_up(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
