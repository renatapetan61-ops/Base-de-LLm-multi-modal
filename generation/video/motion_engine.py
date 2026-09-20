"""
Apolo Zenith 1.9 — Zenith Motion Engine.

Interpreta descrições de movimento do prompt e produz:
  - motion tokens (vocabulário de movimentos) condicionando o VideoDiT;
  - campos de velocidade latentes (B, 2, T, H, W) como prior de movimento
    (warping guide) — magnitude e direção por região temporal.

Movimentos modelados: walking, running, jumping, falling, flying, driving,
water, fire, smoke, hair/cloth movement, facial expression, hand gestures,
object interaction, camera movement (composto com o Camera Engine).

STATUS: IMPLEMENTED (arquitetura) / REQUIRES TRAINING.
"""

import re
from typing import Dict, List

import torch
import torch.nn as nn

MOTION_VOCAB = [
    "static", "walking", "running", "jumping", "falling", "flying", "driving",
    "water", "fire", "smoke", "hair_movement", "cloth_movement",
    "facial_expression", "hand_gesture", "object_interaction", "camera_movement",
    "rain", "explosion", "dancing", "talking",
]

# sinônimos comuns para parsing simples de prompts (pt/en)
_MOTION_ALIASES = {
    "caminhando": "walking", "walking": "walking", "walk": "walking",
    "correndo": "running", "running": "running", "run": "running",
    "pulando": "jumping", "jumping": "jumping",
    "caindo": "falling", "falling": "falling",
    "voando": "flying", "flying": "flying",
    "dirigindo": "driving", "driving": "driving",
    "agua": "water", "água": "water", "water": "water",
    "fogo": "fire", "fire": "fire",
    "fumaça": "smoke", "fumaca": "smoke", "smoke": "smoke",
    "cabelo": "hair_movement", "hair": "hair_movement",
    "roupa": "cloth_movement", "cloth": "cloth_movement",
    "expressão": "facial_expression", "expression": "facial_expression",
    "gesto": "hand_gesture", "gesture": "hand_gesture",
    "chuva": "rain", "rain": "rain",
    "explosão": "explosion", "explosion": "explosion",
    "dançando": "dancing", "dancing": "dancing",
    "falando": "talking", "talking": "talking",
}

# intensidade base (velocidade de warp) por movimento
_MOTION_MAGNITUDE = {
    "static": 0.0, "walking": 0.3, "running": 0.7, "jumping": 0.8,
    "falling": 0.9, "flying": 0.8, "driving": 0.6, "water": 0.4,
    "fire": 0.5, "smoke": 0.3, "hair_movement": 0.2, "cloth_movement": 0.3,
    "facial_expression": 0.1, "hand_gesture": 0.3, "object_interaction": 0.4,
    "camera_movement": 0.2, "rain": 0.5, "explosion": 0.9, "dancing": 0.6,
    "talking": 0.15,
}


class ZenithMotionEngine(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.motion_embed = nn.Embedding(len(MOTION_VOCAB), embed_dim)
        # projeta (magnitude, direção dominante) como vetor animado no tempo
        self.field_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, 4)
        )  # (dx, dy, dz_temporal, magnitude_gate)

    def parse_motions(self, prompt: str) -> List[str]:
        """Extrai movimentos mencionados no prompt (ordem de ocorrência)."""
        text = prompt.lower()
        found: Dict[int, str] = {}
        for alias, name in _MOTION_ALIASES.items():
            pos = text.find(alias)
            if pos >= 0:
                found.setdefault(pos, name)
        motions = [name for _, name in sorted(found.items())]
        return motions or ["static"]

    def get_motion_embedding(self, motions: List[str], device: torch.device) -> torch.Tensor:
        """Embedding médio dos movimentos detectados: (1, 1, D)."""
        ids = torch.tensor([MOTION_VOCAB.index(m) for m in motions], device=device)
        return self.motion_embed(ids).mean(dim=0, keepdim=True).unsqueeze(0)

    def motion_prior_field(
        self, motions: List[str], T: int, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """
        Campo de velocidade latente (1, 2, T, H, W) usado como prior de warp.
        A fase temporal oscila conforme a magnitude do movimento dominante.
        """
        emb = self.get_motion_embedding(motions, device)          # (1,1,D)
        params = self.field_proj(emb.squeeze(1))                  # (1,4)
        dx, dy, dz, gate = params[0]
        mag = max(_MOTION_MAGNITUDE.get(m, 0.1) for m in motions)
        t = torch.linspace(0, 1, T, device=device)
        phase = torch.sin(2 * torch.pi * (t * (1 + torch.tanh(dz))))
        vx = torch.tanh(dx) * mag * (0.5 + 0.5 * phase)
        vy = torch.tanh(dy) * mag * (0.5 + 0.5 * phase)
        field = torch.stack([vx, vy], dim=0)                      # (2, T)
        field = field.view(1, 2, T, 1, 1).expand(1, 2, T, H, W).contiguous()
        gate_scale = torch.sigmoid(gate)
        return field * gate_scale
