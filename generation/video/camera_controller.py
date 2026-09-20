"""
Apolo Zenith 1.9 — Zenith Cinematic Camera Controller.
Interpreta termos de direção de fotografia e cinematografia em linguagem natural,
convertendo-os em parâmetros estruturados de pose, movimento e embeddings de câmera.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn

CAMERA_COMMANDS = {
    "close_up": {"fov": 30.0, "distance": 1.2, "dolly": 0.0, "angle": 0.0},
    "medium_shot": {"fov": 50.0, "distance": 2.5, "dolly": 0.0, "angle": 0.0},
    "wide_shot": {"fov": 75.0, "distance": 6.0, "dolly": 0.0, "angle": 0.0},
    "extreme_wide_shot": {"fov": 95.0, "distance": 15.0, "dolly": 0.0, "angle": 0.0},
    "dolly_in": {"fov": 50.0, "distance": 2.0, "dolly": -1.0, "angle": 0.0},
    "dolly_out": {"fov": 50.0, "distance": 4.0, "dolly": 1.0, "angle": 0.0},
    "pan_left": {"fov": 50.0, "distance": 2.5, "dolly": 0.0, "angle": -30.0},
    "pan_right": {"fov": 50.0, "distance": 2.5, "dolly": 0.0, "angle": 30.0},
    "low_angle": {"fov": 55.0, "distance": 2.0, "dolly": 0.0, "angle": -15.0},
    "high_angle": {"fov": 55.0, "distance": 2.0, "dolly": 0.0, "angle": 25.0},
    "orbit": {"fov": 50.0, "distance": 3.0, "dolly": 0.0, "angle": 180.0},
    "crane_up": {"fov": 60.0, "distance": 3.5, "dolly": 0.0, "angle": 45.0},
    "static": {"fov": 50.0, "distance": 2.5, "dolly": 0.0, "angle": 0.0}
}

class ZenithCinematicCameraController(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        # Projeta [fov, distance, dolly_speed, angle] para o espaço latente
        self.param_proj = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def parse_camera_prompt(self, text: str) -> Dict[str, Any]:
        """Extrai intenções de câmera a partir da descrição da cena priorizando o enquadramento inicial."""
        text_lower = text.lower()
        earliest_idx = float("inf")
        best_mode = "medium_shot"
        
        for cmd_name in CAMERA_COMMANDS:
            variants = [
                cmd_name,
                cmd_name.replace("_", " "),
                cmd_name.replace("_", "-")
            ]
            for var in variants:
                pos = text_lower.find(var)
                if pos != -1 and pos < earliest_idx:
                    earliest_idx = pos
                    best_mode = cmd_name
                    break
                    
        selected_mode = best_mode
        params = CAMERA_COMMANDS[selected_mode]
        return {
            "mode": selected_mode,
            "fov": params["fov"],
            "distance": params["distance"],
            "dolly": params["dolly"],
            "angle": params["angle"]
        }

    def get_camera_embedding(self, camera_params: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Gera o tensor de embedding de câmera para condicionar a atenção temporal."""
        vec = torch.tensor([
            camera_params["fov"] / 100.0,
            camera_params["distance"] / 10.0,
            camera_params["dolly"],
            camera_params["angle"] / 180.0
        ], dtype=torch.float32, device=device).unsqueeze(0) # (1, 4)
        
        return self.param_proj(vec).unsqueeze(1) # (1, 1, embed_dim)
