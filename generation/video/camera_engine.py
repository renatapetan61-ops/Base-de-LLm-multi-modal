"""
Apolo Zenith 1.9 — Zenith Cinematic Camera Engine.

Estende o ZenithCinematicCameraController com:
  - vocabulário cinematográfico completo (shots, ângulos, movimentos, lentes);
  - trajetórias de câmera ao longo do tempo (por frame): pan/tilt/dolly/truck/
    orbit/zoom/crane/handheld são convertidos em uma sequência de embeddings
    (B, T, D) que condicionam o VideoDiT frame a frame.

O controlador original permanece para compatibilidade; este engine produz
trajetórias temporais reais em vez de um único vetor estático.

STATUS: IMPLEMENTED (trajetórias são priors geométricos) / REQUIRES TRAINING
(o efeito da trajetória sobre o conteúdo é aprendido pelo VideoDiT).
"""

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn

# fov, distância, altura e vetores de movimento por segundo
CAMERA_TRAJECTORIES: Dict[str, Dict[str, Any]] = {
    "static":      {"fov": 50, "distance": 3.0, "delta": [0, 0, 0]},
    "close_up":    {"fov": 30, "distance": 1.2, "delta": [0, 0, 0]},
    "medium_shot": {"fov": 50, "distance": 2.5, "delta": [0, 0, 0]},
    "wide_shot":   {"fov": 75, "distance": 6.0, "delta": [0, 0, 0]},
    "extreme_wide_shot": {"fov": 95, "distance": 15.0, "delta": [0, 0, 0]},
    "pov":         {"fov": 60, "distance": 1.0, "delta": [0, 0.05, 0]},
    "over_the_shoulder": {"fov": 45, "distance": 1.8, "delta": [0, 0, 0]},
    "low_angle":   {"fov": 55, "distance": 2.0, "delta": [0, -0.2, 0]},
    "high_angle":  {"fov": 55, "distance": 2.0, "delta": [0, 0.2, 0]},
    "birds_eye":   {"fov": 70, "distance": 12.0, "delta": [0, 0.3, 0]},
    "tracking_shot": {"fov": 50, "distance": 3.0, "delta": [1.0, 0, 0]},
    "dolly":       {"fov": 50, "distance": 3.0, "delta": [0, 0, -0.8]},
    "dolly_in":    {"fov": 45, "distance": 3.0, "delta": [0, 0, -0.8]},
    "dolly_out":   {"fov": 55, "distance": 2.0, "delta": [0, 0, 0.8]},
    "truck":       {"fov": 50, "distance": 3.0, "delta": [0.8, 0, 0]},
    "pan":         {"fov": 50, "distance": 3.0, "delta": [0.5, 0, 0], "rotate": "yaw"},
    "pan_left":    {"fov": 50, "distance": 3.0, "delta": [-0.5, 0, 0], "rotate": "yaw"},
    "pan_right":   {"fov": 50, "distance": 3.0, "delta": [0.5, 0, 0], "rotate": "yaw"},
    "tilt":        {"fov": 50, "distance": 3.0, "delta": [0, 0.5, 0], "rotate": "pitch"},
    "tilt_up":     {"fov": 50, "distance": 3.0, "delta": [0, 0.5, 0], "rotate": "pitch"},
    "tilt_down":   {"fov": 50, "distance": 3.0, "delta": [0, -0.5, 0], "rotate": "pitch"},
    "zoom":        {"fov": 30, "distance": 3.0, "delta": [0, 0, -1.0], "fov_delta": -20},
    "zoom_in":     {"fov": 30, "distance": 3.0, "delta": [0, 0, -1.0], "fov_delta": -20},
    "zoom_out":    {"fov": 70, "distance": 2.0, "delta": [0, 0, 1.0], "fov_delta": 20},
    "orbit":       {"fov": 50, "distance": 3.0, "delta": [0, 0, 0], "orbit": True},
    "crane":       {"fov": 55, "distance": 4.0, "delta": [0.3, 0.8, 0]},
    "crane_up":    {"fov": 60, "distance": 3.5, "delta": [0, 0.9, 0]},
    "crane_down":  {"fov": 55, "distance": 6.0, "delta": [0, -0.9, 0]},
    "handheld":    {"fov": 55, "distance": 2.5, "delta": [0, 0, 0], "noise": 0.15},
    "steadicam":   {"fov": 50, "distance": 2.8, "delta": [0.6, 0.05, 0], "noise": 0.03},
    "drone":       {"fov": 70, "distance": 10.0, "delta": [0.4, 0.2, -0.4]},
    "macro":       {"fov": 20, "distance": 0.3, "delta": [0, 0, 0]},
    "telephoto":   {"fov": 15, "distance": 20.0, "delta": [0, 0, 0]},
    "wide_angle":  {"fov": 100, "distance": 2.0, "delta": [0, 0, 0]},
    "rack_focus":  {"fov": 40, "distance": 2.0, "delta": [0, 0, 0], "focus_shift": 1.0},
}


class ZenithCameraEngine(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.frame_proj = nn.Sequential(
            nn.Linear(8, embed_dim // 2), nn.SiLU(), nn.Linear(embed_dim // 2, embed_dim)
        )

    def parse_camera_prompt(self, prompt: str) -> str:
        import re
        text = prompt.lower().replace("-", "_")
        best, best_pos = "medium_shot", float("inf")
        for name in CAMERA_TRAJECTORIES:
            for var in (name, name.replace("_", " ")):
                m = re.search(rf"(?<!\w){re.escape(var)}(?!\w)", text)
                if m and m.start() < best_pos:
                    best, best_pos = name, m.start()
        return best

    def trajectory(self, mode: str, num_frames: int, device: torch.device) -> torch.Tensor:
        """
        Gera trajetória de câmera por frame.
        Returns: (1, T, 8) com [fov_n, dist_n, dx, dy, dz, yaw, pitch, focus].
        """
        spec = CAMERA_TRAJECTORIES[mode]
        t = torch.linspace(0, 1, num_frames, device=device)
        fov = spec["fov"] / 100.0
        dist = spec["distance"] / 20.0
        dx, dy, dz = spec["delta"]

        pos_x = dx * t
        pos_y = dy * t
        pos_z = dz * t
        yaw = torch.zeros_like(t)
        pitch = torch.zeros_like(t)
        focus = torch.full_like(t, spec.get("focus_shift", 0.0))

        if spec.get("rotate") == "yaw":
            yaw = dx * t
        if spec.get("rotate") == "pitch":
            pitch = dy * t
        if spec.get("orbit"):
            pos_x = torch.sin(2 * math.pi * t) * (dist * 5)
            pos_z = (torch.cos(2 * math.pi * t) - 1) * (dist * 5)
            yaw = 2 * math.pi * t
        fov_seq = torch.full_like(t, fov) + (spec.get("fov_delta", 0) / 100.0) * t
        dist_seq = torch.full_like(t, dist) + pos_z

        if spec.get("noise"):
            g = torch.Generator(device=device)
            shake = spec["noise"] * torch.randn(num_frames, device=device, generator=g)
            pos_x = pos_x + shake.cumsum(0) * 0.05

        traj = torch.stack([fov_seq, dist_seq, pos_x, pos_y, pos_z, yaw, pitch, focus], dim=-1)
        return traj.unsqueeze(0)

    def get_camera_embedding_sequence(
        self, mode: str, num_frames: int, device: torch.device
    ) -> torch.Tensor:
        """(1, T, D) — embedding de câmera por frame."""
        traj = self.trajectory(mode, num_frames, device)
        return self.frame_proj(traj)
