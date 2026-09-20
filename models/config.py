"""
Módulo de Configuração Central do Apolo Zenith 1.9.
Carrega configurações declarativas a partir de YAML e valida parâmetros arquiteturais.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml

@dataclass
class ZenithModelConfig:
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    d_ffn: int = 1024
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    max_position_embeddings: int = 4096
    vocab_size: int = 32000
    aux_loss_coef: float = 0.01
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    total_parameters_target: Optional[int] = None
    active_parameters_per_token: Optional[int] = None

@dataclass
class MultimodalConfig:
    latent_dim: int = 512
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    temporal_frames: int = 16
    fps: int = 24
    n_mels: int = 80
    sample_rate: int = 24000

@dataclass
class ZenithConfig:
    name: str = "Apolo Zenith 1.9"
    version: str = "1.9.0"
    active_profile: str = "mobile_proto_125m"
    model: ZenithModelConfig = field(default_factory=ZenithModelConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    generation_timesteps: int = 25
    camera_motions: List[str] = field(default_factory=lambda: [
        "dolly_in", "dolly_out", "pan_left", "pan_right", 
        "tilt_up", "tilt_down", "zoom_in", "zoom_out", "static"
    ])

    @classmethod
    def from_yaml(cls, yaml_path: str, profile_override: Optional[str] = None) -> "ZenithConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
            
        system = raw_data.get("system", {})
        profile_name = profile_override or system.get("active_profile", "mobile_proto_125m")
        profiles = raw_data.get("profiles", {})
        
        if profile_name not in profiles:
            raise ValueError(f"Profile '{profile_name}' not found in configuration profiles: {list(profiles.keys())}")
            
        p_cfg = profiles[profile_name]
        model_cfg = ZenithModelConfig(
            d_model=p_cfg.get("d_model", 512),
            num_layers=p_cfg.get("num_layers", 6),
            num_heads=p_cfg.get("num_heads", 8),
            num_kv_heads=p_cfg.get("num_kv_heads", 2),
            head_dim=p_cfg.get("head_dim", 64),
            d_ffn=p_cfg.get("d_ffn", 1024),
            num_experts=p_cfg.get("num_experts", 8),
            num_experts_per_tok=p_cfg.get("num_experts_per_tok", 2),
            num_shared_experts=p_cfg.get("num_shared_experts", 1),
            max_position_embeddings=p_cfg.get("max_position_embeddings", 4096),
            vocab_size=p_cfg.get("vocab_size", 32000),
            aux_loss_coef=p_cfg.get("aux_loss_coef", 0.01),
            total_parameters_target=p_cfg.get("total_parameters_target"),
            active_parameters_per_token=p_cfg.get("active_parameters_per_token"),
        )
        
        mm_raw = raw_data.get("multimodal", {})
        vis_raw = mm_raw.get("vision", {})
        vid_raw = mm_raw.get("video", {})
        aud_raw = mm_raw.get("audio", {})
        
        mm_cfg = MultimodalConfig(
            latent_dim=mm_raw.get("latent_dim", 512),
            image_size=vis_raw.get("image_size", 256),
            patch_size=vis_raw.get("patch_size", 16),
            in_channels=vis_raw.get("in_channels", 3),
            temporal_frames=vid_raw.get("temporal_frames", 16),
            fps=vid_raw.get("fps", 24),
            n_mels=aud_raw.get("n_mels", 80),
            sample_rate=aud_raw.get("sample_rate", 24000)
        )
        
        gen_raw = raw_data.get("generation", {})
        fm_raw = gen_raw.get("flow_matching", {})
        cam_raw = gen_raw.get("camera", {})
        
        return cls(
            name=system.get("name", "Apolo Zenith 1.9"),
            version=system.get("version", "1.9.0"),
            active_profile=profile_name,
            model=model_cfg,
            multimodal=mm_cfg,
            generation_timesteps=fm_raw.get("time_steps", 25),
            camera_motions=cam_raw.get("supported_motions", [])
        )
