"""
Apolo Zenith 1.9 — Zenith Video Engine (v2).

Pipeline generativo real de vídeo baseado em latente espaço-temporal:

  Prompt → Storyboard/Shot Planning → Prompt Encoder → Conditioning
  (câmera por frame + movimento + física + identidade) → VideoDiT
  (rectified flow) → VideoVAE decode → vídeo RGB [B, 3, T, H, W].

Modos suportados:
  - text_to_video
  - image_to_video (a imagem vira o primeiro frame latente — não é zoom/pan)
  - text_image_to_video (imagem + prompt)
  - video_to_video (strength controla quanto preservar)
  - first_last_frame (trajetória entre dois estados)
  - extend_video (Video Continuation Engine com Scene Memory)

STATUS: IMPLEMENTED (arquitetura completa) / REQUIRES TRAINING
(modelos com pesos aleatórios até o treinamento em dados reais).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from generation.common.vae import VideoVAE
from generation.video.camera_engine import ZenithCameraEngine
from generation.video.identity_engine import IdentityBank, ZenithSceneMemory
from generation.video.motion_engine import ZenithMotionEngine
from generation.video.physics_module import ZenithPhysicsModule
from generation.video.storyboard_planner import ZenithStoryboardPlanner
from generation.video.video_dit import ZenithVideoDiT


@dataclass
class VideoResult:
    video: torch.Tensor                     # (B, 3, T, H, W) em [0, 1]
    native_width: int
    native_height: int
    fps: int
    mode: str
    timeline: Any = None
    memory: Optional[ZenithSceneMemory] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video,
            "native_width": self.native_width,
            "native_height": self.native_height,
            "fps": self.fps,
            "mode": self.mode,
            "timeline": self.timeline,
            "metadata": self.metadata,
        }


class ZenithVideoEngine(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        temporal_frames: int = 16,
        latent_size: int = 16,
        latent_channels: int = 4,
        fps: int = 24,
        dit_layers: int = 4,
        dit_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_frames = temporal_frames
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.fps = fps

        self.vae = VideoVAE(latent_channels=latent_channels)
        self.dit = ZenithVideoDiT(
            latent_channels=latent_channels,
            latent_size=latent_size,
            d_model=embed_dim,
            num_layers=dit_layers,
            num_heads=dit_heads,
        )
        self.storyboard_planner = ZenithStoryboardPlanner()
        self.camera_engine = ZenithCameraEngine(embed_dim)
        self.motion_engine = ZenithMotionEngine(embed_dim)
        self.physics = ZenithPhysicsModule(embed_dim)
        self.identity_bank = IdentityBank(embed_dim)

    # ------------------------------------------------------------------
    # Amostragem espaço-temporal (rectified flow + Euler ODE + CFG)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_video(
        self,
        prompt: str,
        cond_tokens: torch.Tensor,            # (B, L, D)
        cond_global: torch.Tensor,            # (B, D)
        num_frames: Optional[int] = None,
        num_steps: int = 12,
        guidance_scale: float = 4.0,
        init_latents: Optional[torch.Tensor] = None,
        start_t: float = 0.0,
        first_frame_latent: Optional[torch.Tensor] = None,   # (B, C, 1, h, w)
        last_frame_latent: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,             # (1, 1, D)
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = cond_tokens.device
        B = cond_tokens.size(0)
        T = num_frames or self.temporal_frames
        H = W = self.latent_size

        motions = self.motion_engine.parse_motions(prompt)
        motion_emb = self.motion_engine.get_motion_embedding(motions, device)
        field = self.motion_engine.motion_prior_field(motions, T, H, W, device)
        field = self.physics.integrate_trajectory(field.expand(B, -1, -1, -1, -1).contiguous())

        cam_mode = self.camera_engine.parse_camera_prompt(prompt)
        cam_seq = self.camera_engine.get_camera_embedding_sequence(cam_mode, T, device)
        cam_seq = cam_seq.expand(B, -1, -1)

        x = torch.randn(B, self.latent_channels, T, H, W, device=device, generator=generator) \
            if init_latents is None else init_latents.clone()

        def _anchors(z: torch.Tensor, tval: float) -> torch.Tensor:
            # re-insere latentes âncora difusos nos passos do ODE
            if first_frame_latent is not None:
                z = z.clone()
                noise = torch.randn_like(z[:, :, :1])
                z[:, :, :1] = first_frame_latent * (1 - tval) + noise * tval
            if last_frame_latent is not None:
                z = z.clone()
                noise = torch.randn_like(z[:, :, -1:])
                z[:, :, -1:] = last_frame_latent * (1 - tval) + noise * tval
            return z

        steps = torch.linspace(start_t, 1.0, num_steps + 1, device=device)
        uncond = torch.zeros_like(cond_tokens)
        for i in range(num_steps):
            t = steps[i].expand(B)
            kw = dict(camera_seq=cam_seq, motion_embed=motion_emb,
                      motion_field=field, identity_embed=identity)
            v_c = self.dit(x, t, cond_tokens, cond_global, **kw)
            if guidance_scale != 1.0:
                v_u = self.dit(x, t, uncond, torch.zeros_like(cond_global), **kw)
                v = v_u + guidance_scale * (v_c - v_u)
            else:
                v = v_c
            x = x + (steps[i + 1] - steps[i]) * v
            x = _anchors(x, steps[i + 1].item())
        return x

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    @torch.no_grad()
    def text_to_video(
        self,
        prompt: str,
        cond_tokens: torch.Tensor,
        cond_global: torch.Tensor,
        num_frames: Optional[int] = None,
        **kwargs,
    ) -> VideoResult:
        timeline = self.storyboard_planner.plan_sequence(prompt)
        z = self._sample_video(prompt, cond_tokens, cond_global, num_frames, **kwargs)
        video = self._decode(z)
        memory = ZenithSceneMemory(
            scene_embedding=cond_global.detach().clone(),
            character_embedding=kwargs.get("identity"),
            environment_embedding=cond_tokens.mean(dim=1, keepdim=True).detach(),
            camera_embedding=None,
            last_frame_latent=z[:, :, -1:].detach(),
        )
        return VideoResult(
            video=video, native_width=video.shape[4], native_height=video.shape[3],
            fps=self.fps, mode="text_to_video", timeline=timeline, memory=memory,
            metadata={"prompt": prompt, "num_frames": video.shape[2]},
        )

    @torch.no_grad()
    def image_to_video(
        self,
        image: torch.Tensor,                  # (B, 3, H, W) — H=W=latent*sf
        prompt: str,
        cond_tokens: torch.Tensor,
        cond_global: torch.Tensor,
        **kwargs,
    ) -> VideoResult:
        first = self.vae.encode(image.unsqueeze(2), sample=False)  # (B, C, 1, h, w)
        # identidade do primeiro frame (protótipo: pooled global do latente)
        identity = None
        if kwargs.pop("preserve_identity", True):
            pooled = first.mean(dim=(3, 4))                        # (B, C, 1)
            identity = torch.zeros(cond_tokens.size(0), 1, self.embed_dim,
                                   device=cond_tokens.device)
            c = min(pooled.shape[1], self.embed_dim)
            identity[:, 0, :c] = pooled[:, :c, 0]
        z = self._sample_video(prompt, cond_tokens, cond_global,
                               first_frame_latent=first, identity=identity, **kwargs)
        video = self._decode(z)
        return VideoResult(
            video=video, native_width=video.shape[4], native_height=video.shape[3],
            fps=self.fps, mode="image_to_video",
            metadata={"prompt": prompt, "num_frames": video.shape[2]},
        )

    text_image_to_video = image_to_video  # T+I2V: mesma rota com prompt obrigatório

    @torch.no_grad()
    def video_to_video(
        self,
        video: torch.Tensor,                  # (B, 3, T, H, W)
        prompt: str,
        cond_tokens: torch.Tensor,
        cond_global: torch.Tensor,
        strength: float = 0.5,
        **kwargs,
    ) -> VideoResult:
        z0 = self.vae.encode(video, sample=False)
        start_t = min(max(1.0 - strength, 0.05), 0.95)
        init = z0 * start_t + torch.randn_like(z0) * (1 - start_t)
        z = self._sample_video(prompt, cond_tokens, cond_global,
                               num_frames=z0.shape[2], init_latents=init,
                               start_t=start_t, **kwargs)
        out = self._decode(z)
        return VideoResult(
            video=out, native_width=out.shape[4], native_height=out.shape[3],
            fps=self.fps, mode="video_to_video",
            metadata={"prompt": prompt, "strength": strength},
        )

    @torch.no_grad()
    def first_last_frame(
        self,
        first_image: torch.Tensor,
        last_image: torch.Tensor,
        prompt: str,
        cond_tokens: torch.Tensor,
        cond_global: torch.Tensor,
        **kwargs,
    ) -> VideoResult:
        first = self.vae.encode(first_image.unsqueeze(2), sample=False)
        last = self.vae.encode(last_image.unsqueeze(2), sample=False)
        z = self._sample_video(prompt, cond_tokens, cond_global,
                               first_frame_latent=first, last_frame_latent=last, **kwargs)
        video = self._decode(z)
        return VideoResult(
            video=video, native_width=video.shape[4], native_height=video.shape[3],
            fps=self.fps, mode="first_last_frame",
            metadata={"prompt": prompt, "num_frames": video.shape[2]},
        )

    @torch.no_grad()
    def extend_video(
        self,
        memory: ZenithSceneMemory,
        prompt: str,
        cond_tokens: torch.Tensor,
        cond_global: torch.Tensor,
        num_frames: Optional[int] = None,
        **kwargs,
    ) -> VideoResult:
        """
        Video Continuation Engine: usa Scene/Character/Environment/Camera Memory
        e o último frame latente do segmento anterior como âncora do primeiro
        frame do novo segmento.
        """
        self.identity_bank.register_identity(
            memory.character_embedding if memory.character_embedding is not None
            else torch.zeros(1, self.embed_dim), slot=0)
        identity = self.identity_bank.aggregate()
        z0 = self._sample_video(
            prompt, cond_tokens, cond_global, num_frames=num_frames,
            first_frame_latent=memory.last_frame_latent,
            identity=identity, **kwargs,
        )
        video = self._decode(z0)
        new_memory = ZenithSceneMemory(
            scene_embedding=cond_global.detach().clone(),
            character_embedding=identity.detach(),
            environment_embedding=memory.environment_embedding,
            camera_embedding=memory.camera_embedding,
            audio_embedding=memory.audio_embedding,
            last_frame_latent=z0[:, :, -1:].detach(),
        )
        return VideoResult(
            video=video, native_width=video.shape[4], native_height=video.shape[3],
            fps=self.fps, mode="extend_video", memory=new_memory,
            metadata={"prompt": prompt, "num_frames": video.shape[2],
                      "continued_from_memory": True},
        )
