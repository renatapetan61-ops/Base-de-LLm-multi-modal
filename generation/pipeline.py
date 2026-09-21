"""
Apolo Zenith 1.9 — Zenith Multimodal Pipeline (fachada de alto nível).

Integra Prompt Encoder + Image Engine + Video Engine + Super-Resolução
Generativa e entrega resultados com metadados de resolução completos
(native/upscaled/final). Suporta o pipeline hierárquico 8K:

  Prompt → Geração latente (resolução nativa adaptativa)
         → Refino temporal → Super-Resolução Generativa (×2 por estágio)
         → Saída final (ex.: 7680×4320 quando solicitado e viável).

Também exporta vídeos via ffmpeg (somente codificação — nunca renderização).

STATUS: IMPLEMENTED / REQUIRES TRAINING (pesos aleatórios).
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from generation.common.prompt_encoder import ZenithPromptEncoder
from generation.common.resolution import ResolutionMetadata, plan_resolution
from generation.common.super_resolution import GenerativeSuperResolution, tiled_apply
from generation.image.image_engine import ImageResult, ZenithImageEngine
from generation.video.video_engine import VideoResult, ZenithVideoEngine


class ZenithMultimodalPipeline(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        latent_size: int = 16,
        temporal_frames: int = 16,
        fps: int = 24,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.d_model = d_model
        self.prompt_encoder = ZenithPromptEncoder(d_model=d_model).to(self.device)
        self.image_engine = ZenithImageEngine(
            d_model=d_model, latent_size=latent_size).to(self.device)
        self.video_engine = ZenithVideoEngine(
            embed_dim=d_model, latent_size=latent_size,
            temporal_frames=temporal_frames, fps=fps).to(self.device)
        # estágio SR compartilhado para refino final de frames
        self.sr_stage = GenerativeSuperResolution(in_channels=3).to(self.device)
        self.video_sr_tile = 128

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompts: List[str]):
        return self.prompt_encoder(prompts)  # (tokens (B,L,D), pooled (B,1,D))

    @staticmethod
    def _ratio(w: int, h: int) -> str:
        """Aspect ratio real medido no tensor de saída (nunca reporta algo não gerado)."""
        import math
        g = math.gcd(w, h)
        return f"{w // g}:{h // g}"

    def _sr_video(self, video: torch.Tensor, stages: int) -> torch.Tensor:
        """SR generativa em cada frame, com tiling por memória."""
        B, C, T, H, W = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        for _ in range(stages):
            frames = tiled_apply(self.sr_stage, frames, tile_size=self.video_sr_tile,
                                 overlap=16)
        H2, W2 = frames.shape[2], frames.shape[3]
        return frames.view(B, T, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

    # ------------------------------------------------------------------
    # IMAGEM
    # ------------------------------------------------------------------
    def text_to_image(
        self, prompt: str, quality: str = "standard", aspect_ratio: str = "1:1",
        **kwargs,
    ) -> Dict[str, Any]:
        plan = plan_resolution(quality, aspect_ratio, max_native_side=256)
        tokens, pooled = self._encode_prompt([prompt])
        result: ImageResult = self.image_engine.text_to_image(tokens, **kwargs)
        img = result.image
        native = (img.shape[3], img.shape[2])
        for _ in range(plan["preset"].sr_stages):
            img = self.sr_stage(img)
        meta = ResolutionMetadata(
            native_width=native[0], native_height=native[1],
            upscaled_width=img.shape[3], upscaled_height=img.shape[2],
            final_width=img.shape[3], final_height=img.shape[2],
            quality=quality,
            aspect_ratio=f"requested:{aspect_ratio};actual:{self._ratio(img.shape[3], img.shape[2])}",
            sr_stages=plan["preset"].sr_stages,
        )
        return {"image": img, "metadata": meta.to_dict(), "mode": "text_to_image"}

    def image_to_image(self, image: torch.Tensor, prompt: str, **kwargs) -> Dict[str, Any]:
        tokens, _ = self._encode_prompt([prompt])
        result = self.image_engine.image_to_image(image, tokens, **kwargs)
        return {"image": result.image, "mode": result.mode}

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor, prompt: str, **kw):
        tokens, _ = self._encode_prompt([prompt])
        result = self.image_engine.inpaint(image, mask, tokens, **kw)
        return {"image": result.image, "mode": result.mode}

    def outpaint(self, image: torch.Tensor, target_size, prompt: str, **kw):
        tokens, _ = self._encode_prompt([prompt])
        result = self.image_engine.outpaint(image, target_size, tokens, **kw)
        return {"image": result.image, "mode": result.mode}

    def edit_image(self, image: torch.Tensor, prompt: str, reference: Optional[torch.Tensor] = None, **kw):
        tokens, _ = self._encode_prompt([prompt])
        identity = self.image_engine.extract_identity(reference) if reference is not None else None
        result = self.image_engine.edit(image, tokens, reference_identity=identity, **kw)
        return {"image": result.image, "mode": result.mode}

    # ------------------------------------------------------------------
    # VÍDEO
    # ------------------------------------------------------------------
    def _video_with_sr(
        self, result: VideoResult, quality: str, aspect_ratio: str,
    ) -> Dict[str, Any]:
        plan = plan_resolution(quality, aspect_ratio, max_native_side=256)
        stages = plan["preset"].sr_stages
        video = self._sr_video(result.video, stages) if stages > 0 else result.video
        meta = ResolutionMetadata(
            native_width=result.native_width, native_height=result.native_height,
            upscaled_width=video.shape[4], upscaled_height=video.shape[3],
            final_width=video.shape[4], final_height=video.shape[3],
            quality=quality, sr_stages=stages,
            aspect_ratio=f"requested:{aspect_ratio};actual:{self._ratio(video.shape[4], video.shape[3])}",
        )
        out = result.to_dict()
        out["metadata"] = {**result.metadata, "resolution": meta.to_dict()}
        out["video"] = video
        return out

    def text_to_video(self, prompt: str, quality: str = "preview",
                      aspect_ratio: str = "16:9", **kwargs) -> Dict[str, Any]:
        tokens, pooled = self._encode_prompt([prompt])
        result = self.video_engine.text_to_video(
            prompt, tokens, pooled.squeeze(1), **kwargs)
        return self._video_with_sr(result, quality, aspect_ratio)

    def image_to_video(self, image: torch.Tensor, prompt: str = "",
                       quality: str = "preview", aspect_ratio: str = "16:9",
                       **kwargs) -> Dict[str, Any]:
        tokens, pooled = self._encode_prompt([prompt or "a scene comes alive"])
        result = self.video_engine.image_to_video(image, prompt, tokens, pooled.squeeze(1), **kwargs)
        return self._video_with_sr(result, quality, aspect_ratio)

    def text_image_to_video(self, image: torch.Tensor, prompt: str, **kwargs):
        return self.image_to_video(image, prompt, **kwargs)

    def video_to_video(self, video: torch.Tensor, prompt: str,
                       quality: str = "preview", aspect_ratio: str = "16:9",
                       **kwargs) -> Dict[str, Any]:
        tokens, pooled = self._encode_prompt([prompt])
        result = self.video_engine.video_to_video(video, prompt, tokens, pooled.squeeze(1), **kwargs)
        return self._video_with_sr(result, quality, aspect_ratio)

    def first_last_frame_video(self, first: torch.Tensor, last: torch.Tensor,
                               prompt: str, quality: str = "preview",
                               aspect_ratio: str = "16:9", **kwargs) -> Dict[str, Any]:
        tokens, pooled = self._encode_prompt([prompt])
        result = self.video_engine.first_last_frame(first, last, prompt, tokens,
                                                    pooled.squeeze(1), **kwargs)
        return self._video_with_sr(result, quality, aspect_ratio)

    def extend_video(self, memory, prompt: str, quality: str = "preview",
                     aspect_ratio: str = "16:9", **kwargs) -> Dict[str, Any]:
        tokens, pooled = self._encode_prompt([prompt])
        result = self.video_engine.extend_video(memory, prompt, tokens,
                                                pooled.squeeze(1), **kwargs)
        return self._video_with_sr(result, quality, aspect_ratio)


# ----------------------------------------------------------------------
# Exportação (codificação apenas)
# ----------------------------------------------------------------------
def export_video_mp4(video: torch.Tensor, path: str, fps: int = 24,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Codifica tensor (B, 3, T, H, W) → MP4 H.264 via ffmpeg.
    O ffmpeg é usado SOMENTE como codificador: todo o conteúdo visual vem
    dos tensores produzidos pelos engines generativos.
    """
    frames = video[0].clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
    frames_u8 = (frames * 255).astype(np.uint8)
    T, H, W, _ = frames_u8.shape
    # H.264 exige dimensões pares
    if H % 2 or W % 2:
        Hc, Wc = H - H % 2, W - W % 2
        frames_u8 = frames_u8[:, :Hc, :Wc]
        H, W = Hc, Wc

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "frames.yuv")
        frames_u8.tofile(raw)
        if metadata:
            with open(os.path.join(tmp, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)  # sidecar de metadados
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
            "-i", raw,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-movflags", "+faststart",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg falhou: {proc.stderr[-500:]}")
    # metadados como sidecar JSON junto ao vídeo
    if metadata:
        with open(os.path.splitext(path)[0] + ".metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    return path
