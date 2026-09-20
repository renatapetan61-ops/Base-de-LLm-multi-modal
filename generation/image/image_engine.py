"""
Apolo Zenith 1.9 — Zenith Image Engine (v2).

Pipeline generativo real de imagem baseado em:
  Text/Vision Conditioning → Latente → Flow Matching DiT → VAE Decode
  → Super-Resolução Generativa (opcional, em estágios ×2).

Capacidades:
  - text_to_image
  - image_to_image (strength)
  - inpainting / object removal (mask)
  - outpainting / image extension
  - image editing guiado por prompt
  - reference conditioning (personagem/estilo/objeto) via Identity Engine
  - high resolution (1024² … 8192²) via SR generativa encadeada

STATUS: IMPLEMENTED (arquitetura completa) / REQUIRES TRAINING (pesos
aleatórios até o treinamento dos DiT/VAE/SR em dados reais).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from generation.common.super_resolution import GenerativeSuperResolution, tiled_apply
from generation.common.vae import ImageVAE
from generation.image.consistency_engine import ZenithIdentityConsistencyEngine
from generation.image.flow_matching_dit import FlowMatchingDiT


@dataclass
class ImageResult:
    image: torch.Tensor                    # (B, 3, H, W) em [0, 1]
    native_size: tuple
    upscaled_size: tuple
    mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "native_size": self.native_size,
            "upscaled_size": self.upscaled_size,
            "mode": self.mode,
            "metadata": self.metadata,
        }


class ZenithImageEngine(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        latent_size: int = 32,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        vae_downsample: int = 2,
        sr_width: int = 48,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.d_model = d_model
        self.vae_downsample = vae_downsample

        self.vae = ImageVAE(in_channels=3, latent_channels=in_channels,
                            downsample_factor=vae_downsample)
        self.dit = FlowMatchingDiT(
            in_channels=in_channels,
            latent_size=latent_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.consistency = ZenithIdentityConsistencyEngine(embed_dim=d_model)
        self.super_resolution = GenerativeSuperResolution(in_channels=3, width=sr_width)
        self.identity_token_proj = nn.Linear(in_channels, d_model)

    # ------------------------------------------------------------------
    # Amostragem (Rectified Flow + Euler ODE + CFG)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_latents(
        self,
        condition: torch.Tensor,               # (B, L, D)
        num_steps: int = 16,
        guidance_scale: float = 4.0,
        init_latents: Optional[torch.Tensor] = None,
        start_t: float = 0.0,                  # >0 para img2img (menos denoise)
        mask: Optional[torch.Tensor] = None,   # (B, 1, lh, lw); 1 = regenerar
        masked_source: Optional[torch.Tensor] = None,  # latente da região preservada
        reference_identity: Optional[torch.Tensor] = None,
        latent_hw: Optional[tuple] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = condition.device
        B = condition.size(0)
        lh, lw = latent_hw or (self.latent_size, self.latent_size)

        x = torch.randn(B, self.in_channels, lh, lw, device=device, generator=generator) \
            if init_latents is None else init_latents.clone()

        cond = self.consistency.condition_latents(
            condition, identity_embedding=reference_identity
        ) if reference_identity is not None else condition

        uncond = torch.zeros_like(cond)
        steps = torch.linspace(start_t, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t = steps[i].expand(B)
            v_c = self.dit(x, t, cond)
            if guidance_scale != 1.0:
                v_u = self.dit(x, t, uncond)
                v = v_u + guidance_scale * (v_c - v_u)
            else:
                v = v_c
            x = x + (steps[i + 1] - steps[i]) * v
            if mask is not None and masked_source is not None:
                # inpainting: re-insere a região preservada difusa no passo t
                noise_src = masked_source * (1 - steps[i + 1]) + \
                    torch.randn_like(masked_source) * steps[i + 1]
                x = mask * x + (1 - mask) * noise_src
        return x

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_image(
        self,
        text_condition: torch.Tensor,
        num_steps: int = 16,
        guidance_scale: float = 4.0,
        reference_identity: Optional[torch.Tensor] = None,
        sr_stages: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Text→Image básico. Retorna tensor RGB (B, 3, H, W)."""
        z = self._sample_latents(
            text_condition, num_steps=num_steps, guidance_scale=guidance_scale,
            reference_identity=reference_identity, **kwargs,
        )
        img = self.vae.decode(z)
        for _ in range(sr_stages):
            img = self.super_resolution(img)
        return img

    @torch.no_grad()
    def text_to_image(self, prompt_tokens: torch.Tensor, sr_stages: int = 0, **kwargs) -> ImageResult:
        img = self.generate_image(prompt_tokens, sr_stages=0, **kwargs)
        native = (img.shape[3], img.shape[2])
        for _ in range(sr_stages):
            img = self.super_resolution(img)
        return ImageResult(image=img, native_size=native,
                           upscaled_size=(img.shape[3], img.shape[2]),
                           mode="text_to_image")

    @torch.no_grad()
    def image_to_image(
        self,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        strength: float = 0.6,
        **kwargs,
    ) -> ImageResult:
        z0 = self.vae.encode(image, sample=False)
        start_t = float(torch.clamp(1 - torch.tensor(strength), 0.05, 0.95))
        noise = torch.randn_like(z0)
        x_start = z0 * start_t + noise * (1 - start_t)  # interpolação retificada em t=1-strength
        z = self._sample_latents(prompt_tokens, init_latents=x_start,
                                 start_t=1 - strength, **kwargs)
        img = self.vae.decode(z)
        return ImageResult(image=img, native_size=tuple(img.shape[2:][::-1]),
                           upscaled_size=tuple(img.shape[2:][::-1]), mode="image_to_image")

    @torch.no_grad()
    def inpaint(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,          # (B, 1, H, W), 1 = área a preencher
        prompt_tokens: torch.Tensor,
        **kwargs,
    ) -> ImageResult:
        z0 = self.vae.encode(image, sample=False)
        lh, lw = z0.shape[2], z0.shape[3]
        m = F.interpolate(mask.float(), size=(lh, lw), mode="nearest")
        mask_bin = (m > 0.5).float()
        z = self._sample_latents(prompt_tokens, mask=mask_bin, masked_source=z0, **kwargs)
        z = mask_bin * z + (1 - mask_bin) * z0
        img = self.vae.decode(z)
        return ImageResult(image=img, native_size=tuple(img.shape[2:][::-1]),
                           upscaled_size=tuple(img.shape[2:][::-1]), mode="inpainting")

    @torch.no_grad()
    def outpaint(
        self,
        image: torch.Tensor,
        target_size: tuple,           # (H, W) >= tamanho da imagem
        prompt_tokens: torch.Tensor,
        **kwargs,
    ) -> ImageResult:
        B, _, H, W = image.shape
        th, tw = target_size
        canvas = torch.zeros(B, 3, th, tw, device=image.device, dtype=image.dtype)
        mask = torch.ones(B, 1, th, tw, device=image.device, dtype=image.dtype)
        oy, ox = (th - H) // 2, (tw - W) // 2
        canvas[:, :, oy:oy + H, ox:ox + W] = image
        mask[:, :, oy:oy + H, ox:ox + W] = 0.0
        result = self.inpaint(canvas, mask, prompt_tokens, **kwargs)
        result.mode = "outpainting"
        return result

    @torch.no_grad()
    def edit(
        self,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        strength: float = 0.5,
        reference_identity: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ImageResult:
        """Edição guiada por prompt preservando identidade (face/character)."""
        result = self.image_to_image(
            image, prompt_tokens, strength=strength,
            reference_identity=reference_identity, **kwargs,
        )
        result.mode = "image_editing"
        return result

    @torch.no_grad()
    def extract_identity(self, reference_image: torch.Tensor) -> torch.Tensor:
        """Extrai embedding de identidade de uma imagem de referência (Vision→Identity)."""
        z = self.vae.encode(reference_image, sample=False)          # (B, C, h, w)
        tokens = self.identity_token_proj(z.flatten(2).transpose(1, 2))  # (B, hw, D)
        return self.consistency.extract_identity_embedding(tokens)
