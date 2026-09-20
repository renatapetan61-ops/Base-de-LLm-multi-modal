"""
Apolo Zenith 1.9 — Spatiotemporal Video Encoder.
Converte volumes de vídeo 4D (batch, canais, frames, altura, largura) em tokens espaço-temporais.
"""

import torch
import torch.nn as nn

class SpatiotemporalVideoEncoder(nn.Module):
    def __init__(
        self,
        temporal_frames: int = 16,
        image_size: int = 256,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.temporal_frames = temporal_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        
        # Convolução 3D para extração conjunta temporal e espacial
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False
        )
        num_patches_t = temporal_frames // temporal_patch_size
        num_patches_h = image_size // patch_size
        num_patches_w = image_size // patch_size
        self.total_tokens = num_patches_t * num_patches_h * num_patches_w
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_tokens, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_tensor: Tensor com formato (batch_size, in_channels, T, H, W)
        Returns:
            video_tokens: Tensor com formato (batch_size, num_tokens, embed_dim)
        """
        # (B, embed_dim, T_p, H_p, W_p)
        x = self.proj(video_tensor)
        # Flatten para (B, embed_dim, total_tokens) e transpose para (B, total_tokens, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        if x.size(1) <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :x.size(1), :]
        x = self.norm(x)
        return x
