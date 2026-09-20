"""
Apolo Zenith 1.9 — Vision Patch Encoder.
Transforma imagens 2D em sequências de tokens visuais projetados no espaço de raciocínio.
"""

import torch
import torch.nn as nn

class VisionPatchEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Projeção de patches 2D via convolução 2D com stride = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor com formato (batch_size, in_channels, H, W)
        Returns:
            visual_tokens: Tensor com formato (batch_size, num_patches, embed_dim)
        """
        # (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(pixel_values)
        # Flatten para (B, embed_dim, num_patches) e transpose para (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Adição de embeddings posicionais aprendidos
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.norm(x)
        return x
