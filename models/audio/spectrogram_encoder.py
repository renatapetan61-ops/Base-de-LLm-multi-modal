"""
Apolo Zenith 1.9 — Audio Spectrogram Encoder.
Transforma dados de áudio / espectrogramas mel em sequências de tokens latentes de áudio.
"""

import torch
import torch.nn as nn

class AudioSpectrogramEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        embed_dim: int = 512,
        conv_channels: int = 128,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        
        # Convoluções 1D com pooling para condensar a dimensão temporal de áudio
        self.conv1 = nn.Conv1d(n_mels, conv_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, embed_dim, kernel_size=3, stride=2, padding=1)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: Tensor com formato (batch_size, n_mels, time_steps)
        Returns:
            audio_tokens: Tensor com formato (batch_size, downsampled_time, embed_dim)
        """
        x = self.activation(self.conv1(mel_spectrogram))
        x = self.activation(self.conv2(x))
        # (B, embed_dim, T_down) -> (B, T_down, embed_dim)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x
