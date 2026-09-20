"""
Apolo Zenith 1.9 — Zenith Prompt Encoder.

Codificador de texto leve e auto-contido usado pelos engines de imagem e vídeo.
Tokenização determinística por hash (sem dependência de tokenizer externo) seguida
de embeddings treináveis e um pequeno Transformer com pooling por atenção.

Saídas:
  - token embeddings (B, L, D)  → cross-attention do DiT
  - embedding global (B, 1, D)  → condicionamento AdaLN

STATUS: IMPLEMENTED (arquitetura) / REQUIRES TRAINING (pesos aleatórios até treino).
"""

import math
import re
import zlib
from typing import List, Tuple

import torch
import torch.nn as nn


class ZenithPromptEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        max_length: int = 64,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def tokenize(self, text: str) -> List[int]:
        """Tokenização determinística por hash de subpalavras (crc32 % vocab)."""
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        ids: List[int] = []
        for w in words:
            ids.append(zlib.crc32(w.encode("utf-8")) % self.vocab_size)
            # sub-tokens de prefixo aumentam robustez morfológica
            if len(w) > 4:
                ids.append(zlib.crc32(w[:4].encode("utf-8")) % self.vocab_size)
        return ids[: self.max_length]

    def encode_tokens(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna (ids (B, L), mask (B, L) com True = padding)."""
        device = self.token_embed.weight.device
        batch_ids, mask = [], []
        for t in texts:
            ids = self.tokenize(t)
            if not ids:
                ids = [0]
            pad = self.max_length - len(ids)
            batch_ids.append(ids + [0] * pad)
            mask.append([False] * len(ids) + [True] * pad)
        ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        return ids_t, mask_t

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token_embeddings: (B, L, D)
            pooled: (B, 1, D)
        """
        ids, pad_mask = self.encode_tokens(texts)
        h = self.token_embed(ids) + self.pos_embed[:, : ids.size(1)]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        pooled, _ = self.pool_attn(
            self.pool_query.expand(h.size(0), -1, -1), h, h, key_padding_mask=pad_mask
        )
        return self.norm(h), self.norm(pooled)
