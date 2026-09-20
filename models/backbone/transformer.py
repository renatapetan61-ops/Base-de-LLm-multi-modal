"""
Apolo Zenith 1.9 — Backbone Transformer com MoE, RoPE, RMSNorm e GQA.
Constitui o núcleo computacional unificado do modelo fundacional.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from models.config import ZenithModelConfig
from models.attention.gqa import ZenithAttention
from models.moe.moe_layer import ZenithMoELayer

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Reduz operações aritméticas removendo o cálculo explícito de média.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class ZenithTransformerBlock(nn.Module):
    def __init__(self, config: ZenithModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = ZenithAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = ZenithMoELayer(
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_shared_experts=config.num_shared_experts,
            aux_loss_coef=config.aux_loss_coef,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Self-Attention
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out, kv_cache = self.self_attn(normed, attention_mask=attention_mask, past_key_value=past_key_value)
        hidden_states = residual + attn_out
        
        # Pre-LN MoE Feed-Forward
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        moe_out, aux_loss = self.moe(normed)
        hidden_states = residual + moe_out
        
        return hidden_states, aux_loss, kv_cache

class ZenithBackbone(nn.Module):
    def __init__(self, config: ZenithModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            ZenithTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Você deve fornecer input_ids ou inputs_embeds.")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
            
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        next_cache = []
        
        for idx, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, aux_loss, kv_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past_kv
            )
            total_aux_loss = total_aux_loss + aux_loss
            if kv_cache is not None:
                next_cache.append(kv_cache)
                
        hidden_states = self.norm(hidden_states)
        return hidden_states, total_aux_loss, next_cache

class ZenithLMHeadModel(nn.Module):
    def __init__(self, config: ZenithModelConfig):
        super().__init__()
        self.config = config
        self.backbone = ZenithBackbone(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights opcional (compartilhamento de pesos entre embedding e saída)
        self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        hidden_states, aux_loss, _ = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift para treinamento autoregressivo causal
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = ce_loss + aux_loss
            
        return logits, loss, aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _, _ = self(input_ids=input_ids)
            next_token_logits = logits[:, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return input_ids
