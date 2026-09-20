"""
Apolo Zenith 1.9 — Camada Sparse Mixture-of-Experts (MoE) com Shared Expert.
Combina múltiplos experts roteados (routed experts) com um expert compartilhado (shared expert)
garantindo retenção de conhecimentos universais e especialização esparsa.
"""

import torch
import torch.nn as nn
from typing import Tuple
from models.moe.expert import SwiGLUExpert
from models.moe.router import TopKRouter

class ZenithMoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 1,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        
        # Router
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            aux_loss_coef=aux_loss_coef
        )
        
        # Routed Experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model=d_model, d_ffn=d_ffn)
            for _ in range(num_experts)
        ])
        
        # Shared Expert (executado sempre para reter raciocínio basal e sintaxe)
        if num_shared_experts > 0:
            self.shared_expert = SwiGLUExpert(d_model=d_model, d_ffn=d_ffn * num_shared_experts)
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor de ativações com formato (batch_size, seq_len, d_model)
        Returns:
            output: Tensor combinado com formato (batch_size, seq_len, d_model)
            aux_loss: Perda escalar de balanceamento de carga do router
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        topk_weights, topk_indices, aux_loss = self.router(x_flat)
        
        final_output = torch.zeros_like(x_flat)
        
        # Execução esparsa vetorizada por expert
        for expert_idx in range(self.num_experts):
            # Identifica quais tokens escolheram este expert em qualquer das posições Top-K
            token_mask = (topk_indices == expert_idx)
            if not token_mask.any():
                continue
                
            token_rows, k_pos = torch.where(token_mask)
            selected_tokens = x_flat[token_rows]
            
            # Executa a FFN do expert
            expert_out = self.experts[expert_idx](selected_tokens)
            
            # Pesa pelo escore do router
            weights = topk_weights[token_rows, k_pos].unsqueeze(-1)
            final_output.index_add_(0, token_rows, expert_out * weights)
            
        # Adiciona a contribuição do Shared Expert
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            final_output = final_output + shared_out
            
        return final_output.view(batch_size, seq_len, d_model), aux_loss
