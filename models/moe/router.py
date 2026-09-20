"""
Apolo Zenith 1.9 — Top-K Gating Router com Perda Auxiliar de Balanceamento de Carga.
Garante que todos os experts recebam treinamento adequado e previne o colapso
de roteamento (expert starvation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, aux_loss_coef: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor de entrada com formato (batch_size * seq_len, d_model)
        Returns:
            weights: Pesos normalizados para os Top-K experts selecionados (N, top_k)
            indices: Índices dos Top-K experts (N, top_k)
            aux_loss: Perda escalar de balanceamento de carga para otimização conjunta
        """
        # Logits de gating: (N, num_experts)
        logits = self.gate(x)
        
        # Probabilidades Softmax globais
        gate_probs = F.softmax(logits, dim=-1)
        
        # Seleção dos Top-K
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Renormalização dos pesos entre os top-k escolhidos
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Cálculo da perda auxiliar de balanceamento de carga (Auxiliary Load Balancing Loss)
        # L_aux = alpha * E * sum(f_i * P_i)
        # f_i: fração de tokens roteados para expert i
        # P_i: probabilidade média acumulada atribuída ao expert i
        mask = F.one_hot(topk_indices, self.num_experts).sum(dim=1) # (N, num_experts)
        fraction_tokens_per_expert = mask.float().mean(dim=0)
        mean_prob_per_expert = gate_probs.mean(dim=0)
        aux_loss = self.aux_loss_coef * self.num_experts * torch.sum(fraction_tokens_per_expert * mean_prob_per_expert)
        
        return topk_weights, topk_indices, aux_loss
