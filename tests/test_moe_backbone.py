"""
Testes do Backbone Transformer e Sparse MoE do Apolo Zenith 1.9
"""

import unittest
import torch
from models.config import ZenithModelConfig
from models.moe.expert import SwiGLUExpert
from models.moe.router import TopKRouter
from models.moe.moe_layer import ZenithMoELayer
from models.backbone.transformer import ZenithBackbone, ZenithLMHeadModel

class TestMoEAndBackbone(unittest.TestCase):
    def setUp(self):
        self.config = ZenithModelConfig(
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            d_ffn=256,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            vocab_size=1000,
            max_position_embeddings=512
        )

    def test_swiglu_expert(self):
        expert = SwiGLUExpert(d_model=128, d_ffn=256)
        x = torch.randn(2, 8, 128)
        out = expert(x)
        self.assertEqual(out.shape, (2, 8, 128))

    def test_topk_router_and_aux_loss(self):
        router = TopKRouter(d_model=128, num_experts=4, top_k=2)
        x = torch.randn(16, 128)
        weights, indices, aux_loss = router(x)
        self.assertEqual(weights.shape, (16, 2))
        self.assertEqual(indices.shape, (16, 2))
        self.assertGreaterEqual(aux_loss.item(), 0.0)

    def test_moe_layer(self):
        moe = ZenithMoELayer(d_model=128, d_ffn=256, num_experts=4, num_experts_per_tok=2, num_shared_experts=1)
        x = torch.randn(2, 8, 128)
        out, aux_loss = moe(x)
        self.assertEqual(out.shape, (2, 8, 128))
        self.assertGreaterEqual(aux_loss.item(), 0.0)

    def test_lm_head_model_forward_and_loss(self):
        model = ZenithLMHeadModel(self.config)
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        
        logits, loss, aux_loss = model(input_ids=input_ids, labels=labels)
        self.assertEqual(logits.shape, (2, 16, 1000))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

if __name__ == "__main__":
    unittest.main()
