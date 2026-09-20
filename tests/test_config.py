"""
Testes de Configuração do Apolo Zenith 1.9
"""

import unittest
import os
from models.config import ZenithConfig

class TestZenithConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = "/root/apolo-zenith-1.9/configs/apolo_zenith_1_9.yaml"

    def test_load_default_mobile_config(self):
        cfg = ZenithConfig.from_yaml(self.config_path)
        self.assertEqual(cfg.name, "Apolo Zenith 1.9")
        self.assertEqual(cfg.active_profile, "mobile_proto_125m")
        self.assertEqual(cfg.model.d_model, 512)
        self.assertEqual(cfg.model.num_experts, 8)
        self.assertEqual(cfg.model.num_experts_per_tok, 2)

    def test_load_scale_199b_config(self):
        cfg = ZenithConfig.from_yaml(self.config_path, profile_override="scale_199b")
        self.assertEqual(cfg.active_profile, "scale_199b")
        self.assertEqual(cfg.model.d_model, 8192)
        self.assertEqual(cfg.model.num_layers, 64)
        self.assertEqual(cfg.model.num_experts, 64)
        self.assertEqual(cfg.model.num_experts_per_tok, 8)
        self.assertEqual(cfg.model.total_parameters_target, 199000000000)
        self.assertEqual(cfg.model.active_parameters_per_token, 32000000000)

if __name__ == "__main__":
    unittest.main()
