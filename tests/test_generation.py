"""
Testes dos Motores de Geração de Imagem, Vídeo e Áudio do Apolo Zenith 1.9
"""

import unittest
import torch
from generation.image.image_engine import ZenithImageEngine
from generation.video.video_engine import ZenithVideoEngine
from generation.audio.audio_engine import ZenithAudioEngine

class TestGenerativeEngines(unittest.TestCase):
    def test_image_engine_generate(self):
        engine = ZenithImageEngine(in_channels=4, latent_size=16, d_model=128,
                                   num_layers=2, num_heads=4, vae_downsample=2)
        cond = torch.randn(1, 4, 128)
        # Amostragem Euler ODE rápida com 3 passos
        img = engine.generate_image(cond, num_steps=3)
        self.assertEqual(img.shape, (1, 3, 32, 32))

    def test_video_engine_generate(self):
        # Nova arquitetura: ZenithVideoEngine gera vídeo RGB real [B,3,T,H,W]
        # via VideoDiT espaço-temporal + VideoVAE (ver tests/test_zenith_media.py)
        from generation.common.prompt_encoder import ZenithPromptEncoder
        engine = ZenithVideoEngine(embed_dim=128, temporal_frames=2, latent_size=8,
                                   dit_layers=2, dit_heads=4)
        enc = ZenithPromptEncoder(d_model=128)
        tokens, pooled = enc(["close-up dramatic camera dolly in"])
        res = engine.text_to_video("close-up dramatic camera dolly in", tokens,
                                   pooled.squeeze(1), num_steps=2)
        self.assertEqual(res.mode, "text_to_video")
        self.assertEqual(res.fps, 24)
        self.assertEqual(res.video.shape[0], 1)
        self.assertEqual(res.video.shape[1], 3)
        self.assertEqual(res.video.shape[2], 8)   # T=2 latentes × fator temporal 4
        motion = res.video[:, :, 1:] - res.video[:, :, :-1]
        self.assertGreater(motion.abs().mean().item(), 0.0)  # movimento real, não slide

    def test_audio_engine_plan_and_synth(self):
        engine = ZenithAudioEngine(d_model=128)
        plan = engine.plan_audio_cues("Cena de ação épica e passos na chuva", 5.0)
        self.assertEqual(plan.music_mood, "epic_percussion")
        
        tokens = torch.randn(1, 10, 128)
        mel = engine.synthesize_spectrogram(tokens)
        self.assertEqual(mel.shape, (1, 80, 10))

if __name__ == "__main__":
    unittest.main()
