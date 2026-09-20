"""
Testes dos Encoders e Fusão Multimodal do Apolo Zenith 1.9
"""

import unittest
import torch
from models.vision.patch_encoder import VisionPatchEncoder
from models.video.spatiotemporal_encoder import SpatiotemporalVideoEncoder
from models.audio.spectrogram_encoder import AudioSpectrogramEncoder
from models.multimodal.modality_router import ZenithModalityRouter

class TestMultimodalComponents(unittest.TestCase):
    def test_vision_encoder(self):
        encoder = VisionPatchEncoder(image_size=64, patch_size=16, in_channels=3, embed_dim=128)
        imgs = torch.randn(2, 3, 64, 64)
        tokens = encoder(imgs)
        # 64 / 16 = 4 -> 4x4 = 16 patches
        self.assertEqual(tokens.shape, (2, 16, 128))

    def test_video_encoder(self):
        encoder = SpatiotemporalVideoEncoder(
            temporal_frames=4, image_size=64, patch_size=16, temporal_patch_size=2, in_channels=3, embed_dim=128
        )
        video = torch.randn(2, 3, 4, 64, 64)
        tokens = encoder(video)
        # T_patches = 4 / 2 = 2; H_patches = 4; W_patches = 4 -> 2 * 4 * 4 = 32 patches
        self.assertEqual(tokens.shape, (2, 32, 128))

    def test_audio_encoder(self):
        encoder = AudioSpectrogramEncoder(n_mels=80, embed_dim=128, conv_channels=64)
        mels = torch.randn(2, 80, 32)
        tokens = encoder(mels)
        self.assertEqual(tokens.size(0), 2)
        self.assertEqual(tokens.size(2), 128)

    def test_modality_router(self):
        router = ZenithModalityRouter(d_model=128)
        intent = router.route_intent("Crie um código Python e gere uma imagem cinematográfica")
        self.assertIn("code", intent["active_modalities"])
        self.assertIn("image", intent["active_modalities"])
        self.assertTrue(intent["is_multimodal"])

if __name__ == "__main__":
    unittest.main()
