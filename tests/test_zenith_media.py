"""
Testes dos novos motores generativos de Imagem e Vídeo do Apolo Zenith 1.9.

Cobrem os requisitos do plano de correção (seções 17–31):
text_to_image, image_to_image, inpainting, outpainting, reference_image,
high_resolution, text_to_video, image_to_video, video_to_video,
first_last_frame, video_extension, camera_motion, temporal/character
consistency e pipeline 8K.
"""

import unittest

import torch

from generation.common.prompt_encoder import ZenithPromptEncoder
from generation.common.resolution import ASPECT_RATIOS, ResolutionMetadata, plan_resolution
from generation.common.super_resolution import GenerativeSuperResolution, tiled_apply
from generation.image.image_engine import ZenithImageEngine
from generation.pipeline import ZenithMultimodalPipeline
from generation.video.video_engine import ZenithVideoEngine


def _cond(d_model: int = 64):
    enc = ZenithPromptEncoder(d_model=d_model)
    tokens, pooled = enc(["uma mulher caminhando na cidade à noite, chuva leve"])
    return tokens, pooled.squeeze(1)


def _image_engine(d_model: int = 64) -> ZenithImageEngine:
    return ZenithImageEngine(in_channels=4, latent_size=16, d_model=d_model,
                             num_layers=2, num_heads=4, vae_downsample=4, sr_width=16)


def _video_engine(d_model: int = 64) -> ZenithVideoEngine:
    return ZenithVideoEngine(embed_dim=d_model, temporal_frames=2, latent_size=8,
                             latent_channels=4, fps=24, dit_layers=2, dit_heads=4)


class TestImageEngine(unittest.TestCase):
    def test_text_to_image(self):
        tokens, _ = _cond()
        res = _image_engine().text_to_image(tokens, num_steps=2)
        self.assertEqual(res.image.shape, (1, 3, 64, 64))
        self.assertEqual(res.mode, "text_to_image")
        self.assertGreaterEqual(res.image.min().item(), 0.0)
        self.assertLessEqual(res.image.max().item(), 1.0)

    def test_image_to_image(self):
        tokens, _ = _cond()
        img = torch.rand(1, 3, 64, 64)
        res = _image_engine().image_to_image(img, tokens, strength=0.5, num_steps=2)
        self.assertEqual(res.image.shape, (1, 3, 64, 64))
        self.assertEqual(res.mode, "image_to_image")

    def test_inpainting(self):
        tokens, _ = _cond()
        img = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        res = _image_engine().inpaint(img, mask, tokens, num_steps=2)
        self.assertEqual(res.image.shape, (1, 3, 64, 64))

    def test_outpainting(self):
        tokens, _ = _cond()
        img = torch.rand(1, 3, 32, 32)
        res = _image_engine().outpaint(img, (64, 64), tokens, num_steps=2)
        self.assertEqual(res.image.shape, (1, 3, 64, 64))

    def test_reference_image_and_character_consistency(self):
        engine = _image_engine()
        tokens, _ = _cond()
        ref = torch.rand(1, 3, 64, 64)
        identity = engine.extract_identity(ref)
        self.assertEqual(identity.shape, (1, 1, 64))
        img = engine.generate_image(tokens, num_steps=2, reference_identity=identity)
        self.assertEqual(img.shape, (1, 3, 64, 64))

    def test_image_editing(self):
        tokens, _ = _cond()
        img = torch.rand(1, 3, 64, 64)
        res = _image_engine().edit(img, tokens, strength=0.4, num_steps=2)
        self.assertEqual(res.mode, "image_editing")

    def test_high_resolution_super_resolution(self):
        tokens, _ = _cond()
        img = _image_engine().generate_image(tokens, num_steps=2, sr_stages=2)
        self.assertEqual(img.shape, (1, 3, 256, 256))  # 64 × 2 × 2


class TestVideoEngine(unittest.TestCase):
    def test_text_to_video(self):
        tokens, pooled = _cond()
        engine = _video_engine()
        res = engine.text_to_video("close-up de um robô caminhando, tracking shot",
                                   tokens, pooled, num_steps=2)
        self.assertEqual(res.video.shape, (1, 3, 8, 32, 32))
        self.assertEqual(res.mode, "text_to_video")
        self.assertEqual(res.fps, 24)
        self.assertIsNotNone(res.memory)
        self.assertGreaterEqual(res.timeline.shots[0].shot_id, 1)

    def test_image_to_video(self):
        tokens, pooled = _cond()
        engine = _video_engine()
        img = torch.rand(1, 3, 32, 32)
        res = engine.image_to_video(img, "a cena ganha movimento", tokens, pooled,
                                    num_steps=2)
        self.assertEqual(res.video.shape, (1, 3, 8, 32, 32))
        self.assertEqual(res.mode, "image_to_video")

    def test_text_plus_image_to_video(self):
        tokens, pooled = _cond()
        engine = _video_engine()
        img = torch.rand(1, 3, 32, 32)
        res = engine.text_image_to_video(img, "câmera em órbita ao redor da cena",
                                         tokens, pooled, num_steps=2)
        self.assertEqual(res.video.shape[2], 8)

    def test_video_to_video(self):
        tokens, pooled = _cond()
        engine = _video_engine()
        vid = torch.rand(1, 3, 4, 32, 32)
        res = engine.video_to_video(vid, "mais cinematográfico", tokens, pooled,
                                    strength=0.5, num_steps=2)
        self.assertEqual(res.video.shape, (1, 3, 4, 32, 32))

    def test_first_last_frame(self):
        tokens, pooled = _cond()
        engine = _video_engine()
        first = torch.rand(1, 3, 32, 32)
        last = torch.rand(1, 3, 32, 32)
        res = engine.first_last_frame(first, last, "transição suave",
                                      tokens, pooled, num_steps=2)
        self.assertEqual(res.video.shape[2], 8)

    def test_video_extension_with_memory(self):
        tokens, pooled = _cond()
        engine = _video_engine()
        seg1 = engine.text_to_video("personagem andando", tokens, pooled, num_steps=2)
        seg2 = engine.extend_video(seg1.memory, "personagem continua andando",
                                   tokens, pooled, num_steps=2)
        self.assertEqual(seg2.mode, "extend_video")
        self.assertTrue(seg2.metadata["continued_from_memory"])
        self.assertIsNotNone(seg2.memory)

    def test_camera_motion_parsing(self):
        engine = _video_engine()
        mode = engine.camera_engine.parse_camera_prompt(
            "a câmera acompanha em tracking shot lateral")
        self.assertEqual(mode, "tracking_shot")
        traj = engine.camera_engine.get_camera_embedding_sequence("orbit", 8, torch.device("cpu"))
        self.assertEqual(traj.shape, (1, 8, 64))

    def test_motion_engine(self):
        engine = _video_engine()
        motions = engine.motion_engine.parse_motions("uma mulher correndo pela cidade")
        self.assertIn("running", motions)
        field = engine.motion_engine.motion_prior_field(motions, 4, 8, 8, torch.device("cpu"))
        self.assertEqual(field.shape, (1, 2, 4, 8, 8))


class TestResolutionAnd8K(unittest.TestCase):
    def test_resolution_metadata_fields(self):
        meta = ResolutionMetadata(64, 64, 512, 512, 7680, 4320, "8k", "16:9", 3)
        d = meta.to_dict()
        for field in ("native_width", "native_height", "upscaled_width",
                      "upscaled_height", "final_width", "final_height"):
            self.assertIn(field, d)
        self.assertEqual((d["final_width"], d["final_height"]), (7680, 4320))

    def test_quality_presets_and_aspects(self):
        for q in ("preview", "standard", "high", "ultra", "8k"):
            plan = plan_resolution(q, "16:9")
            self.assertIn("preset", plan)
        for ar, (aw, ah) in ASPECT_RATIOS.items():
            plan = plan_resolution("standard", ar)
            m = plan["metadata"]
            self.assertAlmostEqual(m.final_width / m.final_height, aw / ah, delta=0.15)

    def test_8k_pipeline_metadata_not_fake(self):
        """Native 480@8K preset: metadados distinguem nativo de final (sem upscale falso)."""
        plan = plan_resolution("8k", "16:9")
        m = plan["metadata"]
        self.assertLess(m.native_width, m.final_width)
        self.assertEqual((m.final_width, m.final_height), (7680, 4320))

    def test_tiled_super_resolution(self):
        sr = GenerativeSuperResolution(in_channels=3, width=16, num_blocks=1)
        img = torch.rand(1, 3, 64, 64)
        out = tiled_apply(sr, img, tile_size=32, overlap=8)
        self.assertEqual(out.shape, (1, 3, 128, 128))
        self.assertFalse(torch.isnan(out).any())


class TestMultimodalPipeline(unittest.TestCase):
    def test_end_to_end_small(self):
        pipe = ZenithMultimodalPipeline(d_model=64, latent_size=8, temporal_frames=2,
                                        fps=24, device="cpu")
        pipe.image_engine.latent_size = 8
        out = pipe.text_to_video("wide shot de uma cidade futurista, drone shot",
                                 quality="preview", num_steps=2)
        self.assertIn("video", out)
        res_meta = out["metadata"]["resolution"]
        for field in ("native_width", "upscaled_width", "final_width"):
            self.assertIn(field, res_meta)
        self.assertGreater(out["video"].shape[-1], out["native_width"])


if __name__ == "__main__":
    unittest.main()
