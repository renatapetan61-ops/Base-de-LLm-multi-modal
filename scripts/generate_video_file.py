"""
Apolo Zenith 1.9 — Geração de vídeo de exemplo via Zenith Video Engine.

CORRIGIDO: anteriormente este script desenhava sprites 2D com PIL ImageDraw
(o bug crítico). Agora os frames vêm do pipeline generativo latente
(VideoDiT espaço-temporal + VideoVAE + Motion/Camera/Physics Engines).
ffmpeg é usado apenas para codificação H.264 dos tensores gerados.

NOTA HONESTA: pesos ainda não treinados (REQUIRES TRAINING). O vídeo valida o
pipeline (resolução, fps, movimento latente, metadados) — a qualidade visual
fotorrealista depende do treinamento com dados reais.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generation.pipeline import ZenithMultimodalPipeline, export_video_mp4


def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "apolo_zenith_sample_video.mp4")

    pipe = ZenithMultimodalPipeline(d_model=128, latent_size=16, temporal_frames=6, fps=24)
    prompt = (
        "Wide shot cinematográfico de um programador diante de uma interface de LLM, "
        "tracking shot lento, luz azul neon, profundidade de campo rasa"
    )
    print("[Zenith] Gerando vídeo via VideoDiT (latente espaço-temporal)...")
    result = pipe.text_to_video(prompt, quality="preview", num_steps=8)
    video = result["video"]
    print(f"[Zenith] Tensor: {tuple(video.shape)} @ {result['fps']}fps")

    export_video_mp4(video, out_path, fps=result["fps"],
                     metadata=result["metadata"]["resolution"])
    print(f"[Zenith] Vídeo escrito em: {out_path}")
    print("[Zenith] Metadados:", result["metadata"]["resolution"])


if __name__ == "__main__":
    main()
