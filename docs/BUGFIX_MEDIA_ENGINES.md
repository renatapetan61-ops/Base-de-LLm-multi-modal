# BUG CRÍTICO — Relatório Técnico: Geração de Imagem e Vídeo (p.md)

## BUG ENCONTRADO

O vídeo `apolo_zenith_programador_llm.mp4` (22s, 640×480, 24fps) não era
produzido por nenhum modelo generativo. Ele era renderizado quadro a quadro por
`scripts/generate_story_video.py` usando sprites 2D com **PIL ImageDraw**
(formas geométricas, texto, efeitos de slide/Ken Burns) e codificado com
ffmpeg. Do mesmo modo, `scripts/generate_video_file.py` e
`scripts/generate_image_file.py` usavam PIL/canvas.

Os "engines" (`generation/image/image_engine.py`,
`generation/video/video_engine.py`) existiam, mas:
- não eram usados para produzir os artefatos entregues;
- o video engine projetava ruído aleatório diretamente para RGB via
  `nn.Linear` — sem VAE, sem espaço latente espaço-temporal treinável, sem
  denoising, sem cross-frame modeling real;
- não havia super-resolução generativa, metadados de resolução, nem pipeline 8K.

## CAUSA RAIZ

Renderização procedural 2D (sprites/slides) no caminho de produção, com os
módulos generativos desconectados do resultado final e subdesenvolvidos.

## CORREÇÃO (arquivos alterados/criados)

- `generation/common/prompt_encoder.py` — ZenithPromptEncoder (tokens + pooled).
- `generation/common/timestep.py` — embedding sinusoidal compartilhado.
- `generation/common/vae.py` — ImageVAE conv e VideoVAE 3D (latente [B,C,T,H,W]).
- `generation/common/resolution.py` — presets preview/standard/high/ultra/8k,
  aspect ratios, `ResolutionMetadata` (native/upscaled/final).
- `generation/common/super_resolution.py` — SR generativa ×2 encadeável +
  `tiled_apply` (tiles com overlap/blending para 8K).
- `generation/image/flow_matching_dit.py` — DiT com self-attn + cross-attn aos
  tokens de condição + AdaLN-Zero (antes: atenção comum aditiva).
- `generation/image/image_engine.py` — v2: text_to_image, image_to_image,
  inpainting, outpainting, edit, extract_identity, sr_stages.
- `generation/video/video_dit.py` — ZenithVideoDiT: atenção espacial + atenção
  temporal cross-frame + cross-attn + AdaLN por frame.
- `generation/video/camera_engine.py` — vocabulário cinematográfico completo
  (shots, dolly, truck, pan, tilt, orbit, crane, handheld, steadicam, drone,
  macro, telephoto, rack focus…) com trajetórias por frame.
- `generation/video/motion_engine.py` — vocabulário de movimentos +
  campo de velocidade latente como prior.
- `generation/video/physics_module.py` — gravidade, inércia, amortecimento,
  colisão nas bordas como priors diferenciáveis.
- `generation/video/identity_engine.py` — IdentityBank + ZenithSceneMemory
  (cena/personagem/ambiente/câmera/áudio + último frame latente).
- `generation/video/video_engine.py` — v2: text_to_video, image_to_video,
  text_image_to_video, video_to_video, first_last_frame, extend_video
  (Continuation Engine com memória).
- `generation/pipeline.py` — fachada multimodal + exportação MP4 (ffmpeg só
  codifica tensores) + metadados sidecar JSON.
- `scripts/generate_video_file.py` — reescrito para usar o engine.
- `scripts/generate_story_video.py` — marcado DEPRECATED (baseline BEFORE).
- `configs/apolo_zenith_1_9.yaml` — seções image/video resolution, qualidades,
  aspect ratios, SR tiling.
- `tests/test_zenith_media.py` — 20 testes cobrindo todos os modos.
- `tests/test_generation.py` — atualizado para o novo contrato dos engines.

## TEXT-IN-IMAGE / EDITING / REFERÊNCIAS

`Zenith Text-in-Image Module`: implementado como condicionamento por tokens
detalhados do Prompt Encoder (cross-attention no DiT) — qualidade tipográfica
REQUIRES TRAINING. Edição suporta prompt-guided edit, inpainting, outpainting e
reference conditioning (personagem/estilo/objeto via `extract_identity`).

## RESULTADOS

- Testes: 40 passed (0 falhas).
- Vídeo AFTER gerado: `data/processed/apolo_zenith_sample_video.mp4`
  (128×128, 24fps, 1s, proto CPU) + `*.metadata.json` com native/upscaled/final.
- Comparação BEFORE/AFTER: BEFORE = sprites 2D estáticos (22s, 640×480);
  AFTER = latentes espaço-temporais [B,C,T,H,W] gerados por VideoDiT com
  movimento real entre frames (sem slides/Ken Burns). O 8K é atingido via
  pipeline hierárquico (nativo adaptativo + SR generativa em estágios com
  tiling), nunca por upscale bicúbico disfarçado.

## LIMITAÇÕES DE HARDWARE

Ambiente: CPU x86-64 (7 cores, 7GB RAM), sem GPU/CUDA. Testes usam dimensões
reduzidas (d_model 64–128, latente 8–16, 2–6 frames latentes). 8K real
(7680×4320) exige GPU(s); aqui apenas o *pipeline* é validado em escala
reduzida com metadados corretos.

## O QUE AINDA REQUER TREINAMENTO

Todas as redes novas têm pesos **aleatórios** (status REQUIRES TRAINING):
VideoDiT, ImageDiT, VAEs, Prompt Encoder, SR generativa, Motion/Physics
refiners. A arquitetura e os pipelines estão implementados e testados (shapes,
condicionamento, âncoras, memória, CFG, amostragem ODE); a qualidade visual
fotorrealista depende de treinamento com datasets reais e GPU.
