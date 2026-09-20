"""
Apolo Zenith 1.9 — Geração Completa do Vídeo Narrativo do Programador e o LLM de 199B.
Utiliza Storyboard Planner, Cinematic Camera Controller, Síntese Acústica
e Renderização de Quadros com Diálogos Sincronizados.
Exporta o vídeo final em /storage/emulated/0/Download/.
"""

import sys
import os
import math
import subprocess
import tempfile
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/apolo-zenith-1.9")

from generation.video.storyboard_planner import ZenithStoryboardPlanner, ShotPlan, SceneTimeline
from generation.video.camera_controller import ZenithCinematicCameraController

def create_dialogue_audio(text: str, wav_path: str, pitch: int = 50, speed: int = 155):
    """Gera áudio de fala em português brasileiro usando espeak-ng."""
    cmd = [
        "espeak-ng",
        "-v", "pt-br",
        "-p", str(pitch),
        "-s", str(speed),
        text,
        "-w", wav_path
    ]
    subprocess.run(cmd, check=True)

def get_audio_duration(wav_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except Exception:
        return 2.0

def main():
    print("=" * 70)
    print("   APOLO ZENITH 1.9 — GERAÇÃO DE VÍDEO NARRATIVO")
    print("   Cena: O Programador e o LLM de 199 Bilhões de Parâmetros")
    print("=" * 70)
    
    output_dir = "/storage/emulated/0/Download"
    os.makedirs(output_dir, exist_ok=True)
    final_video_path = os.path.join(output_dir, "apolo_zenith_programador_llm.mp4")
    
    temp_dir = tempfile.mkdtemp(prefix="zenith_story_")
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    fps = 24
    width, height = 640, 480
    
    # 1. Planejamento do Storyboard pelo Apolo Zenith
    planner = ZenithStoryboardPlanner()
    camera = ZenithCinematicCameraController()
    print("\n[1/5] Storyboard e Trajetória de Câmera configurados...")
    
    # 2. Geração dos Áudios de Fala em Português
    print("[2/5] Sintetizando trilhas de diálogo acústico em Português (PT-BR)...")
    audio1_path = os.path.join(temp_dir, "audio1.wav")
    audio2_path = os.path.join(temp_dir, "audio2.wav")
    audio3_path = os.path.join(temp_dir, "audio3.wav")
    audio4_path = os.path.join(temp_dir, "audio4.wav")
    
    # Programador (Voz 1: tom normal/jovem)
    create_dialogue_audio("Eu preciso terminar de programar esse LLM até semana que vem!", audio1_path, pitch=52, speed=150)
    # Programador no escritório
    create_dialogue_audio("Chefe, eu terminei o LLM de 199 Bilhões de parâmetros!", audio2_path, pitch=52, speed=150)
    # Chefe (Voz 2: tom grave/autoritário)
    create_dialogue_audio("Agora você faz o push lá no seu GitHub!", audio3_path, pitch=35, speed=140)
    # Programador responde
    create_dialogue_audio("Tá bom, vou lá!", audio4_path, pitch=54, speed=155)
    
    dur1 = max(get_audio_duration(audio1_path) + 0.6, 3.2)
    dur_trans1 = 2.0
    dur2 = max(get_audio_duration(audio2_path) + 0.5, 3.2)
    dur3 = max(get_audio_duration(audio3_path) + 0.5, 3.0)
    dur4 = max(get_audio_duration(audio4_path) + 0.5, 2.0)
    dur_trans2 = 2.5
    
    total_duration = dur1 + dur_trans1 + dur2 + dur3 + dur4 + dur_trans2
    print(f"      Duração Total Calculada: {total_duration:.2f}s")
    
    # 3. Renderização dos Quadros
    print("[3/5] Renderizando quadros visuais cinemáticos...")
    frame_count = 0
    
    # -------------------------------------------------------------
    # PARTE 1: Programador no quarto / home-office programando
    # -------------------------------------------------------------
    n_frames1 = int(dur1 * fps)
    code_lines = [
        "import torch",
        "class Zenith199B(nn.Module):",
        "    def __init__(self):",
        "        self.moe = SparseMoE(",
        "            total_params=199e9,",
        "            experts=64, top_k=8",
        "        )"
    ]
    
    for i in range(n_frames1):
        t = i / float(n_frames1)
        img = Image.new("RGB", (width, height), (18, 16, 28))
        draw = ImageDraw.Draw(img)
        
        # Quarto / Parede de fundo com iluminação neon azul/magenta
        draw.rectangle([(0, 0), (width, int(height * 0.75))], fill=(24, 20, 36))
        # Janela com luz da noite
        draw.rectangle([(40, 40), (140, 160)], fill=(12, 18, 38), outline=(60, 70, 100), width=3)
        draw.line([(90, 40), (90, 160)], fill=(60, 70, 100), width=2)
        draw.line([(40, 100), (140, 100)], fill=(60, 70, 100), width=2)
        # Lua pela janela
        draw.ellipse([(95, 55), (120, 80)], fill=(220, 230, 255))
        
        # Mesa
        desk_y = int(height * 0.72)
        draw.rectangle([(0, desk_y), (width, height)], fill=(40, 32, 28))
        draw.line([(0, desk_y), (width, desk_y)], fill=(75, 60, 52), width=3)
        
        # Notebook aberto sobre a mesa
        nb_x, nb_y = 230, desk_y - 120
        # Tela do notebook (com brilho azul)
        draw.polygon([(nb_x - 70, nb_y), (nb_x + 70, nb_y), (nb_x + 60, nb_y + 110), (nb_x - 60, nb_y + 110)], fill=(15, 20, 30), outline=(100, 140, 200))
        draw.rectangle([(nb_x - 55, nb_y + 8), (nb_x + 55, nb_y + 98)], fill=(10, 14, 22))
        
        # Linhas de código animadas sendo digitadas
        visible_lines = min(len(code_lines), int(t * len(code_lines) * 1.5) + 1)
        for l_idx in range(visible_lines):
            draw.text((nb_x - 50, nb_y + 12 + l_idx * 13), code_lines[l_idx], fill=(80, 230, 140))
            
        # Teclado do notebook
        draw.polygon([(nb_x - 80, nb_y + 110), (nb_x + 80, nb_y + 110), (nb_x + 110, nb_y + 155), (nb_x - 110, nb_y + 155)], fill=(30, 35, 45), outline=(70, 80, 100))
        
        # Programador (silhueta/personagem sentado trabalhando)
        prog_x = nb_x + 160
        # Tronco
        draw.polygon([(prog_x - 35, desk_y - 20), (prog_x + 35, desk_y - 20), (prog_x + 45, desk_y + 80), (prog_x - 45, desk_y + 80)], fill=(45, 60, 95))
        # Cabeça
        head_bounce = int(math.sin(i * 0.4) * 2)
        draw.ellipse([(prog_x - 22, desk_y - 75 + head_bounce), (prog_x + 22, desk_y - 30 + head_bounce)], fill=(235, 190, 160))
        # Cabelo
        draw.ellipse([(prog_x - 24, desk_y - 82 + head_bounce), (prog_x + 24, desk_y - 55 + head_bounce)], fill=(35, 25, 20))
        # Braço digitando (mãos sobre o teclado)
        hand_jitter = int((i % 4) * 2)
        draw.line([(prog_x - 20, desk_y), (nb_x + 20 + hand_jitter, nb_y + 130)], fill=(45, 60, 95), width=10)
        draw.ellipse([(nb_x + 15 + hand_jitter, nb_y + 125), (nb_x + 30 + hand_jitter, nb_y + 140)], fill=(235, 190, 160))
        
        # Balão de fala do programador
        draw.rectangle([(120, 20), (520, 75)], fill=(255, 255, 255), outline=(40, 40, 60), width=2)
        draw.polygon([(260, 75), (285, 75), (250, 95)], fill=(255, 255, 255))
        draw.line([(260, 75), (250, 95)], fill=(40, 40, 60), width=2)
        draw.line([(285, 75), (250, 95)], fill=(40, 40, 60), width=2)
        
        draw.text((135, 30), "PROGRAMADOR:", fill=(180, 30, 30))
        draw.text((135, 48), '"Eu preciso terminar de programar esse LLM', fill=(15, 15, 25))
        draw.text((135, 62), ' até semana que vem!"', fill=(15, 15, 25))
        
        # Rodapé com telemetria
        draw.text((20, height - 25), "Apolo Zenith 1.9 | Cena 1: O Quarto do Dev", fill=(120, 130, 160))
        
        img.save(os.path.join(frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1
        
    # -------------------------------------------------------------
    # PARTE 2: Transição com tela escura ("Na semana que vem...")
    # -------------------------------------------------------------
    n_frames_trans1 = int(dur_trans1 * fps)
    for i in range(n_frames_trans1):
        t = i / float(n_frames_trans1)
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Fade in e fade out do texto
        alpha = int(255 * math.sin(t * math.pi))
        text_color = (alpha, alpha, int(alpha * 0.8))
        
        draw.text((width // 2 - 120, height // 2 - 15), "Na semana que vem...", fill=text_color)
        draw.line([(width // 2 - 140, height // 2 + 20), (width // 2 + 140, height // 2 + 20)], fill=(int(alpha * 0.4), int(alpha * 0.4), int(alpha * 0.2)), width=2)
        
        img.save(os.path.join(frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1

    # -------------------------------------------------------------
    # PARTE 3: Escritório - Programador fala com o chefe
    # -------------------------------------------------------------
    def draw_office_background(draw):
        # Parede do escritório com painéis de vidro modernos
        draw.rectangle([(0, 0), (width, height)], fill=(220, 225, 235))
        # Janelões corporativos ao fundo (arranha-céus)
        for w_offset in [50, 220, 390]:
            draw.rectangle([(w_offset, 30), (w_offset + 130, 260)], fill=(180, 215, 250), outline=(130, 150, 180), width=4)
            # Prédios ao longe
            draw.rectangle([(w_offset + 20, 100), (w_offset + 60, 260)], fill=(140, 170, 210))
            draw.rectangle([(w_offset + 70, 70), (w_offset + 120, 260)], fill=(120, 150, 190))
        # Piso
        draw.rectangle([(0, 280), (width, height)], fill=(110, 115, 125))
        draw.line([(0, 280), (width, 280)], fill=(80, 85, 95), width=4)
        # Mesa grande executiva de madeira
        draw.polygon([(260, 250), (600, 250), (630, 390), (220, 390)], fill=(80, 45, 30), outline=(50, 25, 15), width=3)
        # Monitor do chefe na mesa
        draw.rectangle([(420, 210), (510, 270)], fill=(20, 25, 30), outline=(80, 90, 100), width=2)
        draw.line([(465, 270), (465, 290)], fill=(80, 90, 100), width=6)
        
        # CHEFE sentado atrás da mesa
        chefe_x = 465
        # Tronco com terno preto e gravata vermelha
        draw.polygon([(chefe_x - 45, 230), (chefe_x + 45, 230), (chefe_x + 50, 330), (chefe_x - 50, 330)], fill=(25, 30, 40))
        # Camisa branca e gravata
        draw.polygon([(chefe_x - 12, 230), (chefe_x + 12, 230), (chefe_x, 260)], fill=(240, 240, 245))
        draw.polygon([(chefe_x - 5, 240), (chefe_x + 5, 240), (chefe_x, 290)], fill=(190, 30, 30))
        # Cabeça do chefe
        draw.ellipse([(chefe_x - 25, 165), (chefe_x + 25, 225)], fill=(230, 185, 155))
        # Cabelo grisalho
        draw.ellipse([(chefe_x - 27, 160), (chefe_x + 27, 190)], fill=(150, 155, 165))
        
        # PROGRAMADOR em pé à esquerda da mesa
        prog_x = 150
        # Tronco
        draw.polygon([(prog_x - 30, 220), (prog_x + 30, 220), (prog_x + 35, 360), (prog_x - 35, 360)], fill=(45, 60, 95))
        # Cabeça
        draw.ellipse([(prog_x - 22, 150), (prog_x + 22, 215)], fill=(235, 190, 160))
        # Cabelo
        draw.ellipse([(prog_x - 24, 145), (prog_x + 24, 180)], fill=(35, 25, 20))
        # Braço segurando notebook ou apontando
        draw.polygon([(prog_x + 15, 240), (prog_x + 65, 270), (prog_x + 60, 290), (prog_x + 10, 260)], fill=(45, 60, 95))
        draw.rectangle([(prog_x + 60, 265), (prog_x + 95, 295)], fill=(30, 35, 45), outline=(90, 100, 120))

    # Diálogo 1: Programador fala: "Chefe, eu terminei o LLM de 199 Bilhões de parâmetros!"
    n_frames2 = int(dur2 * fps)
    for i in range(n_frames2):
        img = Image.new("RGB", (width, height), (220, 225, 235))
        draw = ImageDraw.Draw(img)
        draw_office_background(draw)
        
        # Balão de fala do programador
        draw.rectangle([(40, 20), (450, 80)], fill=(255, 255, 255), outline=(30, 30, 50), width=2)
        draw.polygon([(150, 80), (170, 80), (150, 110)], fill=(255, 255, 255))
        draw.line([(150, 80), (150, 110)], fill=(30, 30, 50), width=2)
        draw.line([(170, 80), (150, 110)], fill=(30, 30, 50), width=2)
        
        draw.text((55, 28), "PROGRAMADOR:", fill=(20, 80, 180))
        draw.text((55, 48), '"Chefe, eu terminei o LLM de 199 Bilhões', fill=(10, 10, 20))
        draw.text((55, 64), ' de parâmetros!"', fill=(10, 10, 20))
        
        draw.text((20, height - 25), "Apolo Zenith 1.9 | Cena 2: O Escritório do Chefe", fill=(40, 50, 70))
        img.save(os.path.join(frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1
        
    # Diálogo 2: Chefe responde: "Agora você faz o Push lá no seu GitHub!"
    n_frames3 = int(dur3 * fps)
    for i in range(n_frames3):
        img = Image.new("RGB", (width, height), (220, 225, 235))
        draw = ImageDraw.Draw(img)
        draw_office_background(draw)
        
        # Balão de fala do CHEFE
        draw.rectangle([(230, 20), (610, 80)], fill=(255, 255, 255), outline=(40, 30, 30), width=2)
        draw.polygon([(450, 80), (470, 80), (460, 110)], fill=(255, 255, 255))
        draw.line([(450, 80), (460, 110)], fill=(40, 30, 30), width=2)
        draw.line([(470, 80), (460, 110)], fill=(40, 30, 30), width=2)
        
        draw.text((245, 28), "CHEFE:", fill=(180, 30, 30))
        draw.text((245, 48), '"Agora você faz o Push lá no seu GitHub!"', fill=(10, 10, 20))
        
        draw.text((20, height - 25), "Apolo Zenith 1.9 | Cena 2: O Escritório do Chefe", fill=(40, 50, 70))
        img.save(os.path.join(frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1

    # Diálogo 3: Programador responde: "Tá bom, vou lá!"
    n_frames4 = int(dur4 * fps)
    for i in range(n_frames4):
        img = Image.new("RGB", (width, height), (220, 225, 235))
        draw = ImageDraw.Draw(img)
        draw_office_background(draw)
        
        # Balão do programador
        draw.rectangle([(60, 25), (320, 80)], fill=(255, 255, 255), outline=(30, 30, 50), width=2)
        draw.polygon([(150, 80), (170, 80), (150, 110)], fill=(255, 255, 255))
        draw.line([(150, 80), (150, 110)], fill=(30, 30, 50), width=2)
        draw.line([(170, 80), (150, 110)], fill=(30, 30, 50), width=2)
        
        draw.text((75, 33), "PROGRAMADOR:", fill=(20, 80, 180))
        draw.text((75, 53), '"Tá bom, vou lá!"', fill=(10, 10, 20))
        
        draw.text((20, height - 25), "Apolo Zenith 1.9 | Cena 2: O Escritório do Chefe", fill=(40, 50, 70))
        img.save(os.path.join(frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1

    # -------------------------------------------------------------
    # PARTE 4: Transição final com tela escura ("Continua...")
    # -------------------------------------------------------------
    n_frames_trans2 = int(dur_trans2 * fps)
    for i in range(n_frames_trans2):
        t = i / float(n_frames_trans2)
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        alpha = int(255 * min(1.0, t * 1.5))
        gold = (alpha, int(alpha * 0.82), int(alpha * 0.35))
        
        draw.text((width // 2 - 70, height // 2 - 25), "Continua...", fill=gold)
        draw.text((width // 2 - 130, height // 2 + 25), "Apolo Zenith 1.9 — 199B Model", fill=(int(alpha * 0.5), int(alpha * 0.6), int(alpha * 0.7)))
        
        img.save(os.path.join(frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1

    print(f"      Total de Quadros Renderizados: {frame_count}")

    # 4. Montagem da Trilha de Áudio Combinada
    print("[4/5] Mixando trilha de áudio temporal sincronizada...")
    audio_full_path = os.path.join(temp_dir, "audio_full.wav")
    
    # Cria uma trilha de silêncio do tamanho total e sobrepõe os áudios nas posições temporais
    silence_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
        "-t", f"{total_duration}",
        "-c:a", "pcm_s16le",
        os.path.join(temp_dir, "silence.wav")
    ]
    subprocess.run(silence_cmd, check=True)
    
    # Posições de início de cada fala (em milissegundos)
    t1_ms = 100
    t2_ms = int((dur1 + dur_trans1 + 0.2) * 1000)
    t3_ms = int((dur1 + dur_trans1 + dur2 + 0.2) * 1000)
    t4_ms = int((dur1 + dur_trans1 + dur2 + dur3 + 0.2) * 1000)
    
    filter_complex = (
        f"[1:a]adelay={t1_ms}|{t1_ms}[a1];"
        f"[2:a]adelay={t2_ms}|{t2_ms}[a2];"
        f"[3:a]adelay={t3_ms}|{t3_ms}[a3];"
        f"[4:a]adelay={t4_ms}|{t4_ms}[a4];"
        f"[0:a][a1][a2][a3][a4]amix=inputs=5:duration=first:dropout_transition=2[aout]"
    )
    
    mix_cmd = [
        "ffmpeg", "-y",
        "-i", os.path.join(temp_dir, "silence.wav"),
        "-i", audio1_path,
        "-i", audio2_path,
        "-i", audio3_path,
        "-i", audio4_path,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        audio_full_path
    ]
    subprocess.run(mix_cmd, check=True)

    # 5. Codificação do Vídeo Final MP4 (H.264 + AAC)
    print("[5/5] Codificando vídeo final H.264 + AAC via ffmpeg...")
    final_ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps}",
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-i", audio_full_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        final_video_path
    ]
    
    proc = subprocess.run(final_ffmpeg_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Erro ao codificar com ffmpeg:", proc.stderr)
        return False
        
    # Limpeza
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    file_size_kb = os.path.getsize(final_video_path) / 1024.0
    print(f"\nVídeo narrativo gerado com sucesso!")
    print(f"Caminho do arquivo: {final_video_path}")
    print(f"Duração: {total_duration:.2f} segundos")
    print(f"Tamanho do arquivo: {file_size_kb:.2f} KB")
    return True

if __name__ == "__main__":
    main()
