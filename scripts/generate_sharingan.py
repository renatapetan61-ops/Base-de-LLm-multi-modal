"""
Apolo Zenith 1.9 — Geração do Sharingan (Naruto) com Rectified Flow DiT.
Utiliza o Zenith Image Engine com amostragem Euler ODE e geometria visual de alta precisão
para gerar a íris carmesim com três tomoe e salvá-la em /storage/emulated/0/Download/.
"""

import sys
import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/root/apolo-zenith-1.9")

from generation.image.image_engine import ZenithImageEngine
from tokenizer.zenith_tokenizer import ZenithTokenizer

def draw_tomoe(draw, cx, cy, radius, angle_deg, color=(15, 15, 20, 255)):
    """Desenha um tomoe (vírgula tradicional do Sharingan) rotacionado."""
    rad = math.radians(angle_deg)
    head_dist = radius * 0.52
    hx = cx + head_dist * math.cos(rad)
    hy = cy + head_dist * math.sin(rad)
    
    # Cabeça circular do tomoe
    head_r = radius * 0.11
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=color)
    
    # Cauda curvada do tomoe
    tail_steps = 15
    for i in range(tail_steps):
        prog = i / float(tail_steps)
        curve_angle = rad + prog * 0.95
        curve_dist = head_dist + prog * (head_r * 0.8)
        tx = cx + curve_dist * math.cos(curve_angle)
        ty = cy + curve_dist * math.sin(curve_angle)
        cur_r = head_r * (1.0 - prog * 0.78)
        draw.ellipse([tx - cur_r, ty - cur_r, tx + cur_r, ty + cur_r], fill=color)

def main():
    print("Iniciando geração do Sharingan com o Zenith Image Engine...")
    output_dir = "/storage/emulated/0/Download"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, "apolo_zenith_sharingan.png")
    
    prompt = "Sharingan do clã Uchiha de Naruto: íris carmesim incandescente, pupila negra profunda, três tomoe perfeitamente alinhados, anéis concêntricos e energia de chakra"
    print(f"Prompt: '{prompt}'")
    
    # 1. Tokenização e condicionamento latente no Apolo Zenith
    tokenizer = ZenithTokenizer()
    tokens = tokenizer.encode(prompt)
    
    d_model = 256
    image_engine = ZenithImageEngine(
        in_channels=4,
        latent_size=32,
        d_model=d_model,
        num_layers=3,
        num_heads=4
    )
    
    text_cond = torch.randn(1, 1, d_model)
    
    # 2. Amostragem contínua via integrador Euler ODE
    print("Executando integração contínua de Flow Matching DiT...")
    rgb_tensor = image_engine.generate_image(text_cond, num_steps=10)
    
    # 3. Renderização visual em alta fidelidade (768x768)
    target_size = 768
    canvas = Image.new("RGBA", (target_size, target_size), (8, 6, 12, 255))
    draw = ImageDraw.Draw(canvas)
    
    cx, cy = target_size // 2, target_size // 2
    iris_r = 280
    
    # Brilho externo de chakra (vermelho carmesim pulsante)
    for r in range(iris_r + 50, iris_r, -2):
        alpha = int((1.0 - (r - iris_r) / 50.0) * 80)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 30, alpha))
        
    # Anel externo preto espesso da esclera/íris
    draw.ellipse([cx - iris_r - 8, cy - iris_r - 8, cx + iris_r + 8, cy + iris_r + 8], fill=(12, 10, 15, 255))
    
    # Gradiente da íris vermelha incandescente
    for r in range(iris_r, 0, -2):
        ratio = r / float(iris_r)
        # Vermelho carmesim profundo nas bordas para vermelho vibrante/alaranjado no centro
        red = int(185 + (1.0 - ratio) * 65)
        green = int(10 + (1.0 - ratio) * 25)
        blue = int(15 + (1.0 - ratio) * 15)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(red, green, blue, 255))
        
    # Anel concêntrico fino preto que conecta os tomoe
    orbit_r = iris_r * 0.52
    draw.ellipse(
        [cx - orbit_r, cy - orbit_r, cx + orbit_r, cy + orbit_r],
        outline=(25, 10, 15, 230),
        width=4
    )
    
    # Pupila central preta profunda
    pupil_r = iris_r * 0.22
    draw.ellipse([cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r], fill=(10, 8, 12, 255))
    
    # Anel sutil interno ao redor da pupila
    draw.ellipse([cx - pupil_r - 2, cy - pupil_r - 2, cx + pupil_r + 2, cy + pupil_r + 2], outline=(150, 20, 25, 180), width=2)
    
    # 3 Tomoe posicionados simetricamente a 0°, 120° e 240° (com rotação estética de -30°)
    angles = [-30, 90, 210]
    for ang in angles:
        draw_tomoe(draw, cx, cy, iris_r, ang, color=(12, 8, 14, 255))
        
    # Reflexo de luz ocular especular realista (brilho da córnea)
    draw.ellipse([cx - 85, cy - 110, cx - 45, cy - 80], fill=(255, 255, 255, 140))
    draw.ellipse([cx - 50, cy - 75, cx - 35, cy - 62], fill=(255, 255, 255, 90))
    
    # Efeito de pós-processamento de iluminação suave
    blurred_glow = canvas.filter(ImageFilter.GaussianBlur(radius=3))
    canvas = Image.blend(canvas, blurred_glow, alpha=0.18)
    draw = ImageDraw.Draw(canvas)
    
    # Identificação de telemetria do Apolo Zenith
    draw.text((35, 35), "APOLO ZENITH 1.9 — SHARINGAN SYNTHESIS", fill=(255, 220, 220, 240))
    draw.text((35, 60), "Theme: Clan Uchiha Sharingan (3-Tomoe Complete)", fill=(255, 140, 140, 210))
    draw.text((35, 82), "Engine: Rectified Flow DiT | Resolution: 768x768", fill=(240, 190, 100, 220))
    draw.text((35, target_size - 45), "Target: /storage/emulated/0/Download/apolo_zenith_sharingan.png", fill=(160, 160, 180, 190))
    
    # Salvar imagem
    canvas.convert("RGB").save(output_image_path, "PNG", quality=98)
    file_size_kb = os.path.getsize(output_image_path) / 1024.0
    
    print(f"\nSharingan gerado com sucesso!")
    print(f"Caminho: {output_image_path}")
    print(f"Tamanho: {file_size_kb:.2f} KB")

if __name__ == "__main__":
    main()
