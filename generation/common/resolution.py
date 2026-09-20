"""
Apolo Zenith 1.9 — Resolução, Qualidade e Metadados.

Distingue explicitamente:
  - native generation resolution (resolução nativa do modelo latente)
  - upscaled resolution (saída da super-resolução generativa)
  - final delivery resolution (arquivo entregue)

Nunca confundir upscale com geração nativa — os metadados carregam os três valores.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
}


@dataclass
class QualityPreset:
    name: str
    final_max_side: int      # maior lado da resolução final
    base_max_side: int       # resolução nativa alvo antes da SR
    sr_stages: int           # número de estágios de super-resolução generativa
    num_steps: int           # passos de amostragem ODE


QUALITY_PRESETS: Dict[str, QualityPreset] = {
    "preview":  QualityPreset("preview",  final_max_side=512,  base_max_side=128,  sr_stages=1, num_steps=8),
    "standard": QualityPreset("standard", final_max_side=1024, base_max_side=192,  sr_stages=1, num_steps=16),
    "high":     QualityPreset("high",     final_max_side=1920, base_max_side=256,  sr_stages=2, num_steps=24),
    "ultra":    QualityPreset("ultra",    final_max_side=3840, base_max_side=384,  sr_stages=2, num_steps=32),
    "8k":       QualityPreset("8k",       final_max_side=7680, base_max_side=480,  sr_stages=3, num_steps=40),
}


@dataclass
class ResolutionMetadata:
    native_width: int
    native_height: int
    upscaled_width: int
    upscaled_height: int
    final_width: int
    final_height: int
    quality: str
    aspect_ratio: str
    sr_stages: int
    note: str = (
        "native_* = resolução da geração latente; upscaled_* = saída da "
        "super-resolução generativa; final_* = resolução entregue."
    )

    def to_dict(self) -> Dict:
        return asdict(self)


def _snap(v: int, mult: int = 8) -> int:
    return max(mult, (v // mult) * mult)


def fit_to_aspect(max_side: int, aspect: str) -> Tuple[int, int]:
    """Calcula (width, height) em que o maior lado == max_side respeitando o aspecto."""
    aw, ah = ASPECT_RATIOS[aspect]
    if aw >= ah:
        w, h = max_side, round(max_side * ah / aw)
    else:
        h, w = max_side, round(max_side * aw / ah)
    return _snap(w), _snap(h)


def plan_resolution(
    quality: str = "standard",
    aspect_ratio: str = "16:9",
    max_native_side: int | None = None,
) -> Dict:
    """
    Planeja resolução nativa/final de forma adaptativa.

    max_native_side limita a resolução nativa conforme memória disponível
    (ex.: dispositivos móveis), mantendo a resolução final entregue via SR.
    """
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Qualidade desconhecida: {quality}. Use {list(QUALITY_PRESETS)}")
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Aspect ratio desconhecido: {aspect_ratio}. Use {list(ASPECT_RATIOS)}")

    preset = QUALITY_PRESETS[quality]
    base_side = preset.base_max_side
    if max_native_side is not None:
        base_side = min(base_side, max_native_side)

    nw, nh = fit_to_aspect(base_side, aspect_ratio)
    fw, fh = fit_to_aspect(preset.final_max_side, aspect_ratio)
    meta = ResolutionMetadata(
        native_width=nw,
        native_height=nh,
        upscaled_width=fw,   # preenchido efetivamente após a SR; mantido igual ao final quando concluída
        upscaled_height=fh,
        final_width=fw,
        final_height=fh,
        quality=quality,
        aspect_ratio=aspect_ratio,
        sr_stages=preset.sr_stages,
    )
    return {"preset": preset, "metadata": meta}
