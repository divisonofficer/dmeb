"""Tonemap operators for figure visualization.

Inputs are linear-radiance HDR arrays (H, W, 3) in float, value range
unbounded but typically [0, ~5+] after our model output. All operators return
clipped uint-friendly LDR in [0, 1].

Names follow the rebuttal's \\rC-m1 discussion: visualization-only, no metric
implications.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import cv2


# ----------------------------------------------------------------------------
# 1. mu-law (current default in ipynb_collate.tonemap_up, mu=1e6)
# ----------------------------------------------------------------------------

def mu_law(hdr: np.ndarray, mu: float = 5e3) -> np.ndarray:
    """Lower mu (~5e3) gives better contrast than the default 1e6."""
    x = np.clip(hdr, 0.0, None)
    return (np.log1p(mu * x) / np.log1p(mu)).clip(0, 1)


# ----------------------------------------------------------------------------
# 2. Reinhard global (closed-form L = x/(1+x), then gamma)
# ----------------------------------------------------------------------------

def reinhard_global(hdr: np.ndarray, gamma: float = 2.2,
                     key: float = 0.18) -> np.ndarray:
    x = np.clip(hdr, 0.0, None)
    # Per-pixel attenuation; key scales the average luminance to a target value.
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2] + 1e-8
    log_avg = np.exp(np.mean(np.log(lum)))
    scaled = x * (key / log_avg)
    mapped = scaled / (1.0 + scaled)
    return (np.clip(mapped, 0, 1) ** (1.0 / gamma))


# ----------------------------------------------------------------------------
# 3. ACES (Krzysztof Narkowicz approximation, widely used in games/film)
# ----------------------------------------------------------------------------

def aces(hdr: np.ndarray, exposure: float = 1.6,
          gamma: float = 2.2) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.clip(hdr, 0.0, None) * exposure
    y = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(y, 0, 1) ** (1.0 / gamma)


# ----------------------------------------------------------------------------
# 4. Hable Uncharted-2 (cinematic feel)
# ----------------------------------------------------------------------------

def _hable_curve(x: np.ndarray) -> np.ndarray:
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def hable(hdr: np.ndarray, exposure: float = 2.0, white: float = 11.2,
          gamma: float = 2.2) -> np.ndarray:
    x = np.clip(hdr, 0.0, None) * exposure
    mapped = _hable_curve(x) / _hable_curve(np.asarray(white, dtype=x.dtype))
    return np.clip(mapped, 0, 1) ** (1.0 / gamma)


# ----------------------------------------------------------------------------
# 5. OpenCV built-ins (Reinhard / Drago / Mantiuk)
# ----------------------------------------------------------------------------

def cv_reinhard(hdr: np.ndarray, gamma: float = 1.5,
                 light_adapt: float = 0.0,
                 color_adapt: float = 0.0) -> np.ndarray:
    op = cv2.createTonemapReinhard(gamma=gamma, intensity=0.0,
                                   light_adapt=light_adapt,
                                   color_adapt=color_adapt)
    out = op.process(hdr.astype(np.float32))
    return np.clip(out, 0, 1)


def cv_drago(hdr: np.ndarray, gamma: float = 1.0,
              saturation: float = 1.0, bias: float = 0.85) -> np.ndarray:
    op = cv2.createTonemapDrago(gamma=gamma, saturation=saturation, bias=bias)
    out = op.process(hdr.astype(np.float32))
    return np.clip(out, 0, 1)


def cv_mantiuk(hdr: np.ndarray, gamma: float = 2.2,
                scale: float = 0.85, saturation: float = 1.2) -> np.ndarray:
    op = cv2.createTonemapMantiuk(gamma=gamma, scale=scale, saturation=saturation)
    out = op.process(hdr.astype(np.float32))
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mu_1e6":   lambda x: mu_law(x, mu=1e6),    # current default
    "mu_5e3":   lambda x: mu_law(x, mu=5e3),
    "reinhard_global": reinhard_global,
    "aces":     aces,
    "hable":    hable,
    "cv_reinhard": cv_reinhard,
    "cv_drago": cv_drago,
    "cv_mantiuk": cv_mantiuk,
}


def apply(name: str, hdr: np.ndarray) -> np.ndarray:
    if name not in REGISTRY:
        raise ValueError(f"unknown tonemap {name}; pick from {list(REGISTRY)}")
    return REGISTRY[name](hdr).clip(0, 1)
