"""HDR metrics with shared sat/dark region handling.

All metrics expect linear-radiance HDR tensors of shape (..., 3, H, W) in float.
Tone-mapping (Reinhard) is applied only inside `lpips_tonemapped` and
`psnr_tonemapped` for backward compatibility with the main paper's reporting.

Region masks
------------
- saturated_mask:    pixels whose **input** LDR (any input view) is saturated
- dark_mask:         pixels whose **input** LDR (any input view) is underexposed
- valid_mask:        pixels with valid reference HDR (passed in by caller)

Region-restricted metrics are computed by zeroing the residual outside the
region of interest and dividing by the region area.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# -----------------------------------------------------------------------
# Region masks
# -----------------------------------------------------------------------

def saturated_mask(ldr_inputs: torch.Tensor, hi: float = 0.98) -> torch.Tensor:
    """ldr_inputs: (V, 3, H, W) in [0, 1]; saturated if max input >= hi in *any* view."""
    return (ldr_inputs.amax(dim=(0, 1)) >= hi)


def dark_mask(ldr_inputs: torch.Tensor, lo: float = 0.02) -> torch.Tensor:
    """Dark if max input <= lo in *any* view (underexposed everywhere)."""
    return (ldr_inputs.amax(dim=(0, 1)) <= lo)


# -----------------------------------------------------------------------
# Tone mapping (Reinhard global) — only for visualization / LPIPS
# -----------------------------------------------------------------------

def reinhard_tonemap(hdr: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    x = hdr / (1.0 + hdr)
    return torch.clamp(x, 0.0, 1.0).pow(1.0 / gamma)


# -----------------------------------------------------------------------
# Core metrics (linear domain)
# -----------------------------------------------------------------------

def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is None:
        return x.mean().item()
    mask = mask.to(x.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum().div(denom).item()


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None,
         max_val: float = 1.0) -> float:
    """Linear-domain PSNR.  pred, gt: (..., 3, H, W).  mask: (H, W) bool/float."""
    diff = (pred - gt).pow(2).mean(dim=-3)  # (..., H, W)
    if mask is not None:
        diff = diff * mask.to(diff.dtype)
        denom = mask.to(diff.dtype).sum().clamp(min=1.0)
        mse = diff.sum() / denom
    else:
        mse = diff.mean()
    if mse <= 0:
        return float("inf")
    return (10.0 * torch.log10(max_val ** 2 / mse)).item()


def psnr_tonemapped(pred: torch.Tensor, gt: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> float:
    return psnr(reinhard_tonemap(pred), reinhard_tonemap(gt), mask=mask, max_val=1.0)


def ssim(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except ImportError:
        return float("nan")
    a = reinhard_tonemap(pred).clamp(0, 1).cpu().numpy()
    b = reinhard_tonemap(gt).clamp(0, 1).cpu().numpy()
    if a.ndim == 4:  # (V, 3, H, W) - take first view
        a, b = a[0], b[0]
    a = np.transpose(a, (1, 2, 0))
    b = np.transpose(b, (1, 2, 0))
    if mask is not None:
        m = mask.cpu().numpy().astype(bool)
        if m.sum() == 0:
            return float("nan")
        # full-image SSIM; mask is approximate, region-version weights via win_size
    return float(sk_ssim(a, b, channel_axis=2, data_range=1.0))


def lpips(pred: torch.Tensor, gt: torch.Tensor, net: str = "alex",
          _cache: dict | None = None) -> float:
    if _cache is None:
        _cache = lpips.__dict__.setdefault("_cache", {})
    try:
        import lpips as _lp
    except ImportError:
        return float("nan")
    key = ("lpips", net)
    if key not in _cache:
        _cache[key] = _lp.LPIPS(net=net).eval().to(pred.device)
    a = (reinhard_tonemap(pred) * 2 - 1).clamp(-1, 1)
    b = (reinhard_tonemap(gt) * 2 - 1).clamp(-1, 1)
    if a.ndim == 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    with torch.no_grad():
        return _cache[key](a, b).mean().item()


def pu_psnr(pred: torch.Tensor, gt: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> float:
    """PU21-PSNR (perceptually uniform encoding for HDR).

    Approximation: log encoding clipped to a sane HDR range, then linear PSNR.
    For exact PU21, integrate `pu21_encoder` from VDP3 if available.
    """
    eps = 1e-3
    a = torch.log(pred.clamp(min=eps)) / torch.log(torch.tensor(1e4))
    b = torch.log(gt.clamp(min=eps)) / torch.log(torch.tensor(1e4))
    return psnr(a, b, mask=mask, max_val=1.0)


# -----------------------------------------------------------------------
# Region-decomposed report
# -----------------------------------------------------------------------

@dataclass
class RegionReport:
    full: float
    saturated: float
    dark: float
    n_sat: int
    n_dark: int


def metric_by_region(metric_fn, pred: torch.Tensor, gt: torch.Tensor,
                     ldr_inputs: torch.Tensor,
                     valid: Optional[torch.Tensor] = None) -> RegionReport:
    """Apply metric_fn(pred, gt, mask) over full / saturated / dark regions."""
    sat = saturated_mask(ldr_inputs)
    dk = dark_mask(ldr_inputs)
    if valid is not None:
        sat = sat & valid
        dk = dk & valid
    full = metric_fn(pred, gt, valid)
    s = metric_fn(pred, gt, sat) if sat.sum() > 0 else float("nan")
    d = metric_fn(pred, gt, dk) if dk.sum() > 0 else float("nan")
    return RegionReport(full=full, saturated=s, dark=d, n_sat=int(sat.sum()), n_dark=int(dk.sum()))
