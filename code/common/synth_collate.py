"""Variant-aware collate for the synthetic (calra2) dataset (P0-4).

Synth scene layout (per /mnt/d/calra2/<scene>/):
    hdr_<view>/<idx>.exr          # linear HDR radiance  (raw range ~0..0.02)
    hdr_<view>_sub/<idx>.exr       # same scene, separate copy used for sub-exp sim
    ground_truth_depth_<view>/<idx>.png  (uint16 millimeters)
    calibration.npz: K_<view>, Transform_<view>_rear

Available views: rear, rear_sub, left, left_sub, right, right_sub
(plus mid, mid_sub for example_hex_type1_extended_*).

Because synth HDR is dim by default (max ~0.01), we apply a per-frame
auto-normalization that scales each HDR to a consistent training-distribution
range before hdr2ldr's target_intensity loop kicks in. The normalizer can be
overridden via the `hdr_scale_override` kwarg, which is what the magic-number
grid search uses.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import imageio.v3 as iio
import cv2

from common.ipynb_collate import hdr2ldr


# View names available in calra2 (mid/_sub only in extended scenes)
ALL_VIEWS = ["rear", "left", "right", "mid"]


# ---------------------------------------------------------------------------
# Single-scene loader
# ---------------------------------------------------------------------------

def _load_exr(path: Path, shape=(576, 768)) -> torch.Tensor:
    """Load an EXR file and return (3, H, W) float CUDA tensor in BGR-converted RGB."""
    img = iio.imread(str(path))
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] >= 3:
        # imageio gives RGB; calra2 was written via OpenCV (BGR)
        img = img[..., ::-1]
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.copy()).permute(2, 0, 1).float()


def _load_depth_png(path: Path, shape=(576, 768)) -> torch.Tensor:
    raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
    raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(raw)


def list_synth_scenes(root: str | Path,
                      type_filter: Optional[list[str]] = None) -> list[Path]:
    """Return paths of synth scenes matching given type prefixes."""
    root = Path(root)
    if type_filter is None:
        type_filter = ["example_hex_type1_extended_", "example_hex_type1_exr_"]
    scenes = []
    for s in sorted(os.listdir(root)):
        if any(s.startswith(p) for p in type_filter):
            sd = root / s
            if (sd / "calibration.npz").exists() and (sd / "hdr_left").is_dir():
                scenes.append(sd)
    return scenes


def list_frames(scene_dir: Path) -> list[int]:
    files = [f for f in os.listdir(scene_dir / "hdr_left") if f.endswith(".exr")]
    files.sort(key=lambda x: int(x.split(".")[0]))
    return [int(f.split(".")[0]) for f in files]


def available_views(scene_dir: Path) -> list[str]:
    return [v for v in ALL_VIEWS if (scene_dir / f"hdr_{v}").is_dir()]


# ---------------------------------------------------------------------------
# Variant input set definition
# ---------------------------------------------------------------------------

def make_synth_input_sides(variant: str, ref_view: str, available: list[str]) -> list[str]:
    """input view names (without _sub suffix) for each variant.

    For SV/MV ablation. with_gt is irrelevant on synth (no separate GT camera);
    GT is the *target* view's HDR directly.
    """
    pool = [v for v in available if v in ALL_VIEWS]
    if variant in ("full", "mv_me"):
        # all available views, both main and sub
        return [v + suf for v in pool for suf in ("", "_sub")]
    if variant == "mv_se":
        # all views, main exposure only
        return [v for v in pool]
    if variant == "sv_me":
        return [ref_view, ref_view + "_sub"]
    if variant == "ref_ldr":
        # duplicated for N>=2
        return [ref_view, ref_view]
    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Frame -> DMEB inputs
# ---------------------------------------------------------------------------

def collate_synth_variant(scene_dir: Path, frame_idx: int, variant: str,
                          ref_view: str = "rear",
                          target_view: str = "left",
                          shape=(576, 768),
                          hdr_scale: float = 100.0,
                          sparse_depth_ratio: float = 0.01):
    """Build DMEB inputs for one synth frame.

    Returns:
        rgbs, prompts, sats, Ks_in, Kinvs_in, Ts_in,
        Ks_tgt, Ts_tgt, ldr_min_gts, tgt_rgbs, gt_hdr_target
    All tensors are CUDA, batched (B=1).
    `gt_hdr_target` is the linear HDR of the target view (for PSNR).
    """
    available = available_views(scene_dir)
    if target_view not in available:
        # fallback to first available view
        target_view = available[0]
    if ref_view not in available:
        ref_view = available[0]

    input_sides_logical = make_synth_input_sides(variant, ref_view, available)
    calibration = dict(np.load(scene_dir / "calibration.npz", allow_pickle=True))

    rgbs, prompts, sats, Ks_in, Ts_in, ldr_min_gts = [], [], [], [], [], []

    for s_i, side in enumerate(input_sides_logical):
        # split _sub suffix
        is_sub = side.endswith("_sub")
        view = side[:-4] if is_sub else side
        suf = "_sub" if is_sub else ""

        # GT HDR
        hdr_path = scene_dir / f"hdr_{view}{suf}" / f"{frame_idx}.exr"
        hdr = _load_exr(hdr_path, shape=shape)  # (3, H, W) on CPU
        hdr = hdr.cuda() * hdr_scale  # bring into target range

        # Random exposure for this view (deterministic by index for reproducibility)
        target_intensity = [0.001, 0.005] if s_i % 2 == 0 else [0.3, 0.45]
        exp0 = torch.tensor(1.0, device="cuda")
        ldr, exp_out = hdr2ldr(hdr, exp0, target_intensity=target_intensity,
                                noise_std=0.001)

        rgb6 = torch.cat([ldr, ldr / exp_out.clamp(min=1e-6)], dim=0).unsqueeze(0)
        sat = ((ldr > 0.005) & (ldr < 0.99) & (hdr < 0.99)).float()
        sat = sat.max(dim=0, keepdim=True)[0].unsqueeze(0)

        # Sparse depth prompt (using GT depth + random mask)
        depth_path = scene_dir / f"ground_truth_depth_{view}{suf}" / f"{frame_idx}.png"
        if depth_path.exists():
            depth = _load_depth_png(depth_path, shape=shape).cuda()
            mask = torch.rand_like(depth) < sparse_depth_ratio
            sparse = torch.where(mask, depth, torch.zeros_like(depth))
        else:
            sparse = torch.zeros(shape).cuda()
        prompt = sparse.unsqueeze(0).unsqueeze(0)

        K = torch.from_numpy(calibration[f"K_{view}{suf}"]).float().cuda().unsqueeze(0)
        T_key = f"Transform_{view}{suf}_rear"
        if T_key in calibration:
            T = torch.from_numpy(calibration[T_key]).float().cuda().unsqueeze(0)
        else:
            T = torch.eye(4).cuda().unsqueeze(0)

        ldr_min_gts.append(torch.tensor([[0.001]]).cuda())
        rgbs.append(rgb6)
        prompts.append(prompt)
        sats.append(sat)
        Ks_in.append(K)
        Ts_in.append(T)

    # Target view (always the requested target, plus its sub for redundancy if asked)
    Ks_tgt, Ts_tgt = [], []
    tgt_rgbs = []
    gt_hdr_target_tensor = None
    for tv in [target_view]:
        K = torch.from_numpy(calibration[f"K_{tv}"]).float().cuda().unsqueeze(0)
        T_key = f"Transform_{tv}_rear"
        if T_key in calibration:
            T = torch.from_numpy(calibration[T_key]).float().cuda().unsqueeze(0)
        else:
            T = torch.eye(4).cuda().unsqueeze(0)
        Ks_tgt.append(K); Ts_tgt.append(T)

        hdr_t = _load_exr(scene_dir / f"hdr_{tv}" / f"{frame_idx}.exr", shape=shape)
        hdr_t = hdr_t.cuda() * hdr_scale
        gt_hdr_target_tensor = hdr_t.clone()
        tgt6 = torch.cat([hdr_t, hdr_t.clone()], dim=0).unsqueeze(0)
        tgt_rgbs.append(tgt6)

    rgbs = torch.cat(rgbs, dim=0).unsqueeze(0)
    prompts = torch.cat(prompts, dim=0).unsqueeze(0) / 1000.0
    sats = torch.cat(sats, dim=0).unsqueeze(0)
    Ks_in = torch.cat(Ks_in, dim=0).unsqueeze(0)
    Ts_in = torch.cat(Ts_in, dim=0).unsqueeze(0)
    Ks_tgt = torch.cat(Ks_tgt, dim=0).unsqueeze(0)
    Ts_tgt = torch.cat(Ts_tgt, dim=0).unsqueeze(0)
    ldr_min_gts = torch.cat(ldr_min_gts, dim=0).unsqueeze(0)
    tgt_rgbs = torch.cat(tgt_rgbs, dim=0).unsqueeze(0)

    return (rgbs, prompts, sats, Ks_in, Ks_in, Ts_in,
            Ks_tgt, Ts_tgt, ldr_min_gts, tgt_rgbs, gt_hdr_target_tensor)
