"""Variant-aware collate for SV/MV ablation (P0-1, P1-1).

Forks ipynb_collate.collate_batch_input by parameterizing the choice of
`input_sides`. Everything else (HDR -> LDR simulation, sat masks, prompts,
camera params, target views) stays identical to ensure fair comparison.
"""
from __future__ import annotations

from typing import Sequence

import torch

from common.ipynb_collate import hdr2ldr


# Available 12-cam sides
ALL_12CAM = ["12_rear", "12_left", "12_right",
             "12_rear_sub", "12_left_sub", "12_right_sub"]


def _exp(frame, side: str) -> float:
    e = frame[f"exposure_{side}"]
    return float(e.item() if hasattr(e, "item") else e)


def make_input_sides(frame, variant: str, with_gt: bool = True,
                     ref_view: str = "12_rear") -> list[str]:
    """Compute input_sides list for a given variant.

    `with_gt=True` (ipynb default): lucid_left / lucid_right are also fed as
    input — model sees GT-side views. Useful when matching the training
    distribution but contaminates the SV/MV ablation since every variant gains
    a free GT-aware input.
    `with_gt=False`: only 12-cam views are inputs; lucid_left/right are used
    purely as evaluation targets. This is the cleaner ablation.
    """
    if variant in ("full", "mv_me"):
        # Replicate ipynb behaviour: min+max 12-cam (each picked once),
        # then *= 2, then with_gt insertion of "left", then +="left","right".
        exps = [(s, _exp(frame, s)) for s in ALL_12CAM]
        exps_sorted = sorted(exps, key=lambda x: x[1])
        min_exp = exps_sorted[0][1]
        max_exp = exps_sorted[-1][1]
        cams = [s for s, e in exps if e in (min_exp, max_exp)]
        cams = cams * 2  # ipynb does input_sides *= 2
        if with_gt:
            cams = cams[:3] + ["left"] + cams[3:]
            cams += ["left", "right"]
        return cams

    if variant == "mv_se":
        # median exposure of each {rear, left, right} (no _sub)
        cams = ["12_rear", "12_left", "12_right"]
        if with_gt:
            cams += ["left", "left", "right"]
        return cams

    if variant == "sv_me":
        # ref camera + its _sub
        cams = [ref_view, f"{ref_view}_sub"]
        if with_gt:
            cams += ["left", "left", "right"]
        return cams

    if variant == "ref_ldr":
        # Duplicate the reference view to avoid N=1 forward crashes.
        # Functionally equivalent to a single view because both copies are identical.
        cams = [ref_view, ref_view]
        if with_gt:
            cams += ["left", "left", "right"]
        return cams

    raise ValueError(f"unknown variant: {variant}")


def collate_batch_variant(frame, variant: str,
                          exp_adjust: float = 50.0,
                          hdr_adjust: float = 4920.762683082901,
                          ref_view: str = "12_rear",
                          with_gt: bool = True):
    """Build DMEB inputs for a given SV/MV variant.

    `with_gt=True` matches the ipynb training/inference distribution
    (lucid_left/right also fed as input). `with_gt=False` keeps lucid
    purely as evaluation target — cleaner ablation.

    Returns a tuple matching ipynb collate_batch_input:
        (rgbs, prompts, sats, Ks_in, Kinvs_in, Ts_in,
         Ks_tgt, Ts_tgt, ldr_min_gts, tgt_rgbs)
    """
    input_sides = make_input_sides(frame, variant, with_gt=with_gt, ref_view=ref_view)

    rgbs = []
    prompts = []
    sats = []
    Ks_in = []
    Ts_in = []
    ldr_min_gts = []
    tgt_rgbs = []
    Ks_tgt = []
    Ts_tgt = []

    for s_i, side in enumerate(input_sides):
        rgb = frame[f"rgb_{side}"].unsqueeze(0).cuda() * 1.0
        if "12" not in side:
            with torch.no_grad():
                exp_hdr = frame[f"exposure_{side}"] / exp_adjust
                hdr_input = frame[f"rgb_{side}"].float()
                hdr_raw = hdr_input.clone() * hdr_adjust
            ldr_, exp_hdr = hdr2ldr(
                hdr_raw, exp_hdr,
                target_intensity=[0.001, 0.005] if s_i % 2 == 0 else [0.3, 0.45],
                noise_std=0.001,
            )
            noisy = torch.concat([ldr_, ldr_ / exp_hdr], dim=0).unsqueeze(0)
            rgb = noisy.clamp(0, 1).cuda()
            sat = (
                (ldr_ > 0.005)
                & (ldr_ < 0.99)
                & (hdr_input < 0.99)
            ).float().cuda()
            sat = torch.max(sat, dim=0, keepdim=True)[0].unsqueeze(0)
        else:
            exp = frame[f"exposure_{side}"] / exp_adjust
            sat = ((rgb > 0.03) & (rgb < 0.99)).float().cuda()
            rgb = torch.concat([rgb, rgb / exp], dim=1)
            sat = torch.max(sat, dim=1, keepdim=True)[0]

        ldr_min_gt = frame[f"min_dn_{side}"] * 0.1
        if ldr_min_gt > 0.05:
            ldr_min_gt = 0.05 + (ldr_min_gt - 0.05) ** 2
        if "12" not in side:
            ldr_min_gt = 0.001
        ldr_min_gts.append(torch.tensor([ldr_min_gt]).unsqueeze(0).cuda())

        rgbs.append(rgb)
        prompts.append(frame[f"lidar_{side}"].unsqueeze(0).cuda())
        Ks_in.append(frame[f"K_{side}"].unsqueeze(0).cuda())
        Ts_in.append(frame[f"E_{side}"].unsqueeze(0).cuda())
        sats.append(sat)

    # Target views (always lucid left+right for fair PSNR/SSIM)
    for side in ["left", "right"]:
        Ks_tgt.append(frame[f"K_{side}"].unsqueeze(0).cuda())
        Ts_tgt.append(frame[f"E_{side}"].unsqueeze(0).cuda())
        tgt_rgb = frame[f"rgb_{side}"].unsqueeze(0).cuda() * hdr_adjust
        tgt_exp = frame[f"exposure_{side}"] / exp_adjust
        tgt_rgb = torch.concat([tgt_rgb, tgt_rgb / tgt_exp], dim=1)
        tgt_rgbs.append(tgt_rgb)

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
            Ks_tgt, Ts_tgt, ldr_min_gts, tgt_rgbs)
