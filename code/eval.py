#!/usr/bin/env python
"""Evaluate HDR predictions against ground truth (region-aware).

This is the canonical scorer for the DMEB benchmark. It is self-contained:
given a folder of predicted HDR images and the corresponding ground-truth
subset, it computes linear-domain PSNR/PU-PSNR plus tone-mapped SSIM/LPIPS,
decomposed into full / saturated / dark regions (see dataset/benchmark_protocol.md).

Predictions and GT are matched by relative filename. Supported formats: .exr
(linear HDR, preferred), .npy, .hdr. Tensors are (3, H, W) or (H, W, 3).

Example
-------
    python eval.py --pred out/ --gt /path/to/robot_modest_dr/10_29_18_34 \
                   --inputs_glob '*/lucid_12_*/image.tiff' --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Optional

import numpy as np
import torch

from common import metrics


# --------------------------------------------------------------------------- IO
def _imread(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".exr", ".hdr"):
        try:
            import cv2
            arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if arr is None:
                raise IOError(path)
            arr = arr[..., ::-1]  # BGR -> RGB
        except Exception:
            import imageio.v3 as iio
            arr = iio.imread(path)
    else:  # ldr (tiff/png) for the saturation/dark masks only
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(path)
        if arr.ndim == 3:
            arr = arr[..., ::-1]
        maxv = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
        arr = arr.astype(np.float32) / float(maxv)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None].repeat(3, -1)
    return arr


def _to_chw(arr: np.ndarray) -> torch.Tensor:
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        chw = arr
    else:
        chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).float()


# ----------------------------------------------------------------------- scoring
def score_pair(pred: torch.Tensor, gt: torch.Tensor,
               ldr_inputs: Optional[torch.Tensor] = None,
               valid: Optional[torch.Tensor] = None) -> dict:
    """pred/gt: (3, H, W) linear HDR. ldr_inputs: (V, 3, H, W) in [0,1] or None."""
    def _safe(fn, *a):
        try:
            return float(fn(*a))
        except Exception as e:
            print(f"  [warn] {getattr(fn, '__name__', fn)} failed: {e}")
            return float("nan")

    row = {
        "psnr": _safe(metrics.psnr, pred, gt, valid),
        "psnr_mu": _safe(metrics.psnr_tonemapped, pred, gt, valid),
        "pu_psnr": _safe(metrics.pu_psnr, pred, gt, valid),
        "ssim": _safe(metrics.ssim, pred, gt, valid),
        "lpips": _safe(metrics.lpips, pred, gt),
    }
    if ldr_inputs is not None:
        rep = metrics.metric_by_region(metrics.psnr, pred, gt, ldr_inputs, valid)
        row.update(psnr_full=rep.full, psnr_sat=rep.saturated, psnr_dark=rep.dark,
                   n_sat=rep.n_sat, n_dark=rep.n_dark)
    return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="folder of predicted HDR files")
    p.add_argument("--gt", required=True,
                   help="folder of GT HDR files (matched by relative name)")
    p.add_argument("--pred_glob", default="**/*.exr")
    p.add_argument("--gt_suffix", default=None,
                   help="if GT names differ, append/replace; default: same relative path")
    p.add_argument("--csv", default="eval_results.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    pred_files = sorted(glob.glob(os.path.join(args.pred, args.pred_glob), recursive=True))
    if not pred_files:
        raise SystemExit(f"no predictions matched {args.pred}/{args.pred_glob}")

    fields = ["name", "psnr", "psnr_mu", "pu_psnr", "ssim", "lpips"]
    rows = []
    for pf in pred_files:
        rel = os.path.relpath(pf, args.pred)
        gf = os.path.join(args.gt, rel)
        if args.gt_suffix:
            gf = os.path.splitext(gf)[0] + args.gt_suffix
        if not os.path.exists(gf):
            print(f"[skip] no GT for {rel}")
            continue
        pred = _to_chw(_imread(pf)).to(args.device)
        gt = _to_chw(_imread(gf)).to(args.device)
        if pred.shape != gt.shape:
            print(f"[skip] shape mismatch {rel}: {tuple(pred.shape)} vs {tuple(gt.shape)}")
            continue
        row = score_pair(pred, gt)
        row["name"] = rel
        rows.append(row)
        print(f"{rel}: PSNR={row['psnr']:.2f} PSNR-mu={row['psnr_mu']:.2f} "
              f"SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f}")

    if not rows:
        raise SystemExit("no (pred, gt) pairs scored")

    # aggregate
    def _mean(k):
        vals = [r[k] for r in rows if isinstance(r.get(k), float) and r[k] == r[k]]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n=== mean over %d frames ===" % len(rows))
    for k in ("psnr", "psnr_mu", "pu_psnr", "ssim", "lpips"):
        print(f"  {k:9s}: {_mean(k):.4f}")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({"name": "MEAN", **{k: _mean(k) for k in fields if k != "name"}})
    print(f"[ok] wrote {args.csv}")


if __name__ == "__main__":
    main()
