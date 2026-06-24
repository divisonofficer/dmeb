#!/usr/bin/env python
"""DMEB reference-pipeline inference.

Runs the DMEB reference model over one robot session and writes per-frame HDR
predictions (linear EXR) in the reference-camera field of view. Score them with
eval.py.

Requirements
------------
- GPU (CUDA) + PyTorch.
- The reference checkpoint (see checkpoints/README.md); pass --ckpt or set
  $DMEB_CKPT.
- Third-party depth/HDR modules on PYTHONPATH (PromptDA, Depth-Anything-V2,
  AFUNet). See third_party/README.md and run third_party/setup_third_party.sh.

This mirrors the evaluation loop used for the paper's main table
(common.dmeb_factory + common.ipynb_collate + models.PromptDAMulti).

Example
-------
    python inference.py --data_root /path/to/robot_modest_dr \
                        --session 10_29_18_34 \
                        --ckpt /path/to/dmeb_ref.pth \
                        --output out/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common.dmeb_factory import make_dmeb_model, DMEB_CKPT
from common.ipynb_collate import collate_batch_input
from dataloader_real import LucidHexDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True,
                   help="subset root containing sessions, e.g. robot_modest_dr/")
    p.add_argument("--session", default="10_29_18_34", help="session dir name")
    p.add_argument("--ckpt", default=DMEB_CKPT)
    p.add_argument("--output", default="out", help="output dir for EXR predictions")
    p.add_argument("--mode", default="rs", choices=["rs", "flow"],
                   help="rs = depth+LiDAR (reference); flow = no-depth ablation")
    p.add_argument("--exp_adjust", type=float, default=50.0)
    p.add_argument("--shape", nargs=2, type=int, default=[576, 768])
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--with_gt", action="store_true",
                   help="include reference cams in inputs (training distribution)")
    return p.parse_args()


def load_dmeb(ckpt_path: str, mode: str):
    model = make_dmeb_model(mode=mode).cuda()
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt_path} (missing={len(miss)} unexpected={len(unexp)})")
    return model.eval()


def save_exr(path: str, chw: torch.Tensor):
    arr = chw.detach().float().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2
        cv2.imwrite(path, arr[..., ::-1])  # RGB -> BGR
    except Exception:
        import imageio.v3 as iio
        iio.imwrite(path, arr)


@torch.no_grad()
def run_frame(model, frame, exp_adjust: float, with_gt: bool):
    batch = collate_batch_input(frame, exp_adjust=exp_adjust, with_gt=with_gt, tri=False)
    rgbs, prompts, sats, Ks_in, _, Ts_in, Ks_tgt, Ts_tgt, ldr_min_gts, tgt_rgbs = batch
    ref_view_lin = tgt_rgbs[:, :, 3:] + torch.randn_like(tgt_rgbs[:, :, :3]) * 1e-4
    out = model(rgbs=rgbs, prompts=prompts, sats=sats,
                Ks=Ks_in, Kinvs=None, Ts=Ts_in,
                Ks_tgt=Ks_tgt, Ts_tgt=Ts_tgt,
                ldr_min=ldr_min_gts, shape_tgt=rgbs.shape[-2:],
                ref_view_lin=ref_view_lin)
    _, hdr_pred, _ = out  # (1, n_tgt, 3, H, W)
    return hdr_pred


def main():
    args = parse_args()
    session_dir = os.path.join(args.data_root, args.session)
    shape = tuple(args.shape)
    ds = LucidHexDataset(scenes=[session_dir], rescale_input=shape, rescale_output=shape)
    print(f"[infer] session={args.session} frames={len(ds)}")

    model = load_dmeb(args.ckpt, args.mode)

    indices = list(range(0, len(ds), args.frame_stride))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    views = ["lucid_left", "lucid_right"]
    for i, idx in enumerate(indices):
        try:
            frame = ds[idx]
        except Exception as e:
            print(f"  [skip] frame {idx}: {e}")
            continue
        hdr_pred = run_frame(model, frame, args.exp_adjust, args.with_gt)
        for tv, name in enumerate(views[: hdr_pred.shape[1]]):
            out_path = os.path.join(args.output, args.session, f"{idx:05d}_{name}.exr")
            save_exr(out_path, hdr_pred[0, tv])
        if i % 20 == 0:
            print(f"  [{i + 1}/{len(indices)}] frame {idx}")
    print(f"[ok] predictions in {args.output}/{args.session}")


if __name__ == "__main__":
    main()
