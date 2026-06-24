"""Single source of truth for DMEB model construction + ckpt path.

The training-time ctor signature changed on 2026-05-09 (compositor refiner +
explicit depth-refiner / hdr-depth coupling kwargs).  All inference scripts
must use these exact kwargs to match the training architecture, otherwise
checkpoint state-dicts have mismatched keys.

Usage
-----
    from common.dmeb_factory import make_dmeb_model, DMEB_CKPT
    model = make_dmeb_model(mode="rs").cuda().eval()
    sd = torch.load(DMEB_CKPT, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    miss, _ = model.load_state_dict(sd, strict=False)

Real ckpts (e.g., ``dmeb_real_compositor_epoch466.pth``) are partial
(~1.5 MB, trainable parts only) and CANNOT be loaded directly.  Always
use the *merged* ckpt produced by
``method/depth_warp_hdr/scripts/merge_checkpoint_state.py``.
"""
from __future__ import annotations

import os

from models.PromptDAMulti import PromptDA_MV_All


# Reference checkpoint. Download it from the project page and either set the
# DMEB_CKPT environment variable or pass --ckpt explicitly to the entrypoints.
# Default points at the bundled location under code/checkpoints/.
# See code/checkpoints/README.md for the download link and md5.
DMEB_CKPT = os.environ.get(
    "DMEB_CKPT",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "dmeb_ref.pth"),
)


# Training-time kwargs for PromptDA_MV_All (legacy topk refiner +
# new depth_refiner / hdr-depth coupling kwargs).
# - hdr_refiner_type="topk"  -> HDRRefinerLite (v3 da_adapter/refine_core)
# - refiner_n_tf=4 / refiner_channels=72 -> "rs_t4" base
DMEB_KWARGS: dict = dict(
    hdr_refiner_type="topk",
    refiner_n_tf=4,
    refiner_channels=72,
    hdr_refiner_downsample=1,
    depth_refiner_base=16,
    depth_refiner_downsample=2,
    depth_refiner_max_delta_inv=0.10,
    depth_merge_den_eps=1e-4,
    hdr_depth_coupling_floor=0.5,
    hdr_depth_coupling_strength=0.3,
    hdr_depth_coupling_max=1.0,
)


def make_dmeb_model(mode: str = "rs", **overrides):
    """Construct PromptDA_MV_All with the canonical training kwargs.

    Parameters
    ----------
    mode : "rs" (depth+lidar) | "flow" (no-depth ablation) | ...
    overrides : optional kwargs that take precedence over DMEB_KWARGS.
    """
    kwargs = dict(DMEB_KWARGS)
    kwargs.update(overrides)
    return PromptDA_MV_All(mode=mode, **kwargs)
