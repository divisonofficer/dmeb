"""Unified dataset access for ECCV rebuttal experiments.

Two backends:
- RealDataset:  /mnt/d/data_jinnyeong/jinnyeong_cvpr2026/<session>/<timestamp>/...
- SynthDataset: /mnt/d/calra2/<scene>/{hdr_<view>, ground_truth_depth_<view>, calibration.npz}

Both honor configs/dataset_splits.yaml (single source of truth).
Pseudo-GT HDR for real comes from post/rgb_lucid_<view>.exr where present;
timestamps without a populated post/ are filtered out for HDR-quant experiments
(they are still usable for diagnostic-only runs via include_no_gt=True).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = REPO_ROOT / "configs" / "dataset_splits.yaml"


def load_splits() -> dict:
    return yaml.safe_load(SPLITS_PATH.read_text())


# ---------------------------------------------------------------------------
# Real
# ---------------------------------------------------------------------------

@dataclass
class RealSample:
    session: str
    timestamp: str
    root: Path
    has_pseudo_gt: bool

    def lucid12_path(self, view: str, sub: bool = False) -> Path:
        suffix = "_sub" if sub else ""
        return self.root / f"lucid_12_{view}{suffix}" / "image.tiff"

    def reference_hdr_exr(self, view: str = "left") -> Path | None:
        """Pseudo-GT HDR for the reference view (post/rgb_lucid_<view>.exr)."""
        p = self.root / "post" / f"rgb_lucid_{view}.exr"
        return p if p.exists() else None

    def post_rgb_tiff(self, view: str, sub: bool = False) -> Path | None:
        """Undistorted 4K LDR tiff in post/."""
        suffix = "_sub" if sub else ""
        p = self.root / "post" / f"rgb_lucid_12_{view}{suffix}_4096.tiff"
        return p if p.exists() else None

    def depth_path(self, sensor: str) -> Path | None:
        """sensor in {realsense, helios_tof, ouster}."""
        if sensor == "ouster":
            p = self.root / "post" / "points_compressed_ouster.npy"
        else:
            p = self.root / "post" / f"{sensor}_depth.npy"
        return p if p.exists() else None

    def metadata(self) -> dict:
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            return {}
        import json
        return json.loads(meta_path.read_text())


@dataclass
class RealDataset:
    split: str = "train"  # "train" | "test" | "all"
    require_pseudo_gt: bool = True
    require_devices: bool = True
    splits_cfg: dict = field(default_factory=load_splits)

    def __post_init__(self):
        cfg = self.splits_cfg["real"]
        self.root = Path(cfg["root"])
        self.train_sessions = set(cfg["train_sessions"])
        self.test_sessions = set(cfg["test_sessions"])
        self.required_devices = list(cfg["required_devices"])
        self._cache: list[RealSample] | None = None

    def _session_filter(self, name: str) -> bool:
        if self.split == "train":
            return name in self.train_sessions
        if self.split == "test":
            return name in self.test_sessions
        return name in self.train_sessions or name in self.test_sessions

    def __iter__(self) -> Iterator[RealSample]:
        if self._cache is not None:
            yield from self._cache
            return
        cache: list[RealSample] = []
        for sess in sorted(os.listdir(self.root)):
            if not self._session_filter(sess):
                continue
            sess_dir = self.root / sess
            if not sess_dir.is_dir():
                continue
            for ts in sorted(os.listdir(sess_dir)):
                ts_dir = sess_dir / ts
                if not ts_dir.is_dir() or not ts[0].isdigit():
                    continue
                children = set(os.listdir(ts_dir))
                if self.require_devices and not all(d in children for d in self.required_devices):
                    continue
                # pseudo-GT presence: post/rgb_lucid_left.exr
                has_gt = (ts_dir / "post" / "rgb_lucid_left.exr").exists()
                if self.require_pseudo_gt and not has_gt:
                    continue
                samp = RealSample(session=sess, timestamp=ts, root=ts_dir, has_pseudo_gt=has_gt)
                cache.append(samp)
                yield samp
        self._cache = cache

    def __len__(self) -> int:
        return sum(1 for _ in self)


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

@dataclass
class SynthSample:
    scene: str
    root: Path
    available_views: tuple[str, ...]
    n_frames: int

    def hdr_frame(self, view: str, idx: int, sub: bool = False) -> Path:
        suffix = "_sub" if sub else ""
        return self.root / f"hdr_{view}{suffix}" / f"{idx}.exr"

    def gt_depth_frame(self, view: str, idx: int, sub: bool = False) -> Path:
        suffix = "_sub" if sub else ""
        return self.root / f"ground_truth_depth_{view}{suffix}" / f"{idx}.exr"

    def calibration(self) -> dict:
        return dict(np.load(self.root / "calibration.npz"))


@dataclass
class SynthDataset:
    type_filter: Sequence[str] | None = None  # if None, use primary types
    splits_cfg: dict = field(default_factory=load_splits)

    def __post_init__(self):
        cfg = self.splits_cfg["synthetic"]
        self.root = Path(cfg["root"])
        self.views = tuple(cfg["views"])
        if self.type_filter is None:
            self.type_filter = tuple(cfg["primary_types"])
        else:
            self.type_filter = tuple(self.type_filter)

    def __iter__(self) -> Iterator[SynthSample]:
        for s in sorted(os.listdir(self.root)):
            if not any(s.startswith(p) for p in self.type_filter):
                continue
            sd = self.root / s
            if not sd.is_dir():
                continue
            present = [v for v in self.views if (sd / f"hdr_{v}").is_dir()]
            if not present:
                continue
            n = len(os.listdir(sd / f"hdr_{present[0]}"))
            yield SynthSample(scene=s, root=sd, available_views=tuple(present), n_frames=n)

    def __len__(self) -> int:
        return sum(1 for _ in self)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def reference_view() -> str:
    return load_splits().get("reference_view", "left")
