"""
Base wrapper interface for depth completion modules.

This module defines the abstract base class for all depth completion methods
used in the ablation study. All wrappers implement the same interface for
seamless swapping of depth modules.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class DepthModuleWrapper(nn.Module):
    """
    Abstract base class for depth completion module wrappers.

    All depth completion methods (DepthAnything+ScaleNet, RANSAC, PromptDA,
    DepthPrompting, BPNet) inherit from this class and implement the predict() method.
    """

    def __init__(self, config: dict):
        """
        Initialize depth module wrapper.

        Args:
            config: Configuration dictionary with method-specific parameters
        """
        super().__init__()
        self.config = config
        self.mode = config.get('mode', 'unknown')

    def predict(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
        p_scale: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Predict dense metric depth from RGB and sparse depth.

        Args:
            rgb: (B*N, 3, H, W) - LDR RGB images (0-1 range)
            sparse_depth: (B*N, 1, H, W) - Sparse depth normalized by p_scale
            mask: (B*N, 1, H, W) - Valid depth mask (>0.2 for valid)
            p_scale: (B*N, 1, 1, 1) - Normalization scale factor for denormalization
            intrinsics: (B*N, 3, 3) - Camera intrinsics (optional, required by some methods)

        Returns:
            metric_depth: (B*N, 1, H, W) - Dense metric depth in meters
            aux_outputs: dict - Auxiliary outputs (confidence, intermediate depths, etc.)
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def requires_intrinsics(self) -> bool:
        """
        Check if this depth module requires camera intrinsics.

        Returns:
            True if intrinsics are required, False otherwise
        """
        return False

    def freeze_parameters(self):
        """
        Freeze all parameters for inference-only mode (for ablation study).
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """
        Unfreeze all parameters for training mode.
        """
        self.train()
        for param in self.parameters():
            param.requires_grad = True


def pad_div(x: torch.Tensor, div: int = 14) -> Tuple[torch.Tensor, int, int]:
    """
    Pad tensor to be divisible by div (required for ViT-based models).

    Args:
        x: Input tensor (B, C, H, W)
        div: Divisibility requirement (default: 14 for DINOv2/DepthAnything)

    Returns:
        x_padded: Padded tensor (B, C, H_pad, W_pad)
        H_pad: Padded height
        W_pad: Padded width
    """
    B, C, H, W = x.shape
    H_pad = ((H - 1) // div + 1) * div
    W_pad = ((W - 1) // div + 1) * div

    if H_pad != H or W_pad != W:
        pad_h = H_pad - H
        pad_w = W_pad - W
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    return x, H_pad, W_pad


def downsample_area(x: torch.Tensor, s: float = 0.5, mode: str = 'bilinear') -> torch.Tensor:
    """
    Downsample tensor using area averaging or nearest neighbor.

    Args:
        x: Input tensor (B, C, H, W)
        s: Scale factor (default: 0.5 for half resolution)
        mode: Interpolation mode ('bilinear' or 'nearest')

    Returns:
        Downsampled tensor (B, C, H*s, W*s)
    """
    if s == 1.0:
        return x

    _, _, H, W = x.shape
    H_new = int(H * s)
    W_new = int(W * s)

    if mode == 'nearest':
        return torch.nn.functional.interpolate(
            x, size=(H_new, W_new), mode='nearest'
        )
    else:
        # Use align_corners=False for area-like averaging
        return torch.nn.functional.interpolate(
            x, size=(H_new, W_new), mode='bilinear', align_corners=False
        )
