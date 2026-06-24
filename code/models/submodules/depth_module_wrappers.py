"""
Depth module wrapper implementations for ablation study.

This module contains wrapper classes for all depth completion methods:
- DAScaleNetWrapper: DepthAnythingV2 + learnable MLP ScaleNet (mode="rs")
- DARANSACWrapper: DepthAnythingV2 + geometric RANSAC (mode="ransac")
- PromptDAWrapper: PromptDA depth completion (mode="promptda")
- DepthPromptingWrapper: DepthPrompting with CSPN (mode="depthprompt")
- BPNetWrapper: BPNet bilateral propagation (mode="bpnet")
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import sys

sys.path.append("/jarvis")

from .depth_module_base import DepthModuleWrapper, pad_div
from .da_scaler import ScaleNet
from .depth_ransac import RANSACScaleAligner
from modules.monodepth.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from modules.depth_densify.PromptDA.promptda.promptda import PromptDA


class DAScaleNetWrapper(DepthModuleWrapper):
    """
    Wrapper for DepthAnythingV2 + learnable ScaleNet (current baseline, mode="rs").

    This wraps the existing implementation that uses:
    - Frozen DepthAnythingV2 backbone for monocular relative disparity
    - Learnable ScaleNet (MLP-based) to convert to metric depth
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.debug_logged = False  # Only log first forward pass

        print(f"[DAScaleNetWrapper] Initializing DepthAnything + ScaleNet...")

        # Load frozen DepthAnythingV2 backbone
        self.backbone = DepthAnythingV2(
            encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
        )
        da_weight_path = "modules/monodepth/DepthAnythingV2/pretrained/depth_anything_v2_vitb.pth"
        print(f"[DAScaleNetWrapper] Loading DepthAnythingV2 from {da_weight_path}")
        state = torch.load(da_weight_path, map_location='cpu')
        self.backbone.load_state_dict(state)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        print(f"[DAScaleNetWrapper] DepthAnythingV2 loaded successfully (frozen)")

        # Learnable ScaleNet (frozen for ablation, but can be trained)
        self.da_post = ScaleNet(allow_spatial_affine=False)
        scalenet_params = sum(p.numel() for p in self.da_post.parameters())
        print(f"[DAScaleNetWrapper] ScaleNet initialized ({scalenet_params} params, will load from checkpoint)")

    def predict(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
        p_scale: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            rgb: (B*N, 3, H, W) - LDR RGB [0-1]
            sparse_depth: (B*N, 1, H, W) - Normalized sparse depth (divided by p_scale)
            mask: (B*N, 1, H, W) - Valid mask (>0.2)
            p_scale: (B*N, 1, 1, 1) - Scale factor for denormalization
            intrinsics: Not used

        Returns:
            metric_depth: (B*N, 1, H, W)
            aux_outputs: dict
        """
        BN, _, H, W = rgb.shape

        # DEBUG: Check RGB input (only first time)
        if not self.debug_logged:
            print(f"\n[DAScaleNet DEBUG] First forward pass:")
            print(f"  rgb input range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
            print(f"  rgb NaN: {torch.isnan(rgb).sum().item()}/{rgb.numel()}")
            print(f"  rgb Inf: {torch.isinf(rgb).sum().item()}/{rgb.numel()}")

        # CRITICAL: Clamp RGB to [0, 1] before gamma correction
        # Negative values → NaN when raised to fractional power
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if not self.debug_logged:
            print(f"  rgb (after clamp) range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")

        # Pad to be divisible by 14 (ViT requirement)
        rgb_pad, H_pad, W_pad = pad_div(rgb, div=14)

        # Preprocess: apply inverse gamma for DepthAnything
        rgb_da = rgb_pad[:, :3] ** (1 / 2.2)

        if not self.debug_logged:
            print(f"  rgb_da (after ^(1/2.2)) range: [{rgb_da.min().item():.3f}, {rgb_da.max().item():.3f}]")
            print(f"  rgb_da NaN: {torch.isnan(rgb_da).sum().item()}/{rgb_da.numel()}")

        # Backbone inference (no gradients)
        with torch.no_grad():
            D0 = self.backbone(rgb_da).view(BN, 1, H_pad, W_pad)

        # Crop back to original size
        D0 = D0[..., :H, :W]

        # DEBUG: Check DepthAnything output (only first time)
        if not self.debug_logged:
            print(f"  D0 (DepthAnything) range: [{D0.min().item():.3f}, {D0.max().item():.3f}]")
            print(f"  D0 NaN count: {torch.isnan(D0).sum().item()}/{D0.numel()}")
            print(f"  sparse_depth range: [{sparse_depth.min().item():.3f}, {sparse_depth.max().item():.3f}]")
            print(f"  sparse_depth non-zero: {(sparse_depth > 0).sum().item()}/{sparse_depth.numel()}")
            print(f"  mask sum: {mask.sum().item()}/{mask.numel()}")
            print(f"  p_scale range: [{p_scale.min().item():.3f}, {p_scale.max().item():.3f}]")

        # ScaleNet: convert relative disparity to metric depth
        metric_depth_before = self.da_post(D0, sparse_depth, mask)

        if not self.debug_logged:
            print(f"  ScaleNet output (before * p_scale): [{metric_depth_before.min().item():.3f}, {metric_depth_before.max().item():.3f}]")
            print(f"  ScaleNet NaN count: {torch.isnan(metric_depth_before).sum().item()}/{metric_depth_before.numel()}")

        metric_depth = metric_depth_before * p_scale

        if not self.debug_logged:
            print(f"  metric_depth (after * p_scale): [{metric_depth.min().item():.3f}, {metric_depth.max().item():.3f}]")
            print(f"  metric_depth NaN count: {torch.isnan(metric_depth).sum().item()}/{metric_depth.numel()}")
            self.debug_logged = True

        # Clamp to reasonable range
        metric_depth = torch.clamp(metric_depth, 0.1, 100.0)

        aux_outputs = {"rel_disp": D0}

        return metric_depth, aux_outputs


class DARANSACWrapper(DepthModuleWrapper):
    """
    Wrapper for DepthAnythingV2 + geometric RANSAC (mode="ransac").

    Uses soft RANSAC to fit affine transformation between monocular
    disparity and metric depth. No learnable parameters.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Load frozen DepthAnythingV2 backbone
        self.backbone = DepthAnythingV2(
            encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
        )
        state = torch.load(
            "modules/monodepth/DepthAnythingV2/pretrained/depth_anything_v2_vitb.pth",
            map_location='cpu'
        )
        self.backbone.load_state_dict(state)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Non-learnable RANSAC aligner
        ransac_iters = config.get('ransac_iters', 100)
        self.da_post = RANSACScaleAligner(
            n_iterations=ransac_iters, inlier_threshold=0.1
        )

    def predict(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
        p_scale: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            rgb: (B*N, 3, H, W) - LDR RGB [0-1]
            sparse_depth: (B*N, 1, H, W) - Normalized sparse depth
            mask: (B*N, 1, H, W) - Valid mask
            p_scale: (B*N, 1, 1, 1) - Scale factor
            intrinsics: Not used

        Returns:
            metric_depth: (B*N, 1, H, W)
            aux_outputs: dict
        """
        BN, _, H, W = rgb.shape

        # CRITICAL: Clamp RGB to [0, 1] before gamma correction
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # Pad and preprocess
        rgb_pad, H_pad, W_pad = pad_div(rgb, div=14)
        rgb_da = rgb_pad[:, :3] ** (1 / 2.2)

        # Backbone inference
        with torch.no_grad():
            D0 = self.backbone(rgb_da).view(BN, 1, H_pad, W_pad)

        # Crop back
        D0 = D0[..., :H, :W]

        # RANSAC scale alignment (non-learnable)
        metric_depth = self.da_post(D0, sparse_depth, mask) * p_scale

        # Clamp
        metric_depth = torch.clamp(metric_depth, 0.1, 100.0)

        aux_outputs = {"rel_disp": D0}

        return metric_depth, aux_outputs


class PromptDAWrapper(DepthModuleWrapper):
    """
    Wrapper for PromptDA depth completion (mode="promptda").

    Integrates the PromptDA module from /jarvis/modules/depth_densify/PromptDA/.
    Uses DINOv2 encoder + DPT head for sparse-to-dense depth completion.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        encoder = config.get('promptda_encoder', 'vitb')
        ckpt_path = config.get('promptda_ckpt',
                               f'modules/depth_densify/PromptDA/pretrained/promptda_{encoder}.ckpt')

        # Load PromptDA model
        try:
            self.promptda = PromptDA(encoder=encoder, ckpt_path=ckpt_path)
            self.promptda.eval()
            for p in self.promptda.parameters():
                p.requires_grad = False
            print(f"[PromptDAWrapper] Loaded PromptDA with {encoder} encoder from {ckpt_path}")
        except Exception as e:
            print(f"[PromptDAWrapper] Warning: Could not load PromptDA: {e}")
            print("[PromptDAWrapper] Using dummy model (returns sparse depth as-is)")
            self.promptda = None

    def predict(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
        p_scale: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            rgb: (B*N, 3, H, W) - LDR RGB [0-1]
            sparse_depth: (B*N, 1, H, W) - Normalized sparse depth
            mask: (B*N, 1, H, W) - Valid mask
            p_scale: (B*N, 1, 1, 1) - Scale factor
            intrinsics: Not used

        Returns:
            metric_depth: (B*N, 1, H, W)
            aux_outputs: dict
        """
        if self.promptda is None:
            # Fallback: return sparse depth
            return sparse_depth * p_scale, {}

        BN, _, H, W = rgb.shape

        # CRITICAL: Clamp RGB to [0, 1] before gamma correction
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # PromptDA expects RGB [0-1] and sparse depth in metric scale
        rgb_input = rgb ** (1 / 2.2)  # Convert to linear
        sparse_input = sparse_depth * p_scale  # Denormalize

        # Pad to be divisible by 14 (DINOv2 requirement)
        rgb_pad, H_pad, W_pad = pad_div(rgb_input, div=14)
        sparse_pad, _, _ = pad_div(sparse_input, div=14)

        # PromptDA inference
        with torch.no_grad():
            dense_depth = self.promptda.predict(rgb_pad, sparse_pad)

        # Crop back to original size
        dense_depth = dense_depth[..., :H, :W]

        # PromptDA output is already in metric scale
        dense_depth = torch.clamp(dense_depth, 0.1, 100.0)

        aux_outputs = {"sparse_input": sparse_input}

        return dense_depth, aux_outputs


class DepthPromptingWrapper(DepthModuleWrapper):
    """
    Wrapper for DepthPrompting with CSPN refinement (mode="depthprompt").

    Integrates DepthPrompting from /jarvis/modules/depth_densify/DepthPrompting/.
    Uses monocular backbone + ResNet34 sparse encoder + CSPN propagation.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        backbone = config.get('depthprompt_backbone', 'da_b')
        prop_time = config.get('depthprompt_prop_time', 3)

        # Create args namespace for DepthPrompting
        from argparse import Namespace
        args = Namespace(
            prop_kernel=9,
            conf_prop=True,
            backbone=backbone,
            max_depth=100.0,
            min_depth=0.1,
            init_scailing=False,
            prop_time=prop_time,
            data_name="NYU",
            no_res_pre=True
        )

        # Load DepthPrompting model
        try:
            from modules.depth_densify.DepthPrompting.model.ours.depth_prompt_main import depthprompting
            self.depthprompt = depthprompting(args)
            self.depthprompt.eval()
            for p in self.depthprompt.parameters():
                p.requires_grad = False
            print(f"[DepthPromptingWrapper] Loaded DepthPrompting with {backbone} backbone, {prop_time} iterations")
        except Exception as e:
            print(f"[DepthPromptingWrapper] Warning: Could not load DepthPrompting: {e}")
            print("[DepthPromptingWrapper] Using dummy model")
            self.depthprompt = None

    def predict(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
        p_scale: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            rgb: (B*N, 3, H, W) - LDR RGB [0-1]
            sparse_depth: (B*N, 1, H, W) - Normalized sparse depth
            mask: (B*N, 1, H, W) - Valid mask
            p_scale: (B*N, 1, 1, 1) - Scale factor
            intrinsics: Not used

        Returns:
            metric_depth: (B*N, 1, H, W)
            aux_outputs: dict
        """
        if self.depthprompt is None:
            return sparse_depth * p_scale, {}

        BN, _, H, W = rgb.shape

        # CRITICAL: Clamp RGB to [0, 1] before gamma correction
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # DepthPrompting expects RGB [0-1] with ImageNet normalization
        # and sparse depth in metric scale
        rgb_input = rgb ** (1 / 2.2)
        sparse_input = sparse_depth * p_scale

        # Pad to be divisible by 14
        rgb_pad, H_pad, W_pad = pad_div(rgb_input, div=14)
        sparse_pad, _, _ = pad_div(sparse_input, div=14)

        # Create sample dict
        sample = {
            "rgb": rgb_pad,
            "dep": sparse_pad
        }

        # DepthPrompting inference
        with torch.no_grad():
            output = self.depthprompt(sample)

        # Extract final prediction
        dense_depth = output["pred"]

        # Crop back to original size
        dense_depth = dense_depth[..., :H, :W]

        # Clamp
        dense_depth = torch.clamp(dense_depth, 0.1, 100.0)

        aux_outputs = {
            "pred_init": output.get("pred_init", None),
            "confidence": output.get("confidence", None)
        }

        return dense_depth, aux_outputs


class BPNetWrapper(DepthModuleWrapper):
    """
    Wrapper for BPNet bilateral propagation (mode="bpnet").

    Integrates BPNet from /jarvis/modules/depth_densify/BPNet/.
    Requires camera intrinsics. Uses custom CUDA operations.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        ckpt_path = config.get('bpnet_ckpt', None)

        # Load BPNet model
        try:
            from modules.depth_densify.BPNet.models.BPNet import Pre_MF_Post
            self.bpnet = Pre_MF_Post()

            if ckpt_path is not None:
                state = torch.load(ckpt_path, map_location='cpu')
                self.bpnet.load_state_dict(state)
                print(f"[BPNetWrapper] Loaded BPNet from {ckpt_path}")
            else:
                print("[BPNetWrapper] Warning: No checkpoint provided, using random initialization")

            self.bpnet.eval()
            for p in self.bpnet.parameters():
                p.requires_grad = False

        except Exception as e:
            print(f"[BPNetWrapper] Warning: Could not load BPNet: {e}")
            print("[BPNetWrapper] Make sure CUDA extensions are compiled: cd exts && python setup.py install")
            print("[BPNetWrapper] Using dummy model")
            self.bpnet = None

    def requires_intrinsics(self) -> bool:
        return True

    def predict(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
        p_scale: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            rgb: (B*N, 3, H, W) - LDR RGB [0-1]
            sparse_depth: (B*N, 1, H, W) - Normalized sparse depth
            mask: (B*N, 1, H, W) - Valid mask
            p_scale: (B*N, 1, 1, 1) - Scale factor
            intrinsics: (B*N, 3, 3) - REQUIRED camera intrinsics

        Returns:
            metric_depth: (B*N, 1, H, W)
            aux_outputs: dict
        """
        if self.bpnet is None:
            return sparse_depth * p_scale, {}

        if intrinsics is None:
            print("[BPNetWrapper] Error: BPNet requires camera intrinsics!")
            return sparse_depth * p_scale, {}

        BN, _, H, W = rgb.shape

        # BPNet expects RGB [0-1] and sparse depth in metric scale
        rgb_input = rgb
        sparse_input = sparse_depth * p_scale

        # BPNet inference (returns list of 6 multi-scale predictions)
        with torch.no_grad():
            outputs = self.bpnet(rgb_input, sparse_input, intrinsics)

        # Use final (full resolution) prediction
        dense_depth = outputs[-1]

        # Clamp
        dense_depth = torch.clamp(dense_depth, 0.1, 100.0)

        aux_outputs = {"multiscale_depths": outputs}

        return dense_depth, aux_outputs


class HoleFilledOnlyWrapper(DepthModuleWrapper):
    """
    Baseline wrapper that returns hole-filled input without neural completion.

    Demonstrates value of depth completion networks by showing what happens
    with only geometric interpolation (3x3 iterative averaging).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        print("[HoleFilledOnlyWrapper] Initialized (hole filling only, no neural completion)")

    def predict(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
        p_scale: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns hole-filled sparse depth without neural completion.

        Args:
            rgb: (B*N, 3, H, W) - Not used
            sparse_depth: (B*N, 1, H, W) - Normalized sparse depth (already hole-filled)
            mask: (B*N, 1, H, W) - Valid mask
            p_scale: (B*N, 1, 1, 1) - Scale factor
            intrinsics: Not used

        Returns:
            metric_depth: (B*N, 1, H, W) - Hole-filled depth
            aux_outputs: dict - Empty
        """
        # Denormalize to metric scale
        metric_sparse = sparse_depth * p_scale

        # Hole filling is already done in test script before this wrapper is called
        # (see test_ablation_inference.py simple_hole_filling function)
        # This wrapper just passes through the hole-filled input

        # Clamp to reasonable range
        metric_depth = torch.clamp(metric_sparse, 0.1, 100.0)

        return metric_depth, {}


def create_depth_module(config: dict) -> DepthModuleWrapper:
    """
    Factory function to create depth module wrapper based on config.

    Args:
        config: Configuration dictionary with 'mode' key and method-specific parameters

    Returns:
        DepthModuleWrapper instance

    Raises:
        ValueError: If mode is not supported
    """
    mode = config.get('mode', 'rs')

    if mode == 'rs':
        return DAScaleNetWrapper(config)
    elif mode == 'ransac':
        return DARANSACWrapper(config)
    elif mode == 'holefill':
        return HoleFilledOnlyWrapper(config)
    elif mode == 'promptda':
        return PromptDAWrapper(config)
    elif mode == 'depthprompt':
        return DepthPromptingWrapper(config)
    elif mode == 'bpnet':
        return BPNetWrapper(config)
    else:
        raise ValueError(f"Unknown depth mode: {mode}. Supported: rs, ransac, holefill, promptda, depthprompt, bpnet")
