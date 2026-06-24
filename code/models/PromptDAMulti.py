# mv_hybrid_allviews.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math



from .submodules.grid_warp import precompute_pairwise_T_tgt_from_src,safe_grid_sample, warp_depth_safe, warp_rgb_safe, compose_world_to_target, warp_depth_and_grid,same_view_mask_from_TK, project_points, backproject_depth, pad_div, meshgrid_xy
from .submodules.blind_snr import snr_channel_blind
from .submodules.da_scaler import ScaleNet
from .submodules.guided_util import scale_intrinsics, grid_from_target_depth, precompute_pairwise_T_src_from_tgt, guided_filter_fast, tiny_median, smoothstep, downsample_area, upsample_bilinear
from .submodules.confidencenet import ConfidenceHead
from .submodules.depth_refiner import TinyRefinerGRU, DepthSeqAggGRU, DepthResidualRefinerUNet
from .submodules.hdr_refiner import TopKHDRDenoiserUNet, WinnerTakeMostMixer, TargetlessAlignNet
from .submodules.hdr_refiner_v3 import HDRRefinerLite, DAAdapter
from .submodules.depth_holefiller import DepthHoleFiller
from .submodules.depth_ransac import RANSACScaleAligner
from .submodules.hdr_refiner_v4 import HDRRefinerV4Lite
from .submodules.hdr_refiner_v5 import HDRRefinerV5
from .submodules.hdr_guided_compositor import HDRGuidedCompositor
from .submodules.flow_estimator import FlowEstimatorTiny, warp_with_flow  # [FLOW]


import sys
sys.path.append("/jarvis")
from modules.hdr.AFUNet.models.AFUNet import AFUNet


from modules.depth_densify.PromptDA.promptda.promptda import (
    PromptDA,
)  # 사용자가 준 PromptDA 단일뷰 모듈

"""DepthAnythingV2 Scaling"""
from modules.monodepth.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2



def edge_hint_from_rgb(rgb_bn3):  # [B,N,3,H,W] -> [B,N,1,H,W]
    gx = torch.mean(
        torch.abs(rgb_bn3[:, :, :, :, 1:] - rgb_bn3[:, :, :, :, :-1]),
        dim=2,
        keepdim=True,
    )
    gy = torch.mean(
        torch.abs(rgb_bn3[:, :, :, 1:, :] - rgb_bn3[:, :, :, :-1, :]),
        dim=2,
        keepdim=True,
    )
    # pad to HxW
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.exp(-(gx + gy))  # 낮은 에지에서 더 스무스


def soft_occlusion_weights(
    D_warp,          # [B,M,N,1,H,W]  source depth warped to target view
    z_tgt,           # [B,M,1,H,W]    target/reference depth (broadcast along N)
    vis,             # [B,M,N,1,H,W]  visibility mask (0..1), may include geometric validity
    occ_margin=0.05, # absolute margin in depth units (tune per dataset)
    tau_rel=0.02,    # relative softness; smaller = sharper occlusion
    w_floor=0.05,    # minimal per-view floor weight to avoid holes
    topk=None        # optional int, e.g., 3
):
    B,M,N,_,H,W = D_warp.shape
    # delta > 0 => likely visible (source in front of target surface by margin)
    delta = (z_tgt + occ_margin) - D_warp  # [B,M,N,1,H,W]

    # relative scaling by local depth magnitude to be scale-robust
    scale = (z_tgt.abs().clamp_min(1e-3))
    s = delta / (tau_m := (tau_rel * scale))

    # soft visibility via sigmoid
    w_occ = torch.sigmoid(s)  # in (0,1)

    # combine with existing visibility/confidence
    w = w_occ * vis

    # add small floor to avoid hard zeros, then normalize along N
    if w_floor is not None and w_floor > 0:
        w = w + w_floor

    # optional Top-K pruning by score before normalization
    if topk is not None and topk < N:
        with torch.no_grad():
            # use pre-normalization weights as scores
            scores = w.squeeze(3)  # [B,M,N,H,W]
            topk_idx = scores.topk(k=topk, dim=2).indices.unsqueeze(3)  # [B,M,topk,1,H,W]
            mask = torch.zeros_like(w).scatter_(2, topk_idx, 1.0)
        w = w * mask

    # normalize to sum to 1 over views (avoid div0)
    denom = w.sum(dim=2, keepdim=True).clamp_min(1e-8)
    w_norm = w / denom  # [B,M,N,1,H,W]
    return w_norm


def warp_grid_quality_mask(
    grid: torch.Tensor,
    min_area: float = 1e-4,
    max_area: float = 1000.0,
    edge_margin: float = 1.0001,
) -> torch.Tensor:
    """Reject out-of-bounds and locally collapsed/over-stretched warps.

    grid is target->source in normalized grid_sample coordinates [B,H,W,2].
    A near-zero local Jacobian means many target pixels sample almost the same
    source pixel, which shows up as long smeared occlusion tails in HDR_init.
    """
    if grid.dim() != 4 or grid.shape[-1] != 2:
        raise ValueError(f"Expected grid [B,H,W,2], got {tuple(grid.shape)}")
    B, H, W, _ = grid.shape
    grid_f = torch.nan_to_num(grid.float(), nan=2.0, posinf=2.0, neginf=-2.0)
    inb = (grid_f[..., 0].abs() <= float(edge_margin)) & (
        grid_f[..., 1].abs() <= float(edge_margin)
    )

    if H < 2 or W < 2:
        return inb.unsqueeze(1).to(grid.dtype)

    dx = grid_f[:, :, 1:, :] - grid_f[:, :, :-1, :]
    dy = grid_f[:, 1:, :, :] - grid_f[:, :-1, :, :]
    dx = F.pad(dx.permute(0, 3, 1, 2), (0, 1, 0, 0), mode="replicate").permute(0, 2, 3, 1)
    dy = F.pad(dy.permute(0, 3, 1, 2), (0, 0, 0, 1), mode="replicate").permute(0, 2, 3, 1)

    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)
    dudx = dx[..., 0] / sx
    dvdx = dx[..., 1] / sx
    dudy = dy[..., 0] / sy
    dvdy = dy[..., 1] / sy
    area = (dudx * dvdy - dvdx * dudy).abs()
    area_ok = (area > float(min_area)) & (area < float(max_area))
    valid = (inb & area_ok).float().unsqueeze(1)

    # The Jacobian test is only meant to reject catastrophic collapses. Use a
    # soft dilation+blur so minor source/target resolution mismatch does not
    # create vertical stripe holes in HDR_init.
    dilated = F.max_pool2d(valid, kernel_size=11, stride=1, padding=5)
    softened = F.avg_pool2d(dilated, kernel_size=7, stride=1, padding=3)
    softened = torch.maximum(valid, softened * 0.95)
    softened = softened * inb.float().unsqueeze(1)
    return softened.clamp(0.0, 1.0).to(grid.dtype)


def diffuse_hdr_valid_mask(
    mask: torch.Tensor,
    support: torch.Tensor | None = None,
    dilate_kernel: int = 9,
    blur_kernel: int = 7,
    gain: float = 0.85,
) -> torch.Tensor:
    """Soften thin HDR validity holes caused by resolution mismatch.

    This is for HDR fusion only. It expands nearby valid evidence into narrow
    invalid stripes, but keeps broad unsupported areas low.
    """
    if mask is None:
        return mask
    shape = mask.shape
    x = torch.nan_to_num(mask.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    x = x.reshape(-1, 1, shape[-2], shape[-1])
    dk = max(1, int(dilate_kernel) | 1)
    bk = max(1, int(blur_kernel) | 1)
    y = F.max_pool2d(x, kernel_size=dk, stride=1, padding=dk // 2)
    y = F.avg_pool2d(y, kernel_size=bk, stride=1, padding=bk // 2)
    y = torch.maximum(x, y * float(gain))
    if support is not None:
        s = torch.nan_to_num(support.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        s = s.reshape(-1, 1, shape[-2], shape[-1])
        s = F.max_pool2d(s, kernel_size=7, stride=1, padding=3)
        y = y * s.clamp(0.0, 1.0)
    return y.reshape(shape).clamp(0.0, 1.0).to(mask.dtype)

class PromptDA_MV_All(nn.Module):
    def __init__(
        self,
        mode="pda",
        promptda_encoder="vits",
        promptda_ckpt="modules/depth_densify/PromptDA/pretrained/promptda_small.ckpt",
        use_rgb_guidance=True,
        blend_beta=0.5,
        hdr_refiner_type="topk",  # NEW: "topk" or "unet"
        topk_views=3,  # NEW: K for Top-K refiner
        use_align_net = False,
        hdr_refiner_v4 = False,
        hdr_refiner_afunet = False,
        refiner_channels = 48,
        occ_margin = 1.0,
        refiner_n_tf = 2,
        hdr_depth_coupling_floor = 0.5,
        hdr_depth_coupling_strength = 0.3,
        hdr_depth_coupling_max = 1.0,
        hdr_refiner_downsample = 1,
        use_depth_refiner = True,
        depth_refiner_base = 16,
        depth_refiner_max_delta_inv = 0.10,
        depth_refiner_downsample = 1,
        depth_merge_den_eps = 1e-4,
        depth_merge_mode = "inv_mean",
        depth_merge_peaky_gamma = 1.0,
        depth_anchor_exp_weight = 0.75,
        depth_anchor_tau = 0.05,
        depth_anchor_tau_inv = 0.03,
        depth_anchor_cluster_threshold = 0.35,
        depth_anchor_cluster_gate_tau = 0.08,
        depth_anchor_detach = True,
        hdr_use_anchor_geometry = False,
        hdr_compositor_residual_gain_init = 1e-4,
        depth_config = None,  # NEW: Configuration for depth module ablation
    ):
        super().__init__()
        self.mode = mode
        self.hdr_refiner_type = hdr_refiner_type
        self.top_k = topk_views
        self.winnerMixer = WinnerTakeMostMixer()  #
        self.occ_margin = occ_margin
        self.hdr_depth_coupling_floor = hdr_depth_coupling_floor
        self.hdr_depth_coupling_strength = hdr_depth_coupling_strength
        self.hdr_depth_coupling_max = hdr_depth_coupling_max
        self.hdr_refiner_downsample = max(1, int(hdr_refiner_downsample))
        self.depth_merge_den_eps = float(depth_merge_den_eps)
        if depth_merge_mode not in ("mean", "inv_mean", "anchor_cluster"):
            raise ValueError(f"Unsupported depth_merge_mode: {depth_merge_mode}")
        self.depth_merge_mode = depth_merge_mode
        self.depth_merge_peaky_gamma = max(1.0, float(depth_merge_peaky_gamma))
        self.depth_anchor_exp_weight = float(depth_anchor_exp_weight)
        self.depth_anchor_tau = max(1e-4, float(depth_anchor_tau))
        self.depth_anchor_tau_inv = max(1e-4, float(depth_anchor_tau_inv))
        self.depth_anchor_cluster_threshold = float(depth_anchor_cluster_threshold)
        self.depth_anchor_cluster_gate_tau = max(1e-4, float(depth_anchor_cluster_gate_tau))
        self.depth_anchor_detach = bool(depth_anchor_detach)
        self.hdr_use_anchor_geometry = bool(hdr_use_anchor_geometry)
        self.use_depth_refiner = bool(use_depth_refiner)
        self.depth_refiner = DepthResidualRefinerUNet(
            base=depth_refiner_base,
            max_delta_inv=depth_refiner_max_delta_inv,
            internal_downsample=depth_refiner_downsample,
        )
        if not self.use_depth_refiner:
            for p in self.depth_refiner.parameters():
                p.requires_grad = False

        # Initialize depth module based on config (for ablation modes)
        # Legacy modes (pda, hf, flow) use old implementation for backward compatibility
        if depth_config is not None and mode in ['rs', 'ransac', 'holefill', 'promptda', 'depthprompt', 'bpnet']:
            # New wrapper-based ablation modes
            from .submodules.depth_module_wrappers import create_depth_module
            self.depth_module = create_depth_module(depth_config)
            self.use_depth_wrapper = True
            self.load_state_before_forward = False  # For wrapper mode compatibility

            # Keep backbone reference for HDR refiner (some refiners use DA features)
            if hasattr(self.depth_module, 'backbone'):
                self.backbone = self.depth_module.backbone
            else:
                # Load a dummy backbone for refiner compatibility
                da = DepthAnythingV2(
                    encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
                )
                state = torch.load(
                    "modules/monodepth/DepthAnythingV2/pretrained/depth_anything_v2_vitb.pth"
                )
                da.load_state_dict(state)
                da.eval()
                for p in da.parameters():
                    p.requires_grad = False
                self.backbone = da
        else:
            # Legacy implementation for backward compatibility (pda, hf, flow modes)
            self.use_depth_wrapper = False

            da = DepthAnythingV2(
                    encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
            )

            state = torch.load(
                "modules/monodepth/DepthAnythingV2/pretrained/depth_anything_v2_vitb.pth"
            )
            self.load_state_before_forward = False
            da.load_state_dict(state)

            # Properly freeze all parameters
            da.eval()
            for p in da.parameters():
                p.requires_grad = False

            self.backbone = da
            if mode == "hf":
                self.hallfiler = DepthHoleFiller()

            if mode == "rs":

                # Use learnable ScaleNet for metric depth scaling
                self.da_post = ScaleNet(allow_spatial_affine=False)

            if mode == "ransac":
                da = DepthAnythingV2(
                    encoder="vitb", features=128, out_channels=[96, 192, 384, 768]
                )

                state = torch.load(
                    "/jarvis/modules/monodepth/DepthAnythingV2/pretrained/depth_anything_v2_vitb.pth"
                )
                da.load_state_dict(state)

                # Properly freeze all parameters
                da.eval()
                for p in da.parameters():
                    p.requires_grad = False

                self.backbone = da
                self.da_post = RANSACScaleAligner(n_iterations=100, inlier_threshold=0.1)

            # [FLOW] Optical flow-based alignment (no depth supervision)
            if mode == "flow":
                self.flow_net = FlowEstimatorTiny(in_ch=6, base=32)
                # 사전학습 가중치가 있다면 여기서 load 가능. 없으면 end-to-end로 함께 학습.

        self.conf_head = ConfidenceHead(in_ch=13, mid=32)
        self.use_align_net = use_align_net
        # if use_align_net:
        #     self.align_net = TargetlessAlignNet()
        if hdr_refiner_type == "compositor":
            self.hdr_refiner2 = HDRGuidedCompositor(
                base=max(32, min(refiner_channels, 56)),
                topk_views=topk_views,
                residual_gain_init=hdr_compositor_residual_gain_init,
            )
        elif hdr_refiner_v4:
            da_adapter = DAAdapter(self.backbone, extractor=None,
                                     in_norm=True, out_ch=128,
                                     freeze=True, probed_ch=768)
            self.hdr_refiner2 = HDRRefinerV4Lite(
                da_adapter = da_adapter,
                C = refiner_channels,
            )
        elif hdr_refiner_afunet:
            self.hdr_refiner2 = HDRRefinerV5(
                AFUNet(img_size=(320,480))
            )
        else:
            self.hdr_refiner2 = HDRRefinerLite(
                da_module = self.backbone,
                base = refiner_channels,
                n_tf = refiner_n_tf
            )

        self.blend_beta = blend_beta

        self.instant_pnorm = nn.InstanceNorm2d(1, affine=False)

    def per_view_prior(self, rgbs, prompts, Ks=None, return_timing=False):
        import time

        prior_timings = {}

        # rgbs:[B,N,3,H,W], prompts:[B,N,1,H,W]
        B, N, _, H_orig, W_orig = rgbs.shape

        # [FLOW] depth 기반 방식이 아니므로 depth 경로 완전히 우회
        if self.mode == "flow":
            # downstream에서 D0를 참조하지 않도록 None 반환
            # 필요 시 zeros로 대체 가능: return torch.zeros(B,N,1,H_orig//2,W_orig//2, device=rgbs.device)
            return None

        BN = B * N
        x = rgbs.view(B * N, 3, H_orig, W_orig)
        p = prompts.view(B * N, 1, H_orig, W_orig)

        # downsample to half resolution for depth completion
        x = downsample_area(x, s=0.5)
        p = downsample_area(p, s=0.5, mode="nearest")
        H_half, W_half = x.shape[-2], x.shape[-1]
        # p scale을 계산하여, p를 0~1 언저리로 nomalize.
        q = 0.9
        # (1) 공간 축 flatten → [BN, 1, H*W]
        p_abs_flat = p.abs().reshape(BN, -1).contiguous()
        p_abs_cpu = p_abs_flat.float().cpu()

        p_scale_cpu = torch.zeros(BN, 1)

        valid_threshold = 0.1
        q = 0.9

        for i in range(BN):
            vals = p_abs_cpu[i]
            m = vals > valid_threshold
            nonzero_values = vals[m]

            if nonzero_values.numel() > 0:
                p_scale_cpu[i, 0] = torch.quantile(nonzero_values, q)
            else:
                fallback_values = vals[vals > 0.01]
                if fallback_values.numel() > 0:
                    p_scale_cpu[i, 0] = torch.quantile(fallback_values, q)
                else:
                    p_scale_cpu[i, 0] = 1.0

        p_scale = p_scale_cpu.to(p.device).clamp(min=1e-3).view(BN, 1, 1, 1)

        # NEW: Use unified wrapper interface for ablation modes
        if self.use_depth_wrapper and self.mode in ['rs', 'ransac', 'holefill', 'promptda', 'depthprompt', 'bpnet']:
            # Normalize sparse depth
            p_norm = p / p_scale

            # Create mask (ensure it's on the same device as p_norm)
            mask = (p_norm > 0.2).to(p_norm.device)

            # Downsample intrinsics if provided
            Ks_half = None
            if Ks is not None:
                Ks_flat = Ks.view(BN, 3, 3)
                Ks_half = scale_intrinsics(Ks_flat, 0.5)

            # Call unified wrapper
            metric_depth, aux = self.depth_module.predict(
                rgb=x,
                sparse_depth=p_norm,
                mask=mask,
                p_scale=p_scale,
                intrinsics=Ks_half
            )

            return metric_depth.view(B, N, 1, H_half, W_half)

        # LEGACY: Original implementation for backward compatibility
        if self.mode == "rs" or self.mode == "ransac":
            # Preprocessing stage
            p = p / p_scale
            # [B*N,4,H,W]
            x_da = pad_div(x, div=14)[0]

            # x_da = self.da_preprocess(x_da)
            x_da = x_da[:, :3] ** (1 / 2.2)
            with torch.no_grad():
                D0 = self.backbone(x_da).view(B * N, 1, x_da.shape[-2], x_da.shape[-1])


            mask = (p > 0.2).to(p.device)  # [B*N, 1, 168, 252] - ensure same device
            BN, _, H_pad, W_pad = x_da.shape

            # Use ScaleNet to convert relative disparity to metric depth
            D0 = D0[..., :H_half, :W_half]
            metric_depth = self.da_post(D0, p, mask) * p_scale

            metric_depth = torch.clamp(metric_depth, 0.1, 100.0)

            return metric_depth.view(B, N, 1, H_half, W_half)

        if self.mode == "hf":
            # Preprocessing stage
            p = p / p_scale
            D0 = self.hallfiler(p)
            D0 = D0[..., :H_half, :W_half] * p_scale
            return D0.view(B, N, 1, H_half, W_half)

        if self.mode == "pda":
            # PromptDA preprocessing
            p = p / p_scale

            # Sparse to dense conversion (pass p_scale for proper far region initialization)
            p_dense, p_filled = self.sparse2dense(
                p, x ** (1 / 2.2), depth_scale=p_scale
            )  # x: LDR 3ch


            # Padding and backbone inference
            x, Hpad, Wpad = pad_div(x, div=14)
            p, _, _ = pad_div(p_dense, div=14)
            x = x ** (1 / 2.2)
            D0 = self.backbone(x, p.clamp(0, 1))  # [B*N,1,H,W]
            pnorm, _, _ = self.backbone.normalize(
                p
            )  # [B*N,1,H,W] (for confidence input)

            D0 = D0 * p_scale
            p = p * p_scale
            p_filled = p_filled * p_scale
            D0 = D0[:, :, :H_half, :W_half]
            pnorm = pnorm[:, :, :H_half, :W_half]

            if return_timing:
                return (
                    D0.view(B, N, 1, H_half, W_half),
                    pnorm.view(B, N, 1, H_half, W_half),
                    p[..., :H_half, :W_half].view(B, N, 1, H_half, W_half),
                    prior_timings,
                )
            else:
                return (
                    D0.view(B, N, 1, H_half, W_half),
                    pnorm.view(B, N, 1, H_half, W_half),
                    p[..., :H_half, :W_half].view(B, N, 1, H_half, W_half),
                    p_filled[..., :H_half, :W_half].view(B, N, 1, H_half, W_half),
                )

    def forward(
        self,
        rgbs: torch.Tensor,  # [B,N,6,H,W]
        prompts: torch.Tensor,  # [B,N,1,H,W]
        sats: torch.Tensor,  # [B,N,1,H,W]
        Ks: torch.Tensor,  # [B,N,3,3]
        Kinvs: torch.Tensor,  # [B,N,3,3]
        Ts: torch.Tensor,  # [B,N,4,4] (world <- cam) - sources
        Ks_tgt: torch.Tensor = None,  # [B,M,3,3] (targets) optional; if None, targets=sources
        Ts_tgt: torch.Tensor = None,  # [B,M,4,4] (world <- cam) (targets) optional
        #targets_is_subset: bool = True,  # if True, targets are a subset of sources (identity mapping possible)
        ref_view_lin = None,
        alpha: float = 1.0,
        ldr_min: torch.Tensor = None,  # [B,N] or [B,N,1,1,1] - GT noise level for synthetic data
        shape_tgt: Tuple[int, int] = None,  # (H,W) target shape for regridding,
        only_hdr: bool = False,  # if True, only HDR output is computed
    ):
        import time

        timings = {}
        
        if not self.load_state_before_forward and hasattr(self, "backbone"):
            state = torch.load(
                "/jarvis/modules/monodepth/DepthAnythingV2/pretrained/depth_anything_v2_vitb.pth"
            )
            
            self.backbone.load_state_dict(state)
            self.load_state_before_forward = True
            

        B, N, C6, H_orig, W_orig = rgbs.shape
        
        if shape_tgt is None:
            shape_tgt = (H_orig, W_orig)
        H_tgt, W_tgt = shape_tgt
        
        assert C6 == 6, "rgbs는 [R,G,B,R_lin,G_lin,B_lin] 6채널이어야 합니다."

        # Store original resolution
        H_half, W_half = H_orig // 2, W_orig // 2
        H_tgt_h, W_tgt_h = H_tgt // 2, W_tgt // 2

        # 1) Per-view prior depth & confidence (이미지 인코더는 LDR 3채널 사용)
        ldrs = rgbs[:, :, :3]  # [B,N,3,H,W]
        D0 = self.per_view_prior(
            ldrs, prompts, Ks=Ks, return_timing=False
        )  # [B,N,1,H_half,W_half]

        # In PDA mode, the 4th element is p_filled
        
        # [FLOW] Handle None D0 from flow mode
        if D0 is None:
            # Use zeros as dummy depth for flow mode
            D0 = torch.zeros(B, N, 1, H_half, W_half, device=rgbs.device)
        
        pnorm = self.instant_pnorm(D0.view(B * N, 1, H_half, W_half)).view(
            B, N, 1, H_half, W_half
        )
  
        rgbs_half = downsample_area(rgbs.view(B * N, C6, H_orig, W_orig), s=0.5).view(
            B, N, C6, H_half, W_half
        )
        sats_half = downsample_area(sats.view(B * N, 1, H_orig, W_orig), s=0.5).view(
            B, N, 1, H_half, W_half
        )

        # Scale intrinsics for half resolution
        Ks_half = scale_intrinsics(Ks, s=0.5)
        Kinvs_half = torch.linalg.inv(Ks_half.float()).contiguous()

        # T4: Prepare enhanced features for confidence head
        # Use only observable features (no GT dependency)
        # Training losses will guide ConfidenceHead to learn from weight distribution

        C = self.conf_head(rgbs_half, sats_half, pnorm)  # [B,N,1,H_half,W_half]

        # If target intrinsics/extrinsics not provided, assume targets == sources
        if Ks_tgt is None:
            Ks_tgt_half = Ks_half
        else:
            Ks_tgt_half = scale_intrinsics(Ks_tgt, s=0.5)

        if Ts_tgt is None:
            Ts_tgt_use = Ts
        else:
            Ts_tgt_use = Ts_tgt

        # Compute pairwise transforms [B, M, N, 4, 4]
        T_tgt_from_src = precompute_pairwise_T_tgt_from_src(Ts, Ts_tgt_use)
        Bt, M, N_check, _, _ = T_tgt_from_src.shape
        # same_view_mask = same_view_mask_from_TK(T_tgt_from_src, Ks, Ks_tgt)
        # ldr_orig_expanded = (
        #     rgbs[:, :, :3].unsqueeze(1).expand(B, M, N, 3, H_orig, W_orig)
        # )
        # lin_orig_expanded = (
        #     rgbs[:, :, 3:].unsqueeze(1).expand(B, M, N, 3, H_orig, W_orig)
        # )
        # T_tgt_from_src: [B, M, N, 4, 4]

        # Expand depths/intrinsics to [B, M, N, ...]
        # B and N already known from inputs; infer M from T_tgt_from_src

        assert Bt == B, f"Batch mismatch: {Bt} vs {B}"
        assert N_check == N, f"Source count mismatch: {N_check} vs {N}"

        # ===== [FLOW] Branch: Optical-flow-based all-to-reference warping =====
        if self.mode == "flow":
            assert ref_view_lin is not None, "flow 모드에서는 ref_view_lin (B x M x 3 x H x W)이 필요합니다."
            
            # src: 각 입력 N, tgt: 각 타깃 M 에 대해 flow 추정
            # 여기서는 Linear 채널을 사용 (감마/노출 영향 최소화)
            src_lin = rgbs[:, :, 3:, :, :]      # [B,N,3,H,W]
            tgt_lin = ref_view_lin              # [B,M,3,H,W]
            B_flow, M, _, H, W = tgt_lin.shape

            ldr_warp_full_list = []
            lin_warp_full_list = []
            valid_geo_full_list = []
            flow_list = []  # [FLOW] Store estimated flows for aux output

            # 간단 구현: (m,n) 쌍 루프 (실사용 시 배치 묶음으로 최적화 권장)
            for m in range(M):
                ldr_row = []
                lin_row = []
                valid_row = []
                tgt = tgt_lin[:, m]                        # [B,3,H,W]
                flow_list_sub = []
                for n in range(N):
                    srcL = src_lin[:, n]                   # [B,3,H,W]
                    
                    # Flow 추정은 linear로; LDR도 같은 flow로 warp
                    with torch.cuda.amp.autocast(enabled=False):
                        flow = self.flow_net(srcL.float(), tgt.float())    # [B,2,H,W]

                    # linear warp
                    w_lin, v_lin = warp_with_flow(srcL, flow)              # [B,3,H,W], [B,1,H,W]
                    
                    # ldr warp (같은 flow 재사용)
                    srcLDR = rgbs[:, n, :3]                                 # [B,3,H,W]
                    w_ldr, v_ldr = warp_with_flow(srcLDR, flow)

                    # valid(가시성) 근사: grid 범위 valid 만 사용
                    valid = v_lin

                    ldr_row.append(w_ldr)
                    lin_row.append(w_lin)
                    valid_row.append(valid)
                    
                    # Store first flow for debugging (all flows can be stored similarly)
                    flow_list_sub.append(flow)

                ldr_warp_full_list.append(torch.stack(ldr_row, dim=1))     # [B,N,3,H,W]
                lin_warp_full_list.append(torch.stack(lin_row, dim=1))     # [B,N,3,H,W]
                valid_geo_full_list.append(torch.stack(valid_row, dim=1))  # [B,N,1,H,W]
                flow_list.append(torch.stack(flow_list_sub, dim=1))        # [B,N,2,H,W]
            ldr_warp_full = torch.stack(ldr_warp_full_list, dim=1)         # [B,M,N,3,H,W]
            lin_warp_full = torch.stack(lin_warp_full_list, dim=1)         # [B,M,N,3,H,W]
            valid_geo_up  = torch.stack(valid_geo_full_list, dim=1)        # [B,M,N,1,H,W]
            
            # Depth 관련 텐서는 dummy 로 채움 (refiner에 optional로 들어감)
            D_fused = torch.ones(B_flow, M, 1, H, W, device=rgbs.device) * 10.0
            D_fused_raw_for_aux = D_fused
            depth_refiner_aux = {}
            D_warp_full = torch.ones(B_flow, M, N, 1, H, W, device=rgbs.device) * 10.0
            rerr = torch.zeros(B_flow, M, N, 1, H//2, W//2, device=rgbs.device)
            geo_rel_full = valid_geo_up.clone()
            
            # [FLOW] confidence 를 valid_geo_up에 곱하여 가중치 반영
            # C: [B,N,1,H_half,W_half] -> upsample to [B,N,1,H,W]
            C_upsampled = F.interpolate(
                C.reshape(B * N, 1, H_half, W_half),
                size=(H, W),
                mode="bilinear",
                align_corners=True
            ).reshape(B, N, 1, H, W)  # [B,N,1,H,W]
            
            # Confidence를 모든 target에 브로드캐스트: [B,M,N,1,H,W]
            C_warp = C_upsampled.unsqueeze(1).expand(B, M, N, 1, H, W)
            # valid_geo_up에 confidence 반영
            valid_geo_up = valid_geo_up * C_warp

            # [FLOW] flow 모드에서는 align_net 미사용
            # ldr_warp_full, lin_warp_full, valid_geo_up이 준비됨
            # 이후 공통 코드 경로로 진행
        
        else:
            # ===== [DEPTH] Original depth-based warping path =====
            D_src = D0.unsqueeze(1).expand(B, M, N, 1, H_half, W_half).contiguous()
            Kinv_src = Kinvs_half.unsqueeze(1).expand(B, M, N, 3, 3).contiguous()
            K_tgt = Ks_tgt_half.unsqueeze(2).expand(B, M, N, 3, 3).contiguous()

            Bmn = B * M * N
            D_src_m = D_src.reshape(Bmn, 1, H_half, W_half)
            C_expanded = C.unsqueeze(1).expand(B, M, N, 1, H_half, W_half).contiguous()
            Kinv_src_m = Kinv_src.reshape(Bmn, 3, 3)
            T_tgt_from_src_m = T_tgt_from_src.contiguous().reshape(Bmn, 4, 4)
            K_tgt_m = K_tgt.reshape(Bmn, 3, 3)

            # 2-1) 깊이 워핑 + grid 획득 (half resolution)
            D_warp_m, grid_m, vis_m, rerr_m, z_m, _, C_warp_m = warp_depth_and_grid(
                D_src_m,
                Kinv_src_m,
                T_tgt_from_src_m,
                K_tgt_m,
                H=H_tgt_h,
                W=W_tgt_h,
                C=C_expanded.view(Bmn, 1, H_half, W_half),
            )

            D_warp = D_warp_m.reshape(B, M, N, 1, H_tgt_h, W_tgt_h)
            vis = vis_m.reshape(B, M, N, 1, H_tgt_h, W_tgt_h)
            rerr = rerr_m.reshape(B, M, N, 1, H_tgt_h, W_tgt_h)
            z_tgt = z_m.reshape(B, M, N, 1, H_tgt_h, W_tgt_h)
            C_warp = C_warp_m.reshape(B, M, N, 1, H_tgt_h, W_tgt_h)

            ldr_half = rgbs_half[:, :, :3]
            luma_half = (
                0.2126 * ldr_half[:, :, 0:1]
                + 0.7152 * ldr_half[:, :, 1:2]
                + 0.0722 * ldr_half[:, :, 2:3]
            ).clamp(0.0, 1.0)
            dark_ok_depth = ((luma_half - 0.03) / 0.17).clamp(0.0, 1.0)
            bright_ok_depth = ((0.97 - luma_half) / 0.37).clamp(0.0, 1.0)
            midtone_ok = torch.exp(-torch.abs(torch.log(luma_half.clamp_min(1e-4)) - math.log(0.25)) / 1.25)
            sat_ok_depth = (1.0 - sats_half).clamp(0.0, 1.0)
            q_depth_exp_src = (dark_ok_depth * bright_ok_depth * midtone_ok * sat_ok_depth).clamp(0.0, 1.0)
            q_depth_exp_src_m = q_depth_exp_src.unsqueeze(1).expand(B, M, N, 1, H_half, W_half).contiguous()
            q_depth_exp_warp = F.grid_sample(
                q_depth_exp_src_m.reshape(Bmn, 1, H_half, W_half),
                grid_m,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(B, M, N, 1, H_tgt_h, W_tgt_h).clamp(0.0, 1.0)
            
            # valid_warp = valid_warp_m.view(B, M, N, 1, H_half, W_half)
            
            # not_occluded = (
            #     D_warp <= (z_tgt + occ_margin)
            # ).float()  # [B,M,N,1,H_tgt_h,W_tgt_h]

            # valid_geo = vis * not_occluded  # Simplified: vis + occlusion only
            valid_geo = soft_occlusion_weights(
                D_warp,
                z_tgt,
                vis,
                occ_margin=self.occ_margin,
            )
            # Upsample warping grid to full resolution
            grid_full = F.interpolate(
                grid_m.reshape(Bmn, H_tgt_h, W_tgt_h, 2).permute(0, 3, 1, 2),
                size=(H_tgt, W_tgt),
                mode="bilinear",
                align_corners=True,
            ).permute(
                0, 2, 3, 1
            )  # [Bmn,H_orig,W_orig,2]


            ldr_warp_full_list = []
            lin_warp_full_list = []
            src_grid_list = []
            grid_valid_full_list = []
            sats_warp_list = []
            for tgt_idx in range(M):
                srcs_ldr = []
                srcs_lin = []
                srcs_grid = []
                srcs_grid_valid = []
                sats_warped = []
                for src_idx in range(N):
                    pair_idx = tgt_idx * N + src_idx

                    # Source images at full resolution
                    src_ldr = rgbs[:, src_idx, :3, :, :]  # [B,3,H_orig,W_orig]
                    src_lin = rgbs[:, src_idx, 3:, :, :]  # [B,3,H_orig,W_orig]
                    src_grid_full = grid_full[
                        pair_idx : (Bmn) : (M * N)
                    ]  # [B,H_orig,W_orig,2]
                    grid_valid_full = warp_grid_quality_mask(src_grid_full)

                    # Warp with zero padding. Invalid/occluded pixels should stay
                    # black in HDR_init rather than repeating source borders.
                    warped_ldr = F.grid_sample(
                        src_ldr,
                        src_grid_full,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
                    warped_lin = F.grid_sample(
                        src_lin,
                        src_grid_full,
                        mode="bilinear",
                        padding_mode="zeros",
                      align_corners=True,
                    )
                    warped_sat = F.grid_sample(
                        sats[:, src_idx],
                        src_grid_full,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
                    sats_warped.append(warped_sat)

                    srcs_ldr.append(warped_ldr)
                    srcs_lin.append(warped_lin)
                    srcs_grid.append(src_grid_full)
                    srcs_grid_valid.append(grid_valid_full)
                    
                    

                # stack sources for this target: [B,N,3,H,W]
                ldr_warp_full_list.append(torch.stack(srcs_ldr, dim=1))
                lin_warp_full_list.append(torch.stack(srcs_lin, dim=1))
                src_grid_list.append(torch.stack(srcs_grid, dim=1))
                grid_valid_full_list.append(torch.stack(srcs_grid_valid, dim=1))
                sats_warp_list.append(torch.stack(sats_warped, dim=1))
                

            # stack targets: [B,M,N,3,H,W]
            ldr_warp_full = torch.stack(ldr_warp_full_list, dim=1)
            lin_warp_full = torch.stack(lin_warp_full_list, dim=1)
            src_grid_full = torch.stack(src_grid_list, dim=1)
            grid_valid_full = torch.stack(grid_valid_full_list, dim=1)
            sats_warp_full = torch.stack(sats_warp_list, dim=1)
        
            # 3) Confidence-weighted fusion (Depth) at half resolution


            # SAFETY CHECK: Verify all tensors have matching dimensions
            # This prevents cryptic broadcast errors

            assert rerr.shape[1] == M, f"rerr dimension mismatch: {rerr.shape[1]} vs M={M}"
            assert C.shape[1] == N, f"C dimension mismatch: {C.shape[1]} vs N={N}"
            assert (
                sats_half.shape[1] == N
            ), f"sats_half dimension mismatch: {sats_half.shape[1]} vs N={N}"
            assert (
                valid_geo.shape[1] == M
            ), f"valid_geo dimension mismatch: {valid_geo.shape[1]} vs M={M}"

            fusion_alpha = torch.tensor(alpha, device=rerr.device, dtype=rerr.dtype)
            # print(fusion_alpha.item(), rerr.shape, C_src.shape, SAT.shape, valid_geo.shape)

            # Ensure all tensors have compatible shapes
            w_base = (
                torch.exp(-fusion_alpha * torch.clamp(rerr, 0, 1))
                * C_warp
                # * (1.0 - SAT)
                * valid_geo
            )
            depth_valid = (
                torch.isfinite(D_warp)
                & (D_warp > 0.05)
                & (D_warp < 1000.0)
            ).to(dtype=w_base.dtype)
            w_base = w_base * depth_valid
            D_warp_merge = torch.where(
                depth_valid > 0,
                D_warp,
                torch.full_like(D_warp, 1000.0),
            )

            depth_anchor_aux = {}
            anchor_geometry_compat_half = None
            depth_anchor_posterior_half = None
            if self.depth_merge_peaky_gamma > 1.0:
                # Sharpen source selection at geometry conflicts without changing
                # the underlying confidence/reprojection terms.
                w = w_base.clamp_min(0.0).pow(self.depth_merge_peaky_gamma)
            else:
                w = w_base

            inv_warp = 1.0 / D_warp_merge.clamp_min(0.05)
            if self.depth_merge_mode in ("inv_mean", "anchor_cluster"):
                num = (w * inv_warp).sum(dim=2)
            else:
                num = (w * D_warp_merge).sum(dim=2)  # [B,M,1,H_half,W_half]

            den_raw = w.sum(dim=2)
            den = den_raw.clamp_min(self.depth_merge_den_eps)
            if self.depth_merge_mode in ("inv_mean", "anchor_cluster"):
                D_fused_half = 1.0 / (num / den).clamp_min(1e-6)
            else:
                D_fused_half = num / den

            if self.depth_merge_mode == "anchor_cluster":
                exp_weight = max(0.0, self.depth_anchor_exp_weight)
                w_depth = w * q_depth_exp_warp.clamp_min(1e-4).pow(exp_weight)
                log_score = torch.log(w_depth.clamp_min(1e-12)) / self.depth_anchor_tau
                log_score = log_score.masked_fill(depth_valid <= 0, -1e4)
                p_anchor = torch.softmax(log_score, dim=2) * depth_valid
                p_anchor = p_anchor / p_anchor.sum(dim=2, keepdim=True).clamp_min(1e-6)
                inv_anchor = (p_anchor * inv_warp).sum(dim=2, keepdim=True)
                inv_anchor_ref = inv_anchor.detach() if self.depth_anchor_detach else inv_anchor
                compat = torch.exp(-torch.abs(inv_warp - inv_anchor_ref) / self.depth_anchor_tau_inv) * depth_valid
                w_cluster = w_depth * compat
                den_cluster_raw = w_cluster.sum(dim=2)
                den_cluster = den_cluster_raw.clamp_min(self.depth_merge_den_eps)
                inv_cluster = (w_cluster * inv_warp).sum(dim=2) / den_cluster
                inv_fallback = num / den
                cluster_strength = (den_cluster_raw / w_depth.sum(dim=2).clamp_min(self.depth_merge_den_eps)).clamp(0.0, 1.0)
                gate_cluster = torch.sigmoid(
                    (cluster_strength - self.depth_anchor_cluster_threshold)
                    / self.depth_anchor_cluster_gate_tau
                )
                inv_fused = gate_cluster * inv_cluster + (1.0 - gate_cluster) * inv_fallback
                D_fused_half = 1.0 / inv_fused.clamp_min(1e-6)
                w = w_cluster
                anchor_geometry_compat_half = compat
                depth_anchor_posterior_half = p_anchor
                with torch.no_grad():
                    entropy = -(p_anchor * (p_anchor + 1e-6).log()).sum(dim=2)
                    if N > 1:
                        entropy = entropy / math.log(float(N))
                    inv_mean_dbg = (p_anchor * inv_warp).sum(dim=2, keepdim=True)
                    inv_var_dbg = (p_anchor * (inv_warp - inv_mean_dbg).pow(2)).sum(dim=2).sqrt()
                    depth_anchor_aux = {
                        "depth_anchor_exp_mean": q_depth_exp_warp.mean().detach(),
                        "depth_anchor_entropy": entropy.mean().detach(),
                        "depth_anchor_cluster_strength": cluster_strength.mean().detach(),
                        "depth_anchor_gate_mean": gate_cluster.mean().detach(),
                        "depth_anchor_compat_mean": compat.mean().detach(),
                        "depth_anchor_fallback_ratio": (1.0 - gate_cluster).mean().detach(),
                        "depth_anchor_inv_var": inv_var_dbg.mean().detach(),
                    }

            # Per-target view normalization across N source views.
            # Used both for (a) coupling loss with w_combined and (b) optional HDR mixing.
            w_base_sum = w.sum(dim=2, keepdim=True).clamp_min(self.depth_merge_den_eps)
            w_base_norm_half = w / w_base_sum  # [B,M,N,1,H_h,W_h]
            D_fused_raw_half = D_fused_half
            depth_refiner_aux = {}
            if self.use_depth_refiner:
                D_fused_half, depth_refiner_aux = self.depth_refiner(
                    D_fused=D_fused_half,
                    D_warp=D_warp_merge,
                    w_norm=w_base_norm_half,
                    rerr=rerr,
                    w_strength=w.sum(dim=2),
                )

            # Upsample depth maps back to original resolution (vectorized, no guided filter)
            # Guided filter removed because RGB can be noisy and introduce artifacts


            # Vectorized upsampling: merge B and N dimensions
            D0_full = upsample_bilinear(
                D0.reshape(B * N, 1, H_half, W_half), (H_orig, W_orig)
            ).reshape(B, N, 1, H_orig, W_orig)

            D_fused_raw_full = upsample_bilinear(
                D_fused_raw_half.reshape(B * M, 1, H_tgt_h, W_tgt_h), (H_tgt, W_tgt)
            ).reshape(B, M, 1, H_tgt, W_tgt)

            D_fused_full = upsample_bilinear(
                D_fused_half.reshape(B * M, 1, H_tgt_h, W_tgt_h), (H_tgt, W_tgt)
            ).reshape(B, M, 1, H_tgt, W_tgt)

            # Upsample warped depth for visualization
            D_warp_full = upsample_bilinear(
                D_warp_merge.reshape(B * M * N, 1, H_tgt_h, W_tgt_h), (H_tgt, W_tgt)
            ).reshape(B, M, N, 1, H_tgt, W_tgt)

            # Update variables to use full resolution depth

            D_fused = D_fused_full.clamp(min=0.05, max=1000.0)
            D_fused_raw_for_aux = D_fused_raw_full.clamp(min=0.05, max=1000.0)


            # ===== T2: Log-domain HDR Fusion with Median+MAD Consensus =====


            ldr_final = ldr_warp_full
            lin_final = lin_warp_full
            valid_geo_up = (
                            F.interpolate(
                                valid_geo.view(B * M * N, 1, H_tgt_h, W_tgt_h),
                                size=(H_tgt, W_tgt),
                                mode="bilinear",
                                align_corners=True,
                            )
                            .view(B, M, N, 1, H_tgt, W_tgt)
                            .float()
                        )
            grid_valid_full = grid_valid_full.to(valid_geo_up.dtype)
            valid_geo_up = (valid_geo_up * grid_valid_full).clamp(0.0, 1.0)
            valid_geo_up = diffuse_hdr_valid_mask(
                valid_geo_up,
                support=grid_valid_full,
                dilate_kernel=15,
                blur_kernel=9,
                gain=0.95,
            )
        
        # ===== Common Path for All Modes (depth/flow/etc) =====
        # [FLOW] 모든 모드에서 공통으로 실행: align_net 적용 (선택) 후 HDR 초기화
        
        # align_net 사용 시에만 grid 기반 refinement (flow 모드 제외)
        if self.use_align_net and self.mode != "flow":
            grid_refined, ldr_refined = self.align_net(
                ldr_warp_full, valid_geo_up, src_grid_full,
            )
            # 선택: 동일 Δ를 linear에도 적용
            B_align, M_align, N_align, _, H_align, W_align = ldr_refined.shape
            lin_refined = []
            valid_geo_refined = []
            for m in range(M_align):
                row = []
                row_geo = []
                for n in range(N_align):
                    g = grid_refined[:, m, n]  # [B,H,W,2]
                    src_lin = lin_warp_full[:, m, n]
                    row_geo.append(
                        F.grid_sample(
                            valid_geo_up[:, m, n], g, mode="bilinear", padding_mode="border", align_corners=True
                        )
                    )
                    row.append(F.grid_sample(src_lin, g, mode="bilinear", padding_mode="border", align_corners=True))
                lin_refined.append(torch.stack(row, dim=1))
                valid_geo_refined.append(torch.stack(row_geo, dim=1))
            lin_refined = torch.stack(lin_refined, dim=1)  # [B,M,N,3,H,W]
            valid_geo_refined = torch.stack(valid_geo_refined, dim=1)  # [B,M,N,1,H,W]
        else:
            # align_net 미사용 또는 flow 모드: 그냥 warp 사용
            lin_refined = lin_warp_full
            valid_geo_refined = valid_geo_up
            ldr_refined = ldr_warp_full
            
        # ===== T2: Common HDR Fusion for All Modes =====
        # flow/depth/etc 모든 모드에서 동일한 코드 실행
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure FP32
            lin_final_fp32 = lin_refined.float()
            ldr_final_fp32 = ldr_refined.float()
            sat_mask = (ldr_final_fp32 > 0.999).float()

            valid_mask = valid_geo_refined.expand_as(sat_mask) * (1.0 - sat_mask)

            # Couple depth-merge weight into HDR-merge weight (P1 design intent).
            # Why: previously w_base (geometry+confidence) and w_trap (saturation/brightness)
            # were independent; ConfidenceHead got no HDR-loss gradient. Multiplying valid_geo
            # by w_base_norm routes HDR gradient back through C, and aligns the two weight
            # systems so that "same merge weight that recovers depth also recovers HDR".
            w_base_norm_up_full = None
            if self.mode != "flow":
                # w_base_norm_half: [B,M,N,1,H_tgt_h,W_tgt_h] (computed in depth branch).
                # Detach for forward-only gating: this preserves the design intent
                # (depth fusion weight informs HDR fusion) without routing HDR-loss
                # gradient back through w_base — the per-view division by sum can
                # produce 1/(sum+eps)^2 spikes during backward when w_base.sum→0.
                # ConfidenceHead still trains via D_fused→depth_loss path.
                w_base_norm_up_full = F.interpolate(
                    w_base_norm_half.detach().reshape(B * M * N, 1, H_tgt_h, W_tgt_h),
                    size=(H_tgt, W_tgt),
                    mode="bilinear",
                    align_corners=True,
                ).view(B, M, N, 1, H_tgt, W_tgt)
                # Rescale to keep average ~1.0 across N, then soften the coupling.
                # Directly multiplying by w_base_norm*N was too sharp on small shards:
                # low-confidence views disappeared from HDR fusion and qualitative
                # predictions became dark. The floor keeps every geometrically valid
                # view alive while still biasing HDR fusion toward the depth merge.
                if self.hdr_use_anchor_geometry and anchor_geometry_compat_half is not None:
                    w_base_norm_up_full = F.interpolate(
                        anchor_geometry_compat_half.detach().reshape(B * M * N, 1, H_tgt_h, W_tgt_h),
                        size=(H_tgt, W_tgt),
                        mode="bilinear",
                        align_corners=True,
                    ).view(B, M, N, 1, H_tgt, W_tgt).clamp(0.0, 1.0)
                else:
                    w_base_norm_up_full = w_base_norm_up_full * float(N)
                if self.hdr_depth_coupling_strength <= 0:
                    w_base_norm_up_full = None
                else:
                    w_base_norm_up_full = (
                        self.hdr_depth_coupling_floor
                        + self.hdr_depth_coupling_strength * w_base_norm_up_full
                    )
                    w_base_norm_up_full = w_base_norm_up_full.clamp_min(
                        self.hdr_depth_coupling_floor
                    )
                    if self.hdr_depth_coupling_max > 0:
                        w_base_norm_up_full = w_base_norm_up_full.clamp_max(
                            self.hdr_depth_coupling_max
                        )

            # Compute per-target robust statistics (median, MAD)
            # Process each target view independently
            HDR_init_list = []
            hdr_init_valid_list = []
            w_over_list = []
            w_under_list = []
            w_combined_list = []
            w_effective_list = []
            if self.mode != "flow":
                rerr_full_for_hdr = upsample_bilinear(
                    rerr.reshape(B * M * N, 1, H_tgt_h, W_tgt_h), (H_tgt, W_tgt)
                ).view(B, M, N, 1, H_tgt, W_tgt).float()
                geo_rel_for_hdr = torch.exp(-rerr_full_for_hdr / 1.25)
            else:
                rerr_full_for_hdr = None
                geo_rel_for_hdr = None

            for tgt_idx in range(M):
                # Extract all source views for this target: [B,N,3,H,W]
                lin_tgt = lin_final_fp32[:, tgt_idx]  # [B,N,3,H,W]
                ldr_tgt = ldr_final_fp32[:, tgt_idx]  # [B,N,3,H,W]
                valid_tgt = valid_mask[:, tgt_idx]  # [B,N,3,H,W]
                
                # GT-based approach: Hard threshold at noise floor
                if ldr_min.ndim == 2:
                    ldr_min_expanded = ldr_min.view(B, N, 1, 1, 1)  # [B,N,1,1,1]
                else:
                    ldr_min_expanded = ldr_min  # Already [B,N,1,1,1]

                ldr_min_src = ldr_min_expanded  # per-source noise floor [B,N,1,1,1]

                ldr_max = 0.99
                ldr_mid = 0.25

                # Use channel-mined LDR as a compact brightness indicator
                ldr_minch = ldr_tgt.min(dim=2, keepdim=True).values  # [B,N,1,H,W]

                # over_ok: 1 in safe range, ramps to 0 near ldr_max
                over_ok = torch.ones_like(ldr_minch)
                over_ok = torch.where(
                    ldr_minch <= ldr_mid, over_ok, (ldr_max - ldr_minch) / (ldr_max - ldr_mid)
                )
                #print(sats_warp_full[:, tgt_idx].shape, over_ok.shape, ldr_tgt.shape, ldr_minch.shape)
                if locals().get("sats_warp_full") is not None:
                    over_ok = over_ok * sats_warp_full[:, tgt_idx]
                over_ok = over_ok.clamp(0.0, 1.0)
                
                

                # dark_ok: 1 for mid/bright, ramps from 0->1 between dark and mid
                dark_ok = torch.zeros_like(ldr_minch)
                dark_mask_hi = ldr_minch >= ldr_mid
                dark_floor = (ldr_min_src * 1.5).clamp_max(ldr_mid * 0.95)
                dark_mask_mid = (ldr_minch > dark_floor) & (ldr_minch < ldr_mid)
                dark_ok = torch.where(dark_mask_hi, torch.ones_like(ldr_minch), dark_ok)
                dark_ok = torch.where(
                    dark_mask_mid,
                    (ldr_minch - dark_floor) / (ldr_mid - dark_floor).clamp_min(1e-4),
                    dark_ok,
                )
                dark_ok = dark_ok.clamp(0.0, 1.0)

                # Stack into [B,N,2,H,W] (chan0=over_ok, chan1=dark_ok)
                w_trap_ch = torch.cat([over_ok, dark_ok], dim=2)
                # Store w_combined and snr_weight for visualization
                w_combined = over_ok * dark_ok  # [B,N,1,H,W]
                w_combined_list.append(w_combined.detach())
                w_over_list.append(over_ok.detach())
                w_under_list.append(dark_ok.detach())

                # Prepare valid_geo: collapse channel if present
                if valid_tgt.shape[2] == 3:
                    _valid_geo = valid_tgt.min(dim=2, keepdim=True).values
                else:
                    _valid_geo = valid_tgt

                # Couple per-view depth merge weight into HDR fusion (design intent).
                # _valid_geo: [B,N,1,H,W] ; w_base_norm_up_full[:, tgt_idx]: [B,N,1,H,W]
                if w_base_norm_up_full is not None:
                    _valid_geo = _valid_geo * w_base_norm_up_full[:, tgt_idx]
                if geo_rel_for_hdr is not None:
                    _valid_geo = _valid_geo * geo_rel_for_hdr[:, tgt_idx]

                # Final HDR evidence is where the thin stripe holes matter most:
                # geometry may already be softened, but exposure/depth weights can
                # make the actual mixer mask sparse again. Diffuse the final
                # per-source evidence, constrained by nearby warp support.
                evidence_raw = (_valid_geo.clamp(0.0, 1.0) * w_combined).clamp(0.0, 1.0)
                evidence_support = None
                if locals().get("grid_valid_full", None) is not None:
                    evidence_support = grid_valid_full[:, tgt_idx].to(evidence_raw.dtype)
                evidence_soft = diffuse_hdr_valid_mask(
                    evidence_raw,
                    support=evidence_support,
                    dilate_kernel=15,
                    blur_kernel=9,
                    gain=0.95,
                )
                w_effective_list.append(evidence_soft.detach())
                hdr_init_valid_list.append(evidence_soft.sum(dim=1).clamp(0.0, 1.0).detach())

                # Reference LDR (중앙값)
                ldr_tgt_ref = torch.median(ldr_tgt, dim=1, keepdim=True).values

                # Call the mixer with warp된 이미지만 사용
                lin_out_t = self.winnerMixer(
                    lin_src=lin_tgt,
                    ldr_src=ldr_tgt,
                    ldr_tgt=ldr_tgt_ref,
                    valid_geo=evidence_soft,
                    w_trap=torch.ones_like(w_trap_ch),
                    snr_norm=None,
                    rerr=None,
                )

                # Normalize mixer output shape to [B,3,H,W]
                if lin_out_t.dim() == 4:  # [B,3,H,W]
                    hdr_t = lin_out_t
                else:  # [B,1,3,H,W]
                    hdr_t = lin_out_t.squeeze(1)

                # Store for this target view
                HDR_init_list.append(hdr_t.clamp_min(0.0))

            # Stack across target views: [B,M,3,H,W]
            HDR_init = torch.stack(HDR_init_list, dim=1)
            HDR_init = torch.clamp(HDR_init, min=0.0, max=10.0)
            hdr_init_valid = torch.stack(hdr_init_valid_list, dim=1)

            # geo_rel_full 계산 (depth 모드에서만)
            if self.mode != "flow":
                rerr_upsampled_full = rerr_full_for_hdr
                sigma_geo = 0.5
                geo_rel_full_computed = torch.exp(-rerr_upsampled_full / sigma_geo)  # [B,M,N,1,H,W]
            else:
                # flow 모드에서는 geo_rel_full이 이미 valid_geo_up으로 설정됨
                geo_rel_full_computed = geo_rel_full

        HDR_init_detached = HDR_init.detach()

        # valid_warp_dfused: max over source views for each target
        valid_warp_dfused = valid_geo_up.max(dim=2)[0]  # [B,M,1,H,W]

        # depth 모드에서만 D_warp_full_saved 생성
        if self.mode != "flow":
            D_warp_full_saved = D_warp_full.detach()
            geo_rel_full_saved = geo_rel_full_computed.detach()
        else:
            # flow 모드에서는 dummy depth
            D_warp_full_saved = D_warp_full.detach()
            geo_rel_full_saved = valid_geo_up.detach()
    
        # regrid off: use original HDR_init and warped inputs
        HDR_init_final = HDR_init_detached

        ldr_for_refiner = ldr_refined.detach()
        lin_for_refiner = lin_refined.detach()
        # Use the final, diffused HDR evidence for the compositor's source-fill
        # route. Raw valid_geo/w_combined can reintroduce hard mask boundaries
        # even when HDR_init itself has been softened.
        hdr_evidence_for_refiner = torch.stack(w_effective_list, dim=1).detach()
        valid_for_refiner = hdr_evidence_for_refiner
        snr_norm = hdr_evidence_for_refiner  # [B,M,N,1,H,W]
        
        def _resize_spatial(t, size, mode="bilinear"):
            if t is None or t.shape[-2:] == size:
                return t
            shape = t.shape
            c = shape[-3]
            x = t.reshape(-1, c, shape[-2], shape[-1]).float()
            if mode == "nearest":
                x = F.interpolate(x, size=size, mode=mode)
            else:
                x = F.interpolate(x, size=size, mode=mode, align_corners=False)
            return x.to(dtype=t.dtype).reshape(*shape[:-2], *size)

        refiner_out_size = HDR_init_final.shape[-2:]
        refiner_size = refiner_out_size
        if self.hdr_refiner_downsample > 1:
            # HDRRefinerLite applies window attention after two stride-2 encoder
            # stages, so the refiner input must be divisible by win * 4 (=32
            # for the current win=8). Keep the final output size unchanged; this
            # only adjusts the temporary low-res refiner canvas.
            refiner_multiple = 32
            h_ds = max(1, refiner_out_size[0] // self.hdr_refiner_downsample)
            w_ds = max(1, refiner_out_size[1] // self.hdr_refiner_downsample)
            refiner_size = (
                int(math.ceil(h_ds / refiner_multiple) * refiner_multiple),
                int(math.ceil(w_ds / refiner_multiple) * refiner_multiple),
            )

        HDR_init_for_refiner = _resize_spatial(HDR_init_final.detach(), refiner_size)
        ldr_for_refiner_in = _resize_spatial(ldr_for_refiner, refiner_size)
        lin_for_refiner_in = _resize_spatial(lin_for_refiner, refiner_size)
        valid_for_refiner_in = _resize_spatial(valid_for_refiner, refiner_size)
        depth_for_refiner_in = _resize_spatial(D_warp_full_saved, refiner_size)
        conf_for_refiner_in = _resize_spatial(geo_rel_full_saved, refiner_size)
        snr_for_refiner_in = _resize_spatial(snr_norm, refiner_size)
        ref_view_for_refiner = _resize_spatial(ref_view_lin, refiner_size)
        init_valid_for_refiner = _resize_spatial(hdr_init_valid.detach(), refiner_size)

        with torch.cuda.amp.autocast(enabled=False):
            refiner_kwargs = {
                "ldr_warped": ldr_for_refiner_in,
                "linear_warped": lin_for_refiner_in,
                "valid_mask": valid_for_refiner_in,
                "depth_warped": depth_for_refiner_in,
                "confidence": conf_for_refiner_in,
                "snr_norm": snr_for_refiner_in,
                "ref_view_rgb": ref_view_for_refiner,
            }
            if isinstance(self.hdr_refiner2, HDRGuidedCompositor):
                refiner_kwargs["init_valid_mask"] = init_valid_for_refiner
            HDR_refined = self.hdr_refiner2(HDR_init_for_refiner, **refiner_kwargs)
            if HDR_refined.shape[-2:] != refiner_out_size:
                B_ref, M_ref, C_ref = HDR_refined.shape[:3]
                HDR_refined = F.interpolate(
                    HDR_refined.reshape(B_ref * M_ref, C_ref, *HDR_refined.shape[-2:]),
                    size=refiner_out_size,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B_ref, M_ref, C_ref, *refiner_out_size)
        
        # Auxiliary outputs for loss computation and visualization
        aux_dict = {
            "C": C,  # [B,N,1,H_half,W_half] - predicted confidence
            "D_fused": D_fused.detach(),  # [B,M,1,H,W] - final depth after optional refiner
            "D_fused_raw": D_fused_raw_for_aux.detach(),  # [B,M,1,H,W] - pure weighted merge before refiner
            "D_warp": D_warp_full.detach(),  # [B,M,N,1,H_orig,W_orig] - for debugging
            "rerr": rerr.detach().squeeze(3),  # [B,M,N,H_tgt_h,W_tgt_h] - per-view reprojection error
            
            "hdr_init": HDR_init,  # [B,M,3,H_orig,W_orig] - for HDR loss,
            "ldr_warp": ldr_refined.detach(),  # after regrid/blend
            
            "w_over" : torch.stack(
                w_over_list, dim=1
            ),
            "w_under" : torch.stack(
                w_under_list, dim=1
            ),
            "w_combined": torch.stack(
                w_combined_list, dim=1
            ),  # [B,M,N,1,H,W] - trapezoid weights for visualization
            "w_effective": torch.stack(
                w_effective_list, dim=1
            ),  # [B,M,N,1,H,W] - final diffused HDR mixer evidence
            "hdr_init_valid": hdr_init_valid,  # [B,M,1,H,W] - reliable evidence mask for HDR_init
            "valid_warp_dfused": valid_warp_dfused.detach(),  # [B,M,1,H,W] - valid mask for D_fused
        }
        if depth_refiner_aux:
            aux_dict.update({k: v.detach() for k, v in depth_refiner_aux.items()})
            if "depth_refiner_gate" in depth_refiner_aux:
                aux_dict["depth_refiner_gate_train"] = depth_refiner_aux["depth_refiner_gate"]

        if hasattr(self.hdr_refiner2, "last_debug"):
            for k, v in getattr(self.hdr_refiner2, "last_debug", {}).items():
                if torch.is_tensor(v):
                    aux_dict[f"hdr_compositor_{k}"] = v.detach()
        
        # [FLOW] flow 모드에서는 D_0을 반환하지 않음 (depth 정보 없음)
        if self.mode != "flow":
            aux_dict["D_0"] = D0_full  # [B,N,1,H_orig,W_orig] - backbone depth
            # Per-target view-normalized depth fusion weights (sums to 1 across N).
            # Exposed (detached) for monitoring and optional consistency loss.
            # We detach to keep ConfidenceHead's gradient flowing only through
            # D_fused→depth_loss; routing it through 1/sum(w_base) backward is
            # numerically unstable in regions where all source views have
            # near-zero w_base.
            aux_dict["w_base_norm"] = w_base_norm_half.detach()  # [B,M,N,1,H_tgt_h,W_tgt_h]
            aux_dict["w_base_norm_train"] = w_base_norm_half
            if depth_anchor_aux:
                aux_dict.update(depth_anchor_aux)
            if depth_anchor_posterior_half is not None:
                aux_dict["depth_anchor_posterior"] = depth_anchor_posterior_half.detach()
                aux_dict["depth_anchor_posterior_train"] = depth_anchor_posterior_half
            if anchor_geometry_compat_half is not None:
                aux_dict["anchor_geometry_compat"] = anchor_geometry_compat_half.detach()
        else:
            aux_dict["flow"] = torch.stack(flow_list, dim=1).detach()  # [B,2,H,W] - estimated flow
        
        # align_net 사용 시 추가 정보
        if self.use_align_net and self.mode != "flow":
            aux_dict["ldr_warp_aligned"] = ldr_refined.detach()  # [B,M,N,3,H_orig,W_orig]
            aux_dict["lin_warp_aligned"] = lin_refined  # [B,M,N,3,H_orig,W_orig]

        return D_fused, HDR_refined, aux_dict


import time

if __name__ == "__main__":
    # 간단한 테스트
    B, N, H, W = 1, 6, 256, 320
    rgbs = torch.rand(B, N, 6, H, W)
    prompts = torch.rand(B, N, 1, H, W)
    sats = torch.rand(B, N, 1, H, W) * 0.5
    Ks = torch.eye(3).view(1, 1, 3, 3).expand(B, N, -1, -1)
    Kinvs = torch.linalg.inv(Ks.float()).contiguous()  # Pre-compute K_inv
    Ts = torch.eye(4).view(1, 1, 4, 4).expand(B, N, -1, -1).clone()
    Ts[:, :, 0, 3] = (
        torch.linspace(-0.1, 0.1, N).clone().view(1, N).clone()
    )  # x축으로 약간 이동
    M= 1
    Ktargs = torch.eye(3).view(1, 1, 3, 3).expand(B, M, -1, -1)
    Ktargs_inv = torch.linalg.inv(Ktargs.float()).contiguous()
    Ttargs = torch.eye(4).view(1, 1, 4, 4).expand(B, M, -1, -1).clone()
    ldr_min = torch.rand(B, N) * 0.01 + 0.005  # 약간의 노이즈 레벨

    model = PromptDA_MV_All(mode="rs", hdr_refiner_v4=False, hdr_refiner_afunet=False, refiner_n_tf = 4).cuda()
    
    ref_lin = torch.rand(B, M, 3, H, W).cuda()
    
    time_begin = time.time()
    with torch.cuda.amp.autocast(enabled=False):
        D_out, D_HDR, aux = model(
            rgbs.cuda(),
            prompts.cuda(),
            sats.cuda(),
            Ks.cuda(),
            Kinvs.cuda(),
            Ts.cuda(),
            Ks_tgt=Ktargs.cuda(),
            Ts_tgt=Ttargs.cuda(),
            ldr_min=ldr_min.cuda(),
            shape_tgt = (H, W),
            ref_view_lin = ref_lin,
        )
    #print(aux["flow"].shape)
    total_time = time.time() - time_begin

    print("D_out:", D_out.shape)
    print("D_HDR:", D_HDR.shape)
    print("Total inference time:", total_time)
