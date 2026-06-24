import torch
import torch.nn as nn
import torch.nn.functional as F
# ====== PartialConv + Depthwise 경량 CNN ======
class PartialConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.register_buffer("ones", torch.ones(1, 1, k, k))
        self.k, self.s, self.p = k, s, p

    def forward(self, x, m):
        with torch.no_grad():
            valid = F.conv2d(m, self.ones, stride=self.s, padding=self.p).clamp(min=1.0)
            scale = (self.k * self.k) / valid
        y = self.conv(x * m) * scale
        m_out = (valid > 0).float()
        return y, m_out


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1, 1, 0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class PromptUpsampleLite(nn.Module):
    def __init__(
        self, use_rgb=True, base=8, residual_scale=0.5
    ):  # NEW: residual scaling
        super().__init__()
        # Input channels: depth(1) + mask(1) + rgb(3) + far_hint(1) = 6 or 7
        in_ch = 1 + 1 + (3 if use_rgb else 0) + 1  # + far_hint
        self.use_rgb = use_rgb
        self.residual_scale = (
            residual_scale  # Limit residual magnitude (0.5 = ±50% of input)
        )

        # Deeper network with MUCH LARGER receptive field
        self.pconv1 = PartialConv2d(
            in_ch, base, k=7, s=1, p=3, bias=False
        )  # Larger kernel: 5->7
        self.dw1 = DWConv(base)
        self.dw2 = DWConv(base)
        self.dw3 = DWConv(base)
        self.dw4 = DWConv(base)

        # Multi-scale dilated convolutions for LARGE receptive field
        self.dilated2 = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, padding=2, dilation=2, groups=base),
            nn.ReLU(True),
        )
        self.dilated4 = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, padding=4, dilation=4, groups=base),
            nn.ReLU(True),
        )
        self.dilated8 = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, padding=8, dilation=8, groups=base),
            nn.ReLU(True),
        )

        # Fusion layer
        self.fusion = nn.Conv2d(base * 4, base, 1, 1, 0)  # 4 scales

        self.out = nn.Conv2d(base, 1, 3, 1, 1)

    def forward(self, d, m, rgb=None, far_hint=None):
        """
        d: depth [B,1,H,W]
        m: mask [B,1,H,W]
        rgb: [B,3,H,W] or None
        far_hint: [B,1,H,W] far region indicator (1=far, 0=near)
        """
        # Build input: depth + mask + rgb + far_hint
        inputs = [d, m]
        if self.use_rgb and rgb is not None:
            inputs.append(rgb)
        if far_hint is not None:
            inputs.append(far_hint)
        x = torch.cat(inputs, dim=1)

        f, m_out = self.pconv1(x, m)
        f = self.dw1(f)
        f = self.dw2(f)
        f = self.dw3(f)
        f = self.dw4(f)

        # Multi-scale processing
        f_d2 = self.dilated2(f)
        f_d4 = self.dilated4(f)
        f_d8 = self.dilated8(f)

        # Concatenate all scales
        f_multi = torch.cat([f, f_d2, f_d4, f_d8], dim=1)  # [B, 32, H, W]
        f_fused = self.fusion(f_multi)  # [B, 8, H, W]

        # Output as SCALED RESIDUAL correction
        # Scale limits prevent extreme changes (100m→0) while allowing learning
        # residual_scale=0.5 means: 100m input → can predict 50-150m range
        residual = self.out(f_fused) * self.residual_scale
        dense = d + residual  # Residual: CNN learns bounded correction
        return dense


# ====== Iterative Hole Filling Helper ======
def iterative_fill_holes(
    depth,
    mask,
    rgb=None,
    iterations=5,
    kernel_size=15,
    max_depth=100,
    depth_scale=None,
):
    """
    반복적으로 빈 영역을 주변 값으로 채워나감

    KEY STRATEGY:
    1. Conservative far region detection (only VERY confident cases)
    2. Use surrounding depth values to infer hole depth
    3. MAX propagation for sparse regions to avoid averaging down
    4. **NEW**: Support for normalized depth (0~1 range) via depth_scale

    Args:
        depth: [B,1,H,W] - can be in meters OR normalized (0~1)
        mask: [B,1,H,W] (1=valid)
        rgb: [B,3,H,W] or None
        iterations: number of filling iterations
        kernel_size: convolution kernel size
        max_depth: far region initialization value in METERS (default 1000m)
        depth_scale: if provided, depth is assumed normalized (depth = real_depth / depth_scale)
                     and max_depth will be scaled accordingly
    """
    B, _, H, W = depth.shape
    current = depth.clone()
    current_mask = mask.clone()

    # Handle normalized depth: scale max_depth to normalized range
    if depth_scale is not None:
        # depth_scale: [B,1,1,1] or scalar
        max_depth_normalized = max_depth / depth_scale  # Convert 1000m to normalized
        far_init_value = max_depth_normalized
        nearby_far_threshold = 16.0 / depth_scale  # 300m in normalized space
    else:
        far_init_value = max_depth
        nearby_far_threshold = 16.0

    # Vertical position map (0=top, 1=bottom)
    y_coords = (
        torch.linspace(0, 1, H, device=depth.device).view(1, 1, H, 1).expand(B, 1, H, W)
    )

    # Identify initial holes
    initial_holes = (current_mask < 0.5).float()

    # ========== CONSERVATIVE FAR REGION INITIALIZATION ==========
    # Strategy: Only initialize as "far" if VERY confident
    # Otherwise, let iterative filling propagate from nearby valid values

    # Multi-scale sparsity detection
    local_density_small = F.avg_pool2d(
        current_mask, kernel_size=15, stride=1, padding=7
    )
    local_density_large = F.avg_pool2d(
        current_mask, kernel_size=31, stride=1, padding=15
    )

    # Check if there are nearby valid depth values (large window)
    nearby_depth_exists = F.avg_pool2d(
        current_mask, kernel_size=51, stride=1, padding=25
    )

    # Compute average depth of nearby valid pixels (to infer context)
    nearby_depth_sum = F.avg_pool2d(
        current * current_mask, kernel_size=51, stride=1, padding=25
    )
    nearby_depth_avg = nearby_depth_sum / (nearby_depth_exists.clamp_min(1e-6))

    # Far region criteria (CONSERVATIVE):
    # Only mark as far if we have strong evidence

    # 1. Upper region (conservative threshold)
    is_upper = (y_coords < 0.4).float()  # Only upper 40%

    # 2. Extremely sparse (almost no nearby points)
    is_extremely_sparse = (local_density_small < 0.02).float()
    is_large_sparse = (local_density_large < 0.03).float()
    has_no_nearby = (nearby_depth_exists < 0.05).float()

    # 3. Nearby context suggests far (average depth > threshold where points exist)
    # Use the scaled threshold for normalized depth
    nearby_is_far = (nearby_depth_avg > nearby_far_threshold).float() * (
        nearby_depth_exists > 0.01
    ).float()

    # Combine: Only far if VERY confident
    # Case 1: Upper + extremely sparse + no nearby points
    confident_far_geometric = is_upper * is_extremely_sparse * has_no_nearby

    # Case 2: Nearby points are clearly far (> threshold average)
    confident_far_contextual = nearby_is_far

    # Final far region: either geometric or contextual evidence
    far_region_hint = initial_holes * torch.maximum(
        confident_far_geometric, confident_far_contextual
    )

    # Initialize far regions with far_init_value (scaled appropriately)
    current = torch.where(
        far_region_hint.bool(),
        far_init_value,  # This is already scaled if depth_scale was provided
        current,
    )
    current_mask = torch.clamp(current_mask + far_region_hint, 0, 1)

    # ========== ITERATIVE FILLING WITH MAX PROPAGATION ==========
    pad = kernel_size // 2

    for i in range(iterations):
        holes = (current_mask < 0.5).float()

        if holes.sum() < 1:
            break  # All filled

        # Compute local max depth around each pixel (for LiDAR wide gaps)
        # Use max pooling to avoid averaging down to lower values
        max_neighbors = F.max_pool2d(
            current * current_mask + (1 - current_mask) * (-1e9),  # mask out holes
            kernel_size,
            stride=1,
            padding=pad,
        )

        # Also compute normalized average (traditional method)
        num = F.avg_pool2d(current * current_mask, kernel_size, stride=1, padding=pad)
        den = F.avg_pool2d(current_mask, kernel_size, stride=1, padding=pad).clamp_min(
            1e-6
        )
        avg_neighbors = num / den

        # Adaptive blending: use MAX for sparse regions, AVG for dense regions
        # Sparse regions (wide LiDAR gaps) should take MAX to avoid lowering
        # Dense regions can use AVG for smoothing
        neighbor_density = den  # This is essentially local density (0-1)

        # Smooth transition: sparse (< 0.15) -> MAX, dense (> 0.35) -> AVG
        max_weight = 1.0 - smoothstep(
            0.15, 0.35, neighbor_density
        )  # 1.0 at very sparse, 0.0 at dense
        avg_weight = 1.0 - max_weight

        filled = max_weight * max_neighbors + avg_weight * avg_neighbors

        # Update holes
        current = current * current_mask + filled * holes

        # Expand mask: any pixel with valid neighbors becomes valid
        current_mask = torch.clamp(current_mask + (den > 0.005).float(), 0, 1)

        # Expand mask: any pixel with valid neighbors becomes valid
        current_mask = torch.clamp(current_mask + (den > 0.01).float(), 0, 1)

    return current


# ====== Enhanced Hybrid Densifier with Iterative Filling ======
class HybridDensifier(nn.Module):
    """
    지역 밀도 기반 하이브리드 보간기 + 반복적 hole filling:
      - high density: identity(+tiny median)
      - mid density : guided filter / normalized box
      - low density : lightweight CNN (PartialConv+DWConv)
      - post-process: iterative hole filling for remaining zeros
    """

    def __init__(
        self,
        k_density=9,  # local density window
        th_low=0.05,  # extremely-sparse 경계
        th_high=0.40,  # dense 경계
        use_rgb=True,
        guided_r=4,
        guided_eps=1e-3,
        enable_iterative_fill=True,  # NEW: toggle iterative filling
        fill_iterations=5,  # NEW: number of iterations
        fill_kernel_size=15,  # NEW: kernel size for hole filling,
        max_depth=100,
    ):
        super().__init__()
        self.k_density = k_density
        self.th_low = th_low
        self.th_high = th_high
        self.use_rgb = use_rgb
        self.guided_r = guided_r
        self.guided_eps = guided_eps
        self.enable_iterative_fill = enable_iterative_fill
        self.fill_iterations = fill_iterations
        self.fill_kernel_size = fill_kernel_size
        self.cnn = PromptUpsampleLite(
            use_rgb=use_rgb, base=8
        )  # Increased base: 4->8 for more capacity
        self.max_depth = max_depth

    def forward(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor = None,
        depth_scale: torch.Tensor = None,
    ):
        """
        depth: [B,1,H,W] (0=invalid), rgb: [B,3,H,W] or None
        depth_scale: [B,1,1,1] or None - scale factor for normalized depth
        Returns: fully dense depth map (no zeros if iterative filling enabled)

        NEW ORDER:
        1. Fill holes FIRST on raw sparse input (clean initialization)
        2. THEN apply density-based blending (guided/CNN refinement)
        """
        B, _, H, W = depth.shape
        mask = (depth > 0).float()

        # 1) FIRST: Iterative hole filling on RAW sparse input
        # This gives clean initial values before any blending noise
        depth_prefilled = depth.clone()
        if self.enable_iterative_fill:
            depth_prefilled = iterative_fill_holes(
                depth,
                mask,
                rgb=rgb,
                iterations=self.fill_iterations,
                kernel_size=self.fill_kernel_size,
                max_depth=self.max_depth,  # Far region initialization (CARLA typical max)
                depth_scale=depth_scale,  # Pass scale for normalized depth
            )
        if depth_scale is None:
            depth_scale = 1

        # Now depth_prefilled is FULLY DENSE with clean far region initialization
        # Use this as input for all branches
        mask_prefilled = (depth_prefilled > 0).float()

        # 2) local density (0~1) - computed on ORIGINAL mask
        dens = box_count(mask, k=self.k_density)  # [B,1,H,W]

        # 3) Compute far region hint for CNN: upper region + sparse regions
        upper_region = torch.zeros_like(mask)
        upper_region[:, :, : H // 2, :] = 1.0
        sparse_region = (dens < 0.1).float()
        far_hint = torch.clamp(upper_region + sparse_region, 0, 1)

        # 4) Three branch outputs - ALL use prefilled depth
        # 4-1 High density: keep with tiny median denoise
        out_hi = tiny_median(depth_prefilled)

        # 4-2 Mid density: Guided Filter (edge-aware, fast) with LARGER radius
        if rgb is None:
            num = F.avg_pool2d(depth_prefilled * mask_prefilled, 11, 1, 5)
            den = F.avg_pool2d(mask_prefilled, 11, 1, 5).clamp_min(1e-6)
            out_mid = num / den
        else:
            out_mid = guided_filter_fast(rgb, depth_prefilled, r=8, eps=self.guided_eps)

        # 4-3 Low density: lightweight CNN with far hint
        out_lo = self.cnn(depth_prefilled, mask_prefilled, rgb, far_hint)

        # 5) soft gates (smoothstep) - based on ORIGINAL sparsity
        g_hi = smoothstep(self.th_high, 1.0, dens)
        g_lo = 1.0 - smoothstep(0.0, self.th_low, dens)
        g_mid_raw = 1.0 - g_hi - g_lo
        g_mid = torch.clamp(g_mid_raw, 0.0, 1.0)

        # epsilon gating to keep all branches active - STRONGER minimum
        eps = 0.15  # Increased from 0.03 to prevent gate collapse
        g_hi = g_hi * (1 - 3 * eps) + eps
        g_mid = g_mid * (1 - 3 * eps) + eps
        g_lo = g_lo * (1 - 3 * eps) + eps

        # normalize to sum=1
        g_sum = (g_hi + g_mid + g_lo).clamp_min(1e-6)
        g_hi, g_mid, g_lo = g_hi / g_sum, g_mid / g_sum, g_lo / g_sum

        # 6) blend - final output
        dense = g_hi * out_hi + g_mid * out_mid + g_lo * out_lo

        # 7) SOFT anchoring to original sparse values (not hard replacement!)
        # Alpha blend instead of hard override to preserve densifier's work
        alpha_anchor = 0.7  # Trust original measurements 70%, allow 30% correction
        dense = torch.where(
            mask.bool(), alpha_anchor * depth + (1 - alpha_anchor) * dense, dense
        )

        # 8) ADAPTIVE anchoring for far holes: stronger anchor for higher/sparser regions
        # This allows CNN to learn corrections while preventing complete collapse
        prefilled_holes = (mask < 0.5).float() * (depth_prefilled > 0).float()

        # Adaptive anchor weight based on:
        # 1. How far the prefilled value is (higher = more uncertain)
        # 2. How sparse the region is (sparser = less confident CNN should be)
        far_ratio = (depth_prefilled / (self.max_depth + 1e-6)).clamp(0, 1)  # 0-1
        sparsity = 1.0 - dens  # 0-1, higher = sparser

        # Combine: far + sparse → higher anchor weight (trust initialization more)
        # Base anchor: 0.1 (10% minimum anchor)
        # Max anchor: 0.5 (50% for very far + very sparse)
        anchor_weight = 0.1 + 0.4 * far_ratio * sparsity

        # Apply adaptive blending only to holes
        dense = torch.where(
            prefilled_holes.bool(),
            anchor_weight * depth_prefilled + (1 - anchor_weight) * dense,
            dense,
        )

        # Return both final and intermediate for debugging
        if isinstance(depth_scale, torch.Tensor):
            max_depth_scaled = self.max_depth / depth_scale  # [B,1,1,1]
            dense = dense.clamp(depth_scale * 0, max_depth_scaled)
        else:
            dense = dense.clamp(0, self.max_depth)

        if self.enable_iterative_fill:
            return dense, depth_prefilled  # final, intermediate_filled
        return dense
