
# ============================================================
# RESIDUAL SUB-PIXEL ALIGNER (RSA)
# Lightweight, learning-free alignment refinement for HDR fusion
# ============================================================


def box_sum(x, k):
    """Fast window sum using avg_pool2d (O(1) per window)"""
    padding = k // 2
    return F.avg_pool2d(x, k, stride=1, padding=padding) * (k * k)


def flow_to_grid(dx, dy, H, W, device):
    """Convert pixel flow (dx, dy) to grid_sample grid [-1, 1]"""
    # Create normalized coordinate grids
    xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
    ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)

    # Broadcast to match batch/channel dims
    B = dx.size(0)
    xs = xs.expand(B, 1, H, W)
    ys = ys.expand(B, 1, H, W)

    # Apply flow: px → [-1,1] scale
    gx = xs + (dx * 2.0 / max(W - 1, 1))
    gy = ys + (dy * 2.0 / max(H - 1, 1))

    return torch.cat([gx, gy], dim=1).permute(0, 2, 3, 1)  # [B,H,W,2]


def local_cost_sad(I_t, I_s, k=7, r=2, mask=None):
    """
    Exhaustive local search (Winner-Takes-All) using SAD cost.

    Args:
        I_t: [B,1,H,W] - Target grayscale image
        I_s: [B,1,H,W] - Source grayscale image (already warped by depth)
        k: Window size for SAD computation
        r: Search radius in pixels (searches [-r, r]^2)
        mask: [B,1,H,W] - Valid region mask (optional)

    Returns:
        best_dx, best_dy: [B,1,H,W] - Best integer shift at each pixel
        best_cost: [B,1,H,W] - Minimum SAD cost
    """
    B, _, H, W = I_t.shape
    device = I_t.device

    best_cost = None
    best_dx = torch.zeros(B, 1, H, W, device=device)
    best_dy = torch.zeros_like(best_dx)

    # Exhaustive search over [-r, r]^2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Create shift grid
            shift_grid = flow_to_grid(
                torch.full((B, 1, H, W), dx, device=device, dtype=torch.float32),
                torch.full((B, 1, H, W), dy, device=device, dtype=torch.float32),
                H,
                W,
                device,
            )

            # Warp source by shift
            shifted = F.grid_sample(
                I_s,
                shift_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            # Compute SAD in window
            diff = (I_t - shifted).abs()
            if mask is not None:
                diff = diff * mask
            cost = box_sum(diff, k)

            # Update best
            if best_cost is None:
                best_cost = cost
                best_dx.fill_(dx)
                best_dy.fill_(dy)
            else:
                update_mask = cost < best_cost
                best_cost = torch.where(update_mask, cost, best_cost)
                best_dx = torch.where(
                    update_mask, torch.full_like(best_dx, dx), best_dx
                )
                best_dy = torch.where(
                    update_mask, torch.full_like(best_dy, dy), best_dy
                )

    return best_dx, best_dy, best_cost


def iclk_refine(I_t, I_s, init_dx, init_dy, iters=1, k=7, mask=None):
    """
    Inverse Compositional Lucas-Kanade refinement (1-2 iterations).
    Refines integer shift to sub-pixel precision (±0.5~1px).

    Args:
        I_t, I_s: [B,1,H,W] - Target and source grayscale
        init_dx, init_dy: [B,1,H,W] - Initial integer shift from WTA
        iters: Number of Gauss-Newton iterations (1-2 is enough)
        k: Window size for gradient/residual accumulation
        mask: [B,1,H,W] - Valid region mask

    Returns:
        dx, dy: [B,1,H,W] - Refined sub-pixel flow
    """
    B, _, H, W = I_t.shape
    device = I_t.device

    # Compute image gradients (on target, for IC formulation)
    gx = torch.zeros_like(I_t)
    gy = torch.zeros_like(I_t)
    gx[:, :, :, :-1] = I_t[:, :, :, 1:] - I_t[:, :, :, :-1]
    gy[:, :, :-1, :] = I_t[:, :, 1:, :] - I_t[:, :, :-1, :]

    # Compute windowed Hessian: J^T J = [[ΣIx², ΣIxIy], [ΣIxIy, ΣIy²]]
    Ix2 = box_sum(gx * gx, k)
    Iy2 = box_sum(gy * gy, k)
    Ixy = box_sum(gx * gy, k)
    det = (Ix2 * Iy2 - Ixy * Ixy).clamp_min(1e-6)

    # Initialize flow
    dx = init_dx.clone().float()
    dy = init_dy.clone().float()

    # Gauss-Newton iterations
    for _ in range(iters):
        # Warp source by current flow
        grid = flow_to_grid(dx, dy, H, W, device)
        I_s_warp = F.grid_sample(
            I_s, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        # Compute residual
        residual = I_t - I_s_warp
        if mask is not None:
            residual = residual * mask

        # J^T r = [ΣIx·r, ΣIy·r]
        Ixr = box_sum(gx * residual, k)
        Iyr = box_sum(gy * residual, k)

        # Solve 2x2: δp = (J^T J)^-1 (J^T r)
        upd_dx = (Iy2 * Ixr - Ixy * Iyr) / det
        upd_dy = (-Ixy * Ixr + Ix2 * Iyr) / det

        # Update flow (compositional: p ← p + δp)
        dx = (dx + upd_dx).clamp(-3, 3)
        dy = (dy + upd_dy).clamp(-3, 3)

    return dx, dy


def residual_align_one_pair(I_t, I_s_warp, valid, r=2, k=7, stride=4, iclk_iters=1):
    """
    Complete residual alignment pipeline for one (target, source) pair.

    Pipeline:
    1. Sparse WTA search on stride grid
    2. IC-LK sub-pixel refinement (1-2 iterations)
    3. Guided upsampling to dense flow

    Args:
        I_t: [B,1,H,W] - Target grayscale LDR
        I_s_warp: [B,1,H,W] - Source grayscale LDR (already depth-warped)
        valid: [B,1,H,W] - Valid region mask (geo + boundary)
        r: Search radius (2-3 px)
        k: Window size for matching (7-9)
        stride: Sparse sampling stride (4 = process 1/16 pixels)
        iclk_iters: IC-LK iterations (1-2)

    Returns:
        dx_dense, dy_dense: [B,1,H,W] - Dense residual flow in pixels
    """
    B, _, H, W = I_t.shape
    device = I_t.device

    # 0) Create sparse sampling mask
    sparse_mask = torch.zeros_like(valid)
    sparse_mask[:, :, ::stride, ::stride] = 1.0
    valid_sparse = (valid * sparse_mask) > 0.5

    # 1) Exhaustive local search (WTA)
    dx_int, dy_int, _ = local_cost_sad(I_t, I_s_warp, k=k, r=r, mask=valid)

    # 2) IC-LK sub-pixel refinement
    dx_refined, dy_refined = iclk_refine(
        I_t, I_s_warp, dx_int, dy_int, iters=iclk_iters, k=k, mask=valid
    )

    # 3) Sparse → Dense interpolation (edge-aware guided upsample)
    # Keep only sparse samples, zero out others
    dx_sparse = dx_refined * valid_sparse.float()
    dy_sparse = dy_refined * valid_sparse.float()

    # Simple guided upsample: avg pooling with normalization
    kernel_size = stride * 2 + 1
    padding = kernel_size // 2

    # Numerator: sum of sparse flows
    dx_sum = F.avg_pool2d(dx_sparse, kernel_size, stride=1, padding=padding) * (
        kernel_size**2
    )
    dy_sum = F.avg_pool2d(dy_sparse, kernel_size, stride=1, padding=padding) * (
        kernel_size**2
    )

    # Denominator: count of valid sparse samples
    count = F.avg_pool2d(
        valid_sparse.float(), kernel_size, stride=1, padding=padding
    ) * (kernel_size**2)
    count = count.clamp_min(1e-6)

    # Normalize
    dx_dense = (dx_sum / count) * valid
    dy_dense = (dy_sum / count) * valid

    # Final clamp to prevent extreme flows
    dx_dense = dx_dense.clamp(-3, 3)
    dy_dense = dy_dense.clamp(-3, 3)

    return dx_dense, dy_dense
