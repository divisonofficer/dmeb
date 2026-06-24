import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# =========================
# Global caches (device-aware)
# =========================
_OFFSETS_CACHE = {}  # (r_max, device_index) -> (off[K,2], dist2[K])

def _get_circle_offsets(r_max: int, device, dtype=torch.int32):
    key = (r_max, device.index if (device.type == "cuda") else -1)
    if key not in _OFFSETS_CACHE:
        offs = []
        for dy in range(-r_max, r_max+1):
            for dx in range(-r_max, r_max+1):
                if dx*dx + dy*dy <= r_max*r_max:
                    offs.append((dy, dx))
        off = torch.tensor(offs, device=device, dtype=dtype)            # [K,2]
        dist2 = (off.to(torch.float32)**2).sum(dim=1).contiguous()      # [K]
        _OFFSETS_CACHE[key] = (off, dist2)
    return _OFFSETS_CACHE[key]


# =========================
# Helpers
# =========================
@torch.no_grad()
def _minpool(depth: torch.Tensor, r: int) -> torch.Tensor:
    """depth: [H,W], meters. r=0이면 그대로."""
    if r <= 0:
        return depth
    x = depth
    # inf/NaN -> 큰 값으로 치환 후 min-pool
    bad = ~torch.isfinite(x)
    if bad.any():
        x = x.clone()
        x[bad] = torch.finfo(x.dtype).max / 4
    y = -F.max_pool2d((-x)[None, None, ...], kernel_size=2*r+1, stride=1, padding=r)[0, 0]
    return y


# =========================
# Main (optimized)
# =========================
@torch.no_grad()
def occlusion_aware_splat(
    pts_cam: torch.Tensor,          # [N,3], (X,Y,Z>0) in camera coords, meters
    K: torch.Tensor,                # [3,3] intrinsics
    H: int,
    W: int,
    delta_theta: float,             # rad, e.g., 2*pi/1024

    *,  # --- Stage-1 splat (기본 raster) ---
    k1: float = 0.8,
    k2: float = 1e-4,
    r_min: int = 4,
    r_max: int = 16,
    use_inverse_depth: bool = True,
    chunk_points: int = 200_000,    # chunked splatting to limit memory peak

    # --- Single-frame guard-rail (Stage-0 filtering) ---
    single_frame_depth: Optional[torch.Tensor] = None,   # [H,W], meters (0/<=0 = invalid)
    sf_pyramid: Optional[Tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = None,  # precomputed min-pooled pyramid
    
    tau_front: float = 0.03,         # m, front check (작게)
    tau0: float = 0.010,             # m, base occlusion margin
    tau1: float = 0.30,              # m/m, z 비례 여유(원거리 완화)
    pyramid_radii: Tuple[int,int,int] = (1,2,3),  # 3x3, 5x5, 7x7
    z_breaks: Tuple[float,float] = (10.0, 30.0),  # m

    # --- Stage-2 adaptive clean-up (넉넉한 반경으로 근거리 우선 가림) ---
    r_clean: int = 3,                # min-pool 반경(픽셀) for clean-up
    tau_clean0: float = 0.02,        # m
    tau_clean1: float = 0.02,        # m/m (z 비례)
    adaptive_cleanup: bool = True,   # density/gap/z adaptive 반경 사용

    # --- 품질 게이팅(옵션) ---
    support_count: Optional[torch.Tensor] = None,  # [N]
    support_min: int = 0,
    reproj_error_px: Optional[torch.Tensor] = None, # [N]
    reproj_max_px: float = float("inf"),

    return_inv_depth: bool = False,  # True면 최종 inv-depth(미관측=0) 반환
) -> torch.Tensor:
    """
    파이프라인(optimized):
      Stage-0) single-frame 기반 1차 필터(front/occl, depth-adaptive)  [grid_sample 1회로 병합]
      Stage-1) adaptive splatting (chunked, offsets cache, scatter_reduce) → depth_map
      Stage-1.5) single-frame depth 덮어쓰기(보존) → refined는 single-frame 포함 보장
      Stage-2) adaptive asymmetric clean-up (near 중심, density/gap/z 반경, variance-aware margin)
      반환: invalid=0 (depth) / 0 (inv-depth)
    """
    device = pts_cam.device
    dtype  = pts_cam.dtype

    # --- Intrinsics ---
    K = K.to(device=device, dtype=dtype)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # --- 입력 유효성 & 게이팅 ---
    X, Y, Z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    valid = Z > 0
    if support_count is not None:
        valid = valid & (support_count.to(device) >= int(support_min))
    if reproj_error_px is not None:
        valid = valid & (reproj_error_px.to(device, dtype=dtype) <= reproj_max_px)
    if not torch.any(valid):
        print("No valid points after gating.")
        return torch.zeros((H, W), device=device, dtype=dtype)

    X, Y, Z = X[valid], Y[valid], Z[valid]

    # --- Project (u,v) ---
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    margin = float(r_max) + 1.0
    in_img = (u >= -margin) & (u <= (W - 1 + margin)) & (v >= -margin) & (v <= (H - 1 + margin))
    u, v, Z = u[in_img], v[in_img], Z[in_img]
    if u.numel() == 0:
        print("No points project inside image bounds.")
        return torch.zeros((H, W), device=device, dtype=dtype)

    # =========================
    # Stage-0: Single-frame 기반 1차 필터 (grid_sample 1회)
    # =========================
    if single_frame_depth is not None:
        if sf_pyramid is None:
            Dsf = single_frame_depth.to(device=device, dtype=dtype)
            bad = ~(Dsf > 0)
            if bad.any():
                Dsf = Dsf.clone()
                Dsf[bad] = torch.tensor(float('inf'), device=device, dtype=dtype)
            r1, r2, r3 = pyramid_radii
            D1 = _minpool(Dsf, r1) if r1 > 0 else Dsf
            D2 = _minpool(Dsf, r2) if r2 > 0 else Dsf
            D3 = _minpool(Dsf, r3) if r3 > 0 else Dsf
        else:
            # 이미 inf 처리/타입 통일된 pyramid가 들어온다고 가정
            D1, D2, D3 = sf_pyramid

        # Bilinear sample (stacked → single grid_sample)
        # 수정 (정상): [1,3,H,W]
        Dstack = torch.stack([D1, D2, D3], dim=0).unsqueeze(0)   # [1,3,H,W]
        x = (u / (W - 1)) * 2 - 1
        y = (v / (H - 1)) * 2 - 1
        grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
        D_uv = F.grid_sample(
            Dstack, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        ).squeeze(3).squeeze(1)  # [3,N]
        D1_uv, D2_uv, D3_uv = D_uv[0,0], D_uv[0,1], D_uv[0,2] 

        # z-adaptive blend
        b1, b2 = z_breaks
        w1 = torch.clamp((b1 - Z) / max(b1, 1e-6), 0.0, 1.0)
        w3 = torch.clamp((Z - b2) / max(1e-6, b2), 0.0, 1.0)
        w2 = 1.0 - w1 - w3
        Dmin_uv = w1*D1_uv + w2*D2_uv + w3*D3_uv

        finite = torch.isfinite(Dmin_uv)
        tau_occl = tau0 + tau1 * Z  # depth-adaptive occlusion margin

        front_reject = finite & (Z < (Dmin_uv - tau_front))
        occl_reject  = finite & (Z > (Dmin_uv + tau_occl))
        keep = ~(front_reject | occl_reject)

        u, v, Z = u[keep], v[keep], Z[keep]
        if u.numel() == 0:
            return torch.zeros((H, W), device=device, dtype=dtype), (D1,D2,D3)

    # =========================
    # Stage-1: Adaptive splatting (chunked, offsets cache)
    # =========================
    r0  = k1 * fx * float(delta_theta)
    r_px = torch.clamp(r0 + (k2 * Z), r_min, r_max)

    if use_inverse_depth:
        target = torch.zeros((H, W), device=device, dtype=dtype)   # invZ, amax
        reduce_op = "amax"
        pvals = 1.0 / Z
    else:
        target = torch.full((H, W), float('inf'), device=device, dtype=dtype)  # depth, amin
        reduce_op = "amin"
        pvals = Z

    # Offsets cache
    off, dist2 = _get_circle_offsets(r_max, device=target.device)
    off_dx = off[:, 1].view(1, -1)  # [1,K]
    off_dy = off[:, 0].view(1, -1)  # [1,K]
    Ksz = off.shape[0]
    flat = target.view(-1)

    # chunked splatting
    M = u.numel()
    CHUNK = int(chunk_points)
    for s in range(0, M, CHUNK):
        e  = min(s + CHUNK, M)
        u0 = u[s:e].view(-1, 1)
        v0 = v[s:e].view(-1, 1)
        rz = r_px[s:e].view(-1, 1)
        pv = pvals[s:e]  # [m]

        uu = torch.round(u0 + off_dx).long()  # [m,K]
        vv = torch.round(v0 + off_dy).long()
        valid_xy = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        within = dist2.view(1, -1) <= (rz.float()**2)
        mask = valid_xy & within
        if mask.any():
            lin  = (vv[mask] * W + uu[mask])
            vals = pv.view(-1, 1).expand(-1, Ksz)[mask]
            flat.scatter_reduce_(0, lin, vals, reduce=reduce_op, include_self=True)

    # depth map (임시, invalid=0)
    if use_inverse_depth:
        depth_map = torch.zeros_like(target)
        nz = target > 0
        depth_map[nz] = 1.0 / target[nz]
    else:
        depth_map = torch.zeros_like(target)
        finite = torch.isfinite(target) & (target > 0)
        depth_map[finite] = target[finite]

    # =========================
    # Stage-1.5: single-frame 덮어쓰기(보존)
    # =========================
    if single_frame_depth is not None:
        sf = single_frame_depth.to(device=device, dtype=dtype)
        sf_valid = sf > 0
        depth_map[sf_valid] = sf[sf_valid]

    # =========================
    # Stage-2: Adaptive, asymmetric clean-up (near 중심, density/gap/z 반경)
    # =========================
    if r_clean > 0:
        valid_map = depth_map > 0

        # 2.1 Near base (작은 반경)
        r_same = 2
        Dnear_base = _minpool(depth_map, r_same)
        near_valid = (Dnear_base > 0) & torch.isfinite(Dnear_base)

        # 2.2 Local density (near 주변 비슷한 깊이 밀도)
        r_den = 3
        k = 2 * r_den + 1
        eps0 = 0.02
        epsr = 0.02
        Zc = depth_map.clone()
        Zc[~valid_map] = 0
        Dref = Dnear_base
        tol = eps0 + epsr * Dref
        diff = torch.abs(Zc - Dref).clamp(min=0)
        bin_mask = (valid_map & near_valid) & (diff <= tol)
        den = F.avg_pool2d(bin_mask.float()[None, None, ...], k, stride=1, padding=r_den)[0, 0] * (k * k)
        den_norm = (den / float(k * k)).clamp(0, 1)

        # 2.3 Near-far gap
        gap = (depth_map - Dnear_base).clamp(min=0)
        rel_gap = (gap / torch.clamp(depth_map, min=1e-6)).clamp(min=0)
        gap_thr = 0.03
        gap_boost = (rel_gap > gap_thr).float()

        # 2.4 z-adaptive (멀수록 조금 반경↑)
        z_norm = (Dnear_base / 50.0).clamp(0, 1)

        # 2.5 반경 선택 (피라미드)
        r_set = [2, 3, 5, 7, 9]
        idx_base = torch.zeros_like(depth_map, dtype=torch.long)
        add_from_den = (den_norm * 2.0).round().clamp(0, len(r_set)-1).long()
        add_from_gap = (gap_boost * 2.0).long()
        add_from_z   = (z_norm * 1.0).round().clamp(0, len(r_set)-1).long()
        idx = (idx_base + add_from_den + add_from_gap + add_from_z).clamp(0, len(r_set)-1)

        Dpyr = torch.stack([_minpool(depth_map, r) for r in r_set], dim=0)  # [R,H,W]
        idx_flat = idx.view(1, -1)                         # [1,HW]
        Dpyr_flat = Dpyr.view(len(r_set), -1)              # [R,HW]
        Docc_flat = torch.gather(Dpyr_flat, 0, idx_flat)   # [1,HW]
        Docc = Docc_flat.view(depth_map.shape)             # [H,W]
        docc_valid = (Docc > 0) & torch.isfinite(Docc)

        # 2.6 variance-aware margin (평탄한 곳 과삭제 방지)
        r_var = max(2, r_den)
        k_var = 2 * r_var + 1
        sum_m = F.avg_pool2d(valid_map.float()[None, None, ...], k_var, stride=1, padding=r_var)[0, 0] * (k_var * k_var)
        sum_z = F.avg_pool2d(depth_map[None, None, ...],         k_var, stride=1, padding=r_var)[0, 0] * (k_var * k_var)
        mean = torch.zeros_like(depth_map)
        nzm = sum_m > 0
        mean[nzm] = sum_z[nzm] / sum_m[nzm]
        sum_z2 = F.avg_pool2d((depth_map**2)[None, None, ...],   k_var, stride=1, padding=r_var)[0, 0] * (k_var * k_var)
        mean2 = torch.zeros_like(depth_map)
        mean2[nzm] = sum_z2[nzm] / sum_m[nzm]
        var = torch.clamp(mean2 - mean**2, min=0.0)
        sigma = torch.sqrt(var)

        tau0_c = tau_clean0
        tau1_c = tau_clean1
        gamma  = 2.0
        tau_clean_map = tau0_c + tau1_c * Docc + gamma * sigma

        # single-frame near면 더 엄격
        if single_frame_depth is not None:
            sf = single_frame_depth.to(device=device, dtype=dtype)
            sf_mask = (sf > 0)
            tau_clean_map = torch.where(sf_mask, 0.7 * tau_clean_map, tau_clean_map)

        rho = 0.02  # relative margin
        rel_ok = (depth_map - Docc) / torch.clamp(depth_map, min=1e-6) > rho

        # 비대칭 제거: far만 제거
        remove = (valid_map & docc_valid) & (depth_map > (Docc + tau_clean_map)) & rel_ok
        depth_map[remove] = 0.0  # invalid = 0

    # =========================
    # Return (invalid=0)
    # =========================
    if return_inv_depth:
        inv = torch.zeros_like(depth_map)
        vmask = depth_map > 0
        inv[vmask] = 1.0 / depth_map[vmask]
        if single_frame_depth is not None:
            return inv, (D1, D2, D3)
        return inv
    if single_frame_depth is not None:
        return depth_map, (D1, D2, D3)
    return depth_map
