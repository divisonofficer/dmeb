import torch
import torch.nn.functional as F
from typing import Tuple
def safe_grid_sample(
    x: torch.Tensor,
    grid: torch.Tensor,
    t0: float = 0.01,
    t1: float = 0.05,
    valid_thr: float = 0.02,
    align_corners: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 0) grid sanitize (NaN/Inf 제거 + 유효범위 보정)
    grid = grid.contiguous()
    grid = torch.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    # 너무 멀리 나간 grid는 경계에 살짝 근접시켜 backward 안정화
    grid = grid.clamp(-1.0001, 1.0001)

    # 1) 두 모드 샘플
    y_lin  = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=align_corners)
    y_near = F.grid_sample(x, grid, mode="nearest",  padding_mode="zeros",
                           align_corners=align_corners)

    # 2) 거리/alpha는 **FP32에서 계산** (AMP라도 여기만 FP32 강제)
    with torch.cuda.amp.autocast(enabled=False):
        dist_x = (1.0 - grid[..., 0].abs().to(torch.float32))
        dist_y = (1.0 - grid[..., 1].abs().to(torch.float32))
        d = torch.minimum(dist_x, dist_y).unsqueeze(1)              # [B,1,H,W]
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        denom = max(float(t1 - t0), 1e-6)                            # 분모 안정화
        t = ((d - float(t0)) / denom).clamp(0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)                              # smoothstep
        alpha = alpha.to(y_lin.dtype)

    # 3) blend 전 y_lin/y_near sanitize (희귀 케이스 보호)
    y_lin  = torch.nan_to_num(y_lin)
    y_near = torch.nan_to_num(y_near)

    # 4) blend (alpha ∈ [0,1], dtype 일치)
    y = alpha * y_lin + (1 - alpha) * y_near

    # 5) inb/valid도 FP32에서 만들고 sanitize
    with torch.cuda.amp.autocast(enabled=False):
        inb = ((d - 0.0) / max(float(t1 - 0.0), 1e-6)).clamp(0.0, 1.0)
        inb = torch.nan_to_num(inb).to(y.dtype)
        valid = (d >= float(valid_thr)).to(y.dtype)

    return y, inb, valid


def warp_depth_safe(
    D_src: torch.Tensor,
    grid: torch.Tensor,
    z_tgt: torch.Tensor = None,
    z_tol_rel: float = 0.15,
    align_corners: bool = True,
    min_disparity: float = 0.01,  # NEW: min valid disparity (1/max_depth)
) -> Tuple[torch.Tensor, torch.Tensor]:
    # disparity 계산 전 sanitize
    D_src = torch.nan_to_num(D_src, nan=0.0, posinf=0.0, neginf=0.0)
    disparity_src = 1.0 / D_src.clamp_min(1e-6)

    disp_warp, inb, valid_edge = safe_grid_sample(disparity_src, grid, 
                                                  align_corners=align_corners)

    # 샘플 결과도 sanitize
    disp_warp = torch.nan_to_num(disp_warp, nan=0.0, posinf=0.0, neginf=0.0)

    # 역수 변환: invalid 영역은 **detached zero**로 대체 (NaN 경로 차단)
    cond = disp_warp >= float(min_disparity)
    inv = 1.0 / disp_warp.clamp_min(1e-6)
    inv = torch.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
    D_warp_valid = inv
    D_warp = torch.where(cond, D_warp_valid, torch.zeros_like(inv).detach())

    # valid mask
    valid = (D_warp > 0.05).to(D_warp.dtype) * valid_edge * cond.to(D_warp.dtype)

    if z_tgt is not None:
        z_tgt = torch.nan_to_num(z_tgt, nan=0.0, posinf=0.0, neginf=0.0)
        z_ok = ((D_warp - z_tgt).abs() / z_tgt.clamp_min(1e-3) <= float(z_tol_rel)).to(D_warp.dtype)
        valid = valid * z_ok

    return D_warp, valid


def warp_rgb_safe(
    I_src: torch.Tensor,
    grid: torch.Tensor,
    valid_like: torch.Tensor = None,
    align_corners: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safe RGB/feature warping with boundary handling.

    Args:
        I_src: [B,C,H,W] source image/features
        grid: [B,H,W,2] sampling grid
        valid_like: [B,1,H,W] optional additional validity mask (e.g., from depth)
        align_corners: align_corners flag

    Returns:
        Y: [B,C,H,W] warped image/features
        valid: [B,1,H,W] validity mask
    """
    Y, inb, valid_edge = safe_grid_sample(I_src, grid, align_corners=align_corners)
    valid = valid_edge

    # Intersect with depth validity if provided
    if valid_like is not None:
        valid = valid * (valid_like > 0.5).float()

    return Y, valid




def pad_div(x, div=14):
    # 14의 배수로 패딩 (PromptDA 요구사항)
    _, _, H, W = x.shape
    Hpad = (div - H % div) % div
    Wpad = (div - W % div) % div
    x = F.pad(x, (0, Wpad, 0, Hpad), mode="reflect")
    return x, Hpad, Wpad


def meshgrid_xy(B, H, W, device):
    ys, xs = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0).float()  # [3,H,W]
    return grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B,3,H,W]


def backproject_depth(depth, Kinv):
    B, _, H, W = depth.shape
    grid = meshgrid_xy(B, H, W, depth.device).reshape(B, 3, -1)  # [B,3,HW]
    cam = torch.bmm(Kinv, grid)  # [B,3,HW]
    X = (cam * depth.reshape(B, 1, -1)).reshape(B, 3, H, W)
    return X


def project_points(X, K, Ht, Wt):
    B, _, H_, W_ = X.shape
    with torch.cuda.amp.autocast(enabled=False):
        
        x = torch.bmm(K.float(), X.reshape(B, 3, -1).float())  # [B,3,HW]
    
        z = x[:, 2:3, :] + 1e-6
        u = x[:, 0:1, :] / z
        v = x[:, 1:2, :] / z
        # to [-1,1]  ← 여기서 타겟 크기(Ht, Wt)로 정규화
        u = (u / (W_ - 1) - 0.5) * 2.0
        v = (v / (H_ - 1) - 0.5) * 2.0
        grid = torch.stack([u, v], dim=-1).reshape(B, H_, W_, 2)
    grid_up = F.interpolate(
        grid.permute(0, 3, 1, 2),           # -> [B,2,H_,W_]
        size=(Ht, Wt),                      # ✅ 타겟 해상도
        mode="bilinear",
        align_corners=True,                 # ✅ grid_sample과 동일하게
    ).permute(0, 2, 3, 1)                   # -> [B,Ht,Wt,2]
    z_up = F.interpolate(
        z.reshape(B, 1, H_, W_),           # -> [B,1,H_,W_]
        size=(Ht, Wt),                      # ✅ 타겟 해상도
        mode="bilinear",
        align_corners=True,                 # ✅ grid_sample과 동일하게
    )                                       # -> [B,1,Ht,Wt]
    return grid_up, z_up


def same_view_mask_from_TK(
    T_tgt_from_src,
    Ks,
    Ks_tgt,
    rot_deg_thr=0.02,  # 0.02°
    trans_thr=1e-6,  # meters
    k_rel_thr=1e-6,
    k_abs_thr=1e-6,
):
    """
    Returns: same_view_mask [B,M,N,1,1,1] (bool)
    """
    B, M, N, _, _ = T_tgt_from_src.shape
    R = T_tgt_from_src[..., :3, :3]  # [B,M,N,3,3]
    t = T_tgt_from_src[..., :3, 3]  # [B,M,N,3]

    # rotation error (radians)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1, 1)
    ang = torch.acos(cos)  # [B,M,N]

    # translation error
    te = t.norm(dim=-1)  # [B,M,N]

    same_pose = (ang < (rot_deg_thr * 3.14159265 / 180.0)) & (te < trans_thr)

    # intrinsics check: broadcast Ks_tgt[B,M] vs Ks[B,N]
    Ks_exp = Ks.unsqueeze(1).expand(B, M, N, 3, 3)  # [B,M,N,3,3]
    Ks_tgt_ex = Ks_tgt.unsqueeze(2).expand(B, M, N, 3, 3)  # [B,M,N,3,3]
    diff = (Ks_exp - Ks_tgt_ex).abs()
    rel = diff / (Ks_tgt_ex.abs().clamp_min(1e-12))
    same_K = ((diff < k_abs_thr) | (rel < k_rel_thr)).all(dim=(-1, -2))

    same = (same_pose & same_K).reshape(B, M, N, 1, 1, 1)
    return same


@torch.no_grad()
def precompute_pairwise_T_tgt_from_src(T_w_from_cam_src, T_w_from_cam_tgt=None):
    """
    Generalized pairwise transform assembler.

    Args:
        T_w_from_cam_src: [B, N, 4, 4] world <- cam_src (sources)
        T_w_from_cam_tgt: [B, M, 4, 4] world <- cam_tgt (targets). If None,
                          targets are assumed equal to sources and N==M.

    Returns:
        T_tgt_from_src: [B, M, N, 4, 4] transform from source to target coordinate
                         (tgt <- src)

    Backward-compatible: when only a single argument is provided (old callers),
    this behaves exactly as before and returns [B, N, N, 4, 4].
    """
    # Single-argument compatibility
    if T_w_from_cam_tgt is None:
        T_w_from_cam = T_w_from_cam_src
        B, N, _, _ = T_w_from_cam.shape
        T = T_w_from_cam.to(torch.float32).contiguous()

        # decompose
        R_wc = T[..., :3, :3]  # [B,N,3,3]
        t_wc = T[..., :3, 3:4]  # [B,N,3,1]

        R_cw = R_wc.transpose(-1, -2).contiguous()
        t_cw = -torch.matmul(R_cw, t_wc)

        T_cam_from_w = torch.zeros(B, N, 4, 4, dtype=torch.float32, device=T.device)
        T_cam_from_w[..., :3, :3] = R_cw
        T_cam_from_w[..., :3, 3] = t_cw.squeeze(-1)
        T_cam_from_w[..., 3, 3] = 1.0
        T_cam_from_w = T_cam_from_w.contiguous()

        T_tgt_from_w = T_cam_from_w.unsqueeze(2)  # [B,N,1,4,4]
        T_w_from_src = T.unsqueeze(1)  # [B,1,N,4,4]
        T_tgt_from_src = torch.matmul(
            T_tgt_from_w, T_w_from_src
        ).contiguous()  # [B,N,N,4,4]
        return T_tgt_from_src

    # Two-argument path: src vs tgt
    T_src = T_w_from_cam_src.to(torch.float32).contiguous()
    T_tgt = T_w_from_cam_tgt.to(torch.float32).contiguous()
    B, N, _, _ = T_src.shape
    _, M, _, _ = T_tgt.shape

    # Decompose src
    R_wc_s = T_src[..., :3, :3]
    t_wc_s = T_src[..., :3, 3:4]

    # Decompose tgt
    R_wc_t = T_tgt[..., :3, :3]
    t_wc_t = T_tgt[..., :3, 3:4]

    # inverses for targets: T_cam_from_w (tgt)
    R_cw_t = R_wc_t.transpose(-1, -2).contiguous()  # [B,M,3,3]
    t_cw_t = -torch.matmul(R_cw_t, t_wc_t)  # [B,M,3,1]

    T_cam_from_w_t = torch.zeros(B, M, 4, 4, dtype=torch.float32, device=T_src.device)
    T_cam_from_w_t[..., :3, :3] = R_cw_t
    T_cam_from_w_t[..., :3, 3] = t_cw_t.squeeze(-1)
    T_cam_from_w_t[..., 3, 3] = 1.0
    T_cam_from_w_t = T_cam_from_w_t.contiguous()  # [B,M,4,4]

    # T_w_from_src is just T_src expanded
    T_w_from_src = T_src.unsqueeze(1)  # [B,1,N,4,4]

    # Compose: T_tgt_from_src = T_tgt_from_w @ T_w_from_src
    T_tgt_from_src = torch.matmul(
        T_cam_from_w_t.unsqueeze(2), T_w_from_src
    ).contiguous()
    # Result: [B,M,N,4,4]
    return T_tgt_from_src


def compose_world_to_target(X_src, T_tgt_from_src):
    # T_tgt_from_src is pre-computed to avoid lazy wrapper issues
    B, _, H, W = X_src.shape
    Xh = torch.cat([X_src, torch.ones(B, 1, H, W, device=X_src.device)], dim=1).reshape(
        B, 4, -1
    )
    Xh_t = torch.bmm(T_tgt_from_src, Xh)  # [B,4,HW]
    return Xh_t[:, :3, :].reshape(B, 3, H, W)


# def warp_depth_and_grid(
#     D_src_m: torch.Tensor,  # [Bnn,1,H,W]
#     Kinv_src_m: torch.Tensor,  # [Bnn,3,3]
#     T_tgt_from_src_m: torch.Tensor,  # [Bnn,4,4]  (tgt <- src)
#     K_tgt_m: torch.Tensor,  # [Bnn,3,3]
#     H: int,
#     W: int,
#     C=None,
# ):
#     """
#     깊이를 src->tgt로 워핑하면서, feature들도 동일 grid로 샘플링할 수 있도록 grid까지 반환.
#     Now uses safe_grid_sample to prevent boundary tail artifacts.

#     returns:
#       D_warp_m: [Bnn,1,H,W]
#       grid_m:   [Bnn,H,W,2]   <-- 이걸로 6채널 RGB/Linear도 샘플링
#       vis_m:    [Bnn,1,H,W]
#       rerr_m:   [Bnn,1,H,W]   (|z_tgt - D_warp|)
#       z_m:      [Bnn,1,H,W]   (target z)
#       valid_warp: [Bnn,1,H,W] (boundary-safe + z-consistent validity)
#     """
#     # backproject
#     Bnn, _, H_in, W_in = D_src_m.shape
#     X_src = backproject_depth(D_src_m, Kinv_src_m)  # [Bnn,3,H,W]
#     # transform: tgt <- src
#     Xh = torch.cat([X_src, torch.ones_like(D_src_m)], dim=1)  # [Bnn,4,H,W]
#     Xh = Xh.reshape(Xh.shape[0], 4, -1)  # [Bnn,4,HW]
#     Xh_t = torch.bmm(T_tgt_from_src_m, Xh)  # [Bnn,4,HW]
#     X_tgt = Xh_t[:, :3, :].reshape(D_src_m.shape[0], 3, H_in, W_in)  # [Bnn,3,H,W]
#     # project
#     grid_m, z_m = project_points(X_tgt, K_tgt_m, H, W)  # [Bnn,H,W,2], [Bnn,1,H,W]

#     # SAFE GRID SAMPLING: Use warp_depth_safe to prevent boundary tails
#     # This replaces the old bilinear grid_sample with adaptive blending
#     D_warp_m, valid_safe = warp_depth_safe(
#         D_src_m,
#         grid_m,
#         z_tgt=z_m,
#         z_tol_rel=0.15,
#         align_corners=True,
#     )

#     C_warp_m, _ = (
#         warp_depth_safe(
#             C,
#             grid_m,
#             z_tgt=z_m,
#             z_tol_rel=0.15,
#             align_corners=True,
#         )
#         if C is not None
#         else (None, None)
#     )

#     # Legacy visibility check (kept for compatibility)
#     vis_m = (
#         (
#             (grid_m[..., 0] >= -1)
#             & (grid_m[..., 0] <= 1)
#             & (grid_m[..., 1] >= -1)
#             & (grid_m[..., 1] <= 1)
#         )
#         .float()
#         .unsqueeze(1)
#     )

#     # Reprojection error
#     rerr_m = torch.abs(z_m - D_warp_m)

#     # UPDATED VALIDITY: Now includes boundary safety + z-consistency from warp_depth_safe
#     # Additional checks for robustness
#     edge_safe = (
#         (
#             (grid_m[..., 0] > -0.98)
#             & (grid_m[..., 0] < 0.98)
#             & (grid_m[..., 1] > -0.98)
#             & (grid_m[..., 1] < 0.98)
#         )
#         .float()
#         .unsqueeze(1)
#     )

#     pos_depth = (D_warp_m > 0.05).float()

#     # Combine all validity checks
#     # valid_safe already includes: boundary blending + positive depth + z-consistency
#     # Add edge safety margin for extra robustness
#     valid_warp = vis_m * edge_safe * pos_depth * valid_safe
#     #print(D_warp_m.shape,grid_m.shape,vis_m.shape,rerr_m.shape,z_m.shape,valid_warp.shape,C_warp_m.shape)
#     return D_warp_m, grid_m, vis_m, rerr_m, z_m, valid_warp, C_warp_m


def warp_depth_and_grid(
    D_src_m: torch.Tensor,      # [Bnn,1,H_src,W_src]  ← "소스" 해상도
    Kinv_src_m: torch.Tensor,   # [Bnn,3,3]            ← 소스 K^{-1}
    T_tgt_from_src_m: torch.Tensor,  # [Bnn,4,4] (tgt <- src)
    K_tgt_m: torch.Tensor,      # [Bnn,3,3]            ← 타겟 K
    H: int,                     # 타겟 H_t
    W: int,                     # 타겟 W_t
    C=None,
):
    """
    타겟 해상도(H,W)에 대해, grid_sample에 넣을 '타겟→소스' 그리드를 생성하고
    소스 깊이/특징을 안전하게 워핑한다.

    반환:
      D_warp_m: [Bnn,1,H,W]        (타겟 격자에 있는 깊이, 소스에서 샘플)
      grid_m:   [Bnn,H,W,2]        (타겟→소스 그리드, [-1,1] 정규화는 소스 해상도 기준)
      vis_m:    [Bnn,1,H,W]        (그리드 in-bounds)
      rerr_m:   [Bnn,1,H,W]        (|z_tgt_est - D_warp_m|)
      z_m:      [Bnn,1,H,W]        (타겟 추정 Z)
      valid_warp: [Bnn,1,H,W]
      C_warp_m: [Bnn,1,H,W] or None
    """
    Bnn, _, H_src, W_src = D_src_m.shape
    device = D_src_m.device
    dtype  = D_src_m.dtype

    # --- 0) 준비물: K_src (= Kinv_src^{-1}), T_src_from_tgt (= (T_tgt_from_src)^{-1}), Kinv_tgt ---
    K_src_m     = torch.linalg.inv(Kinv_src_m.float()).to(dtype)
    Kinv_tgt_m  = torch.linalg.inv(K_tgt_m.float()).to(dtype)

    # --- 1) 소스 깊이를 타겟 프레임으로 전방 투영하여 대략적인 z_tgt 생성 ---
    # 1-1) 소스 깊이 → 소스 3D (소스 카메라 좌표계)
    X_src = backproject_depth(D_src_m, Kinv_src_m)  # [Bnn,3,H_src,W_src]

    # 1-2) 타겟 좌표계로 변환
    Xh = torch.cat([X_src, torch.ones_like(D_src_m)], dim=1).reshape(Bnn, 4, -1)  # [Bnn,4,HW]
    Xh_tgt = torch.bmm(T_tgt_from_src_m, Xh)                                      # [Bnn,4,HW]
    X_tgt  = Xh_tgt[:, :3, :].reshape(Bnn, 3, H_src, W_src)                       # [Bnn,3,H_src,W_src]

    # 1-3) 타겟 카메라로 투영 → 타겟 프레임 Z를 얻음
    #     여기서 얻는 Z는 소스 격자에 정렬되어 있으니, 타겟 해상도로 부드럽게 업샘플하여 근사 z_tgt로 사용
    _, z_coarse = project_points(X_tgt, K_tgt_m, H_src, W_src)   # z_coarse: [Bnn,1,H_src,W_src]
    z_tgt_est = F.interpolate(z_coarse, size=(H, W), mode="bilinear", align_corners=True)  # [Bnn,1,H,W]

    # --- 2) 타겟 픽셀들의 3D를 만들고(타겟 좌표계), 소스 카메라로 옮긴 뒤 소스 픽셀로 투영 ---
    # 2-1) 타겟 픽셀 좌표 (픽셀 단위) 생성
    ys_t, xs_t = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs_t)
    pix_t = torch.stack([xs_t, ys_t, ones], dim=0).unsqueeze(0).expand(Bnn, -1, -1, -1)  # [Bnn,3,H,W]
    pix_t = pix_t.reshape(Bnn, 3, -1)  # [Bnn,3,HW_t]

    # 2-2) 타겟 픽셀 → 타겟 카메라 3D 방향, 여기에 z_tgt_est 곱해서 3D 위치
    rays_t = torch.bmm(Kinv_tgt_m, pix_t)                          # [Bnn,3,HW_t]
    Zt     = z_tgt_est.reshape(Bnn, 1, -1)                         # [Bnn,1,HW_t]
    X_tgt_dense = rays_t * Zt                                      # [Bnn,3,HW_t]

    # 2-3) 타겟 3D → 소스 3D (T_src_from_tgt 적용)
    #      T_src_from_tgt = (T_tgt_from_src)^{-1}
    with torch.cuda.amp.autocast(enabled=False):
        T_src_from_tgt_m = torch.linalg.inv(T_tgt_from_src_m.float()).to(dtype)
        X_tgt_dense_h = torch.cat([X_tgt_dense, torch.ones(Bnn, 1, H * W, device=device, dtype=dtype)], dim=1)  # [Bnn,4,HW_t]
        X_src_dense_h = torch.bmm(T_src_from_tgt_m, X_tgt_dense_h)                                             # [Bnn,4,HW_t]
        X_src_dense   = X_src_dense_h[:, :3, :]                                                                # [Bnn,3,HW_t]

        # 2-4) 소스 카메라로 투영 → 소스 픽셀 좌표 (정규화 전)
        x_src = torch.bmm(K_src_m, X_src_dense)  # [Bnn,3,HW_t]
        z_src = x_src[:, 2:3, :]

        # 🔴 z 유효성 마스크: 양수 & 최소 깊이 확보
        eps_z = 1e-6
        z_ok = (z_src > eps_z)

        # 🔴 invalid z 경로는 그래프 차단(detach)하여 backward에 NaN이 안타게
        z_safe = torch.where(z_ok, z_src, torch.full_like(z_src, eps_z).detach())

        u_src = x_src[:, 0:1, :] / z_safe
        v_src = x_src[:, 1:2, :] / z_safe

        # 2-5) grid_sample용 [-1,1] 정규화 (중요: 소스 해상도(H_src,W_src) 기준)
        u_n = (u_src / (W_src - 1) - 0.5) * 2.0
        v_n = (v_src / (H_src - 1) - 0.5) * 2.0
        grid_m = torch.stack([u_n, v_n], dim=-1).reshape(Bnn, H, W, 2)  # [Bnn,H_t,W_t,2]

    # --- 3) 안전 워핑: 소스 깊이/특징을 타겟 격자(H,W)로 샘플 ---
    #     깊이는 disparity 도메인에서 샘플하여 노이즈/경계 안전화
    D_warp_m, valid_safe = warp_depth_safe(
        D_src_m, grid_m, z_tgt=z_tgt_est, z_tol_rel=0.15, align_corners=True
    )

    C_warp_m, _ = (
        warp_depth_safe(C, grid_m, z_tgt=z_tgt_est, z_tol_rel=0.15, align_corners=True)
        if C is not None else (None, None)
    )

    # --- 4) 부가 마스크/지표 ---
    # In-bounds visibility (grid가 [-1,1] 안에 있는지)
    vis_m = (
        ((grid_m[..., 0] >= -1.0) & (grid_m[..., 0] <= 1.0) &
         (grid_m[..., 1] >= -1.0) & (grid_m[..., 1] <= 1.0))
        .float().unsqueeze(1)  # [Bnn,1,H,W]
    )

    # 에지 여유 마진
    edge_safe = (
        ((grid_m[..., 0] > -0.98) & (grid_m[..., 0] < 0.98) &
         (grid_m[..., 1] > -0.98) & (grid_m[..., 1] < 0.98))
        .float().unsqueeze(1)
    )

    pos_depth = (D_warp_m > 0.05).float()
    valid_warp = vis_m * edge_safe * pos_depth * valid_safe

    # 재투영 오차 (근사 z_tgt_est와의 차이)
    rerr_m = torch.abs(z_tgt_est - D_warp_m)  # [Bnn,1,H,W]

    # z 반환은 타겟 추정 Z
    
    grid_m = torch.nan_to_num(grid_m, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0001, 1.0001)
    z_tgt_est = torch.nan_to_num(z_tgt_est, nan=0.0, posinf=0.0, neginf=0.0)
    
    z_m = z_tgt_est
    def _chk(name, t):
        if not torch.isfinite(t).all():
            print(f"[NaN/Inf] {name}")
    _chk("D_warp_m", D_warp_m)
    _chk("grid_m", grid_m)
    _chk("vis_m", vis_m)
    _chk("rerr_m", rerr_m)
    _chk("z_m", z_m)
    _chk("valid_warp", valid_warp)
    if C_warp_m is not None:
        _chk("C_warp_m", C_warp_m)
    _chk("z_src", z_src)
    _chk("u_n", u_n)
    _chk("v_n", v_n)
    return D_warp_m, grid_m, vis_m, rerr_m, z_m, valid_warp, C_warp_m
