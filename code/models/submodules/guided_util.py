import torch
import torch.nn.functional as F
def scale_intrinsics(K: torch.Tensor, s: float):
    """K 스케일 조정: fx, fy, cx, cy에 s 곱."""
    K_ = K.clone()
    K_[..., 0, 0] *= s
    K_[..., 1, 1] *= s
    K_[..., 0, 2] *= s
    K_[..., 1, 2] *= s
    return K_




def grid_from_target_depth(
    D_tgt_m: torch.Tensor,
    Kinv_tgt_m: torch.Tensor,
    T_src_from_tgt_m: torch.Tensor,
    K_src_m: torch.Tensor,
    H: int,
    W: int,
    align_corners: bool = True,
):
    """
    Build grid for sampling source images given target depth.
    D_tgt_m: [Bmn,1,H,W]
    Kinv_tgt_m: [Bmn,3,3]
    T_src_from_tgt_m: [Bmn,4,4]
    K_src_m: [Bmn,3,3]

    Returns: grid_ts [Bmn,H,W,2], vis [Bmn,1,H,W], z_src [Bmn,1,H,W]
    """
    device = D_tgt_m.device
    Bmn = D_tgt_m.shape[0]

    # create pixel grid in image coords (u along width, v along height)
    ys = torch.linspace(0, H - 1, H, device=device)
    xs = torch.linspace(0, W - 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    # homogeneous pixel coordinates [3, HW]
    ones = torch.ones_like(grid_x)
    pix = torch.stack([grid_x, grid_y, ones], dim=0).view(3, -1)  # [3, HW]

    # Expand to batch: [Bmn,3,HW]
    pix_b = pix.unsqueeze(0).expand(Bmn, -1, -1).to(device)

    # Compute camera rays in tgt cam: Kinv_tgt_m @ pix_b -> [Bmn,3,HW]
    rays = torch.bmm(Kinv_tgt_m, pix_b)  # direction vectors

    # multiply by depth to get 3D points in tgt cam coords
    D_flat = D_tgt_m.view(Bmn, 1, -1)  # [Bmn,1,HW]
    X_tgt = rays * D_flat  # [Bmn,3,HW]

    # make homogeneous and transform to src cam: Xh_src = T_src_from_tgt_m @ [X;1]
    ones_row = torch.ones(Bmn, 1, X_tgt.shape[-1], device=device)
    Xh_tgt = torch.cat([X_tgt, ones_row], dim=1)  # [Bmn,4,HW]
    Xh_src = torch.bmm(T_src_from_tgt_m, Xh_tgt)  # [Bmn,4,HW]
    X_src = Xh_src[:, :3, :]  # [Bmn,3,HW]
    z_src = X_src[:, 2:3, :].view(Bmn, 1, H, W)

    # project to src image plane: u = (K_src * X_src) -> normalize by z
    proj = torch.bmm(K_src_m, X_src)  # [Bmn,3,HW]
    u = proj[:, 0:1, :] / (proj[:, 2:3, :] + 1e-8)
    v = proj[:, 1:2, :] / (proj[:, 2:3, :] + 1e-8)

    # reshape to [Bmn,H,W]
    u_img = u.view(Bmn, H, W)
    v_img = v.view(Bmn, H, W)

    # convert to grid coords in [-1,1]
    if align_corners:
        gx = (2.0 * u_img / (W - 1)) - 1.0
        gy = (2.0 * v_img / (H - 1)) - 1.0
    else:
        gx = (2.0 * (u_img + 0.5) / W) - 1.0
        gy = (2.0 * (v_img + 0.5) / H) - 1.0

    grid_ts = torch.stack([gx, gy], dim=-1)  # [Bmn,H,W,2]

    # Ensure gx/gy have a channel dim so broadcasting with z_src is correct
    gx_c = gx.unsqueeze(1)  # [Bmn,1,H,W]
    gy_c = gy.unsqueeze(1)  # [Bmn,1,H,W]

    # visibility: within bounds and positive depth
    vis = (
        (gx_c >= -1.0) & (gx_c <= 1.0) & (gy_c >= -1.0) & (gy_c <= 1.0) & (z_src > 1e-6)
    ).float()
    return grid_ts, vis, z_src


# ===== (하이브리드: 전 뷰 동시 처리) =====

@torch.no_grad()
def precompute_pairwise_T_src_from_tgt(T_src_w_from_cam, T_tgt_w_from_cam):
    """
    Compute pairwise transforms (src <- tgt) for all (tgt, src) pairs.
    Inputs are world <- cam (T_w_from_cam) for src and tgt respectively.
    Returns: [B, M, N, 4, 4]
    """
    T_src = T_src_w_from_cam.to(torch.float32).contiguous()
    T_tgt = T_tgt_w_from_cam.to(torch.float32).contiguous()
    B, N, _, _ = T_src.shape
    _, M, _, _ = T_tgt.shape

    # compute cam_from_w for src (inverse of world<-cam)
    R_wc_s = T_src[..., :3, :3]
    t_wc_s = T_src[..., :3, 3:4]
    R_cw_s = R_wc_s.transpose(-1, -2).contiguous()
    t_cw_s = -torch.matmul(R_cw_s, t_wc_s)
    T_cam_from_w_s = torch.zeros(B, N, 4, 4, dtype=torch.float32, device=T_src.device)
    T_cam_from_w_s[..., :3, :3] = R_cw_s
    T_cam_from_w_s[..., :3, 3] = t_cw_s.reshape(B, N, 3)
    T_cam_from_w_s[..., 3, 3] = 1.0

    # w_from_tgt is simply T_tgt (world <- tgt_cam)
    T_w_from_tgt = T_tgt

    # compose: (src <- tgt) = (src <- w) @ (w <- tgt) = cam_from_w_s @ T_w_from_tgt
    T_src_from_w = T_cam_from_w_s.unsqueeze(1)  # [B,1,N,4,4]
    T_w_from_tgt = T_w_from_tgt.unsqueeze(2)  # [B,M,1,4,4]
    T_src_from_tgt = torch.matmul(T_src_from_w, T_w_from_tgt)  # [B,M,N,4,4]

    return T_src_from_tgt.contiguous()

def downsample_area(x: torch.Tensor, s: float = 0.5, mode="area"):
    """anti-aliasing area downsample"""
    H, W = x.shape[-2:]
    nh, nw = int(H * s), int(W * s)
    out = F.interpolate(x, (nh, nw), mode=mode)
    if mode == "nearest":
        out = torch.where(out < 0.05, torch.zeros_like(out), out)
    return out


def upsample_bilinear(x: torch.Tensor, size_hw):
    """bilinear upsample (align_corners=False)"""
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)


# hybrid_densify.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== Utilities ======
def box_count(mask: torch.Tensor, k: int = 9) -> torch.Tensor:
    # mask: [B,1,H,W] in {0,1}
    pad = k // 2
    cnt = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=pad)  # [0,1]
    return cnt  # local density ratio ∈ [0,1]


def smoothstep(edge0, edge1, x):
    # clamp01((x-edge0)/(edge1-edge0))^2 * (3 - 2*...)
    t = torch.clamp((x - edge0) / max(1e-6, (edge1 - edge0)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tiny_median(x):
    # 3x3 median-like with separable approximation (fast)
    return F.avg_pool2d(x, 3, 1, 1)


# ====== Guided Filter (fast, RGB-guided, O(1) 근사) ======
def guided_filter_fast(guide, src, r=4, eps=1e-3):
    # guide: [B,3,H,W], src: [B,1,H,W]
    mean_g = F.avg_pool2d(guide, 2 * r + 1, 1, r)
    mean_s = F.avg_pool2d(src, 2 * r + 1, 1, r)
    mean_gg = F.avg_pool2d(guide * guide, 2 * r + 1, 1, r)
    mean_gs = F.avg_pool2d(guide * src, 2 * r + 1, 1, r)

    var_g = mean_gg - mean_g * mean_g  # [B,3,H,W]
    cov_gs = mean_gs - mean_g * mean_s  # [B,3,H,W]

    # scalar A using channel-mean to avoid 3x3 inversion per-pixel
    var_g_scalar = var_g.mean(1, keepdim=True)  # [B,1,H,W]
    A = cov_gs.mean(1, keepdim=True) / (var_g_scalar + eps)
    b = mean_s - A * mean_g.mean(1, keepdim=True)

    mean_A = F.avg_pool2d(A, 2 * r + 1, 1, r)
    mean_b = F.avg_pool2d(b, 2 * r + 1, 1, r)

    out = mean_A * guide.mean(1, keepdim=True) + mean_b
    return out
