import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.hdr.mertens_torch import mertens_merge_differentiable
from torch.cuda.amp import autocast, GradScaler

def masked_mean(x, mask, eps=1e-6):
    mask = mask.float()
    return (x * mask).sum() / (mask.sum() + eps)


def depth_charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def _depth_grad_xy(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def _pad_last_dim_replicate(x):
    return torch.cat([x, x[..., -1:]], dim=-1)


def _pad_second_last_dim_replicate(x):
    return torch.cat([x, x[..., -1:, :]], dim=-2)


def _max_pool_spatial(x, kernel_size):
    if x.dim() < 4:
        raise ValueError(f"Expected depth tensor with spatial dims, got {tuple(x.shape)}")
    orig_shape = x.shape
    x4 = x.reshape(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
    pooled = F.max_pool2d(
        x4,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return pooled.reshape(orig_shape)


def _depth_to_normal(depth_map):
    gx, gy = _depth_grad_xy(depth_map)
    gx = _pad_last_dim_replicate(gx)
    gy = _pad_second_last_dim_replicate(gy)
    nx = -gx
    ny = -gy
    nz = torch.ones_like(depth_map)
    channel_dim = depth_map.dim() - 3
    n = torch.cat([nx, ny, nz], dim=channel_dim)
    return F.normalize(n, dim=channel_dim, eps=1e-6)


def depth_edge_recall_loss(
    pred_log_depth,
    gt_log_depth,
    valid_mask,
    radius=1,
    recall_weight=1.0,
    precision_weight=0.2,
    margin=0.0,
):
    gx_p, gy_p = _depth_grad_xy(pred_log_depth)
    gx_g, gy_g = _depth_grad_xy(gt_log_depth)
    vmx = valid_mask[..., :, 1:] * valid_mask[..., :, :-1]
    vmy = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]

    gx_p = _pad_last_dim_replicate(gx_p.abs() * vmx)
    gy_p = _pad_second_last_dim_replicate(gy_p.abs() * vmy)
    gx_g = _pad_last_dim_replicate(gx_g.abs() * vmx)
    gy_g = _pad_second_last_dim_replicate(gy_g.abs() * vmy)
    gp = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-12)
    gg = torch.sqrt(gx_g * gx_g + gy_g * gy_g + 1e-12)

    k = 2 * int(radius) + 1
    gp_near = _max_pool_spatial(gp, k)
    gg_near = _max_pool_spatial(gg, k)
    recall = F.relu(gg - gp_near - float(margin))
    precision = F.relu(gp - gg_near - float(margin))
    edge_weight_gt = (gg / gg.detach().mean().clamp_min(1e-6)).detach().clamp(0.0, 5.0)
    edge_weight_p = (gp / gp.detach().mean().clamp_min(1e-6)).detach().clamp(0.0, 5.0)
    recall_loss = masked_mean(recall * edge_weight_gt, valid_mask)
    precision_loss = masked_mean(precision * edge_weight_p, valid_mask)
    loss = float(recall_weight) * recall_loss + float(precision_weight) * precision_loss

    with torch.no_grad():
        edge_mask_gt = (gg > gg.mean().clamp_min(1e-6)).float() * valid_mask
        edge_mask_p = (gp > gp.mean().clamp_min(1e-6)).float() * valid_mask
        edge_recall = masked_mean(((gp_near + float(margin)) >= gg).float(), edge_mask_gt)
        edge_precision = masked_mean(((gg_near + float(margin)) >= gp).float(), edge_mask_p)
    return loss, {
        "edge_recall_loss": float(recall_loss.detach().item()),
        "edge_precision_loss": float(precision_loss.detach().item()),
        "edge_recall": float(edge_recall.detach().item()),
        "edge_precision": float(edge_precision.detach().item()),
        "edge_pred_grad": float(gp.detach().mean().item()),
        "edge_gt_grad": float(gg.detach().mean().item()),
    }


def _depth_loss_legacy(
    pred_depth,
    gt_depth,
    valid_mask=None,
    max_depth=65.0,
    near_thr=15.0,
    n_log_bins=16,
    eps=1e-6,
):
    """
    Depth loss function with range-balanced weighting and multiple terms
    """
    if valid_mask is None:
        valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth)
    valid_mask = valid_mask.float()

    # region masks
    near_mask = ((gt_depth > 0.01) & (gt_depth < near_thr)).float() * valid_mask
    far_mask = ((gt_depth >= near_thr) & (gt_depth <= max_depth)).float() * valid_mask
    inf_mask = ((gt_depth >= max_depth) & torch.isfinite(gt_depth)).float() * valid_mask

    has_near = near_mask.sum() >= 1

    has_inf = inf_mask.sum() >= 1

    # clamp for stability (GT만)
    gt = torch.clamp(gt_depth, 0.01, max_depth)
    pred = torch.clamp(
        pred_depth, min=eps, max=3.0 * max_depth
    )  # Upper limit to prevent explosion

    # ---------- Range-balanced weights (log-bins over metric depth) ----------
    with torch.no_grad():
        # log-space binning over [0.1, max_depth]
        z = torch.clamp(gt, 0.1, max_depth)
        logz = torch.log(z)  # natural log
        logmin, logmax = torch.log(torch.tensor(0.1, device=z.device)), torch.log(
            torch.tensor(max_depth, device=z.device)
        )
        # bin index in [0, n_log_bins-1]
        bin_idx = torch.clamp(
            ((logz - logmin) / (logmax - logmin) * n_log_bins).long(), 0, n_log_bins - 1
        )
        # count per bin
        weights = torch.ones_like(z)
        for k in range(n_log_bins):
            cnt = ((bin_idx == k) & (valid_mask > 0)).sum().float()
            w = 1.0 / (cnt + 1e-6)
            weights = torch.where(bin_idx == k, w, weights)
        # normalize to mean 1 over valid pixels
        w_mean = (weights * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        weights = weights / (w_mean + 1e-6)

    # ---------- Loss terms ----------

    def grad_xy(x):
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    def pad_last_dim_replicate(x):
        return torch.cat([x, x[..., -1:]], dim=-1)

    def pad_second_last_dim_replicate(x):
        return torch.cat([x, x[..., -1:, :]], dim=-2)

    def max_pool_spatial_5d(x, kernel_size):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,M,C,H,W], got {tuple(x.shape)}")
        b, m, c, h, w = x.shape
        x_flat = x.reshape(b * m, c, h, w)
        x_pool = F.max_pool2d(
            x_flat,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        return x_pool.reshape(b, m, c, h, w)

    def depth_to_normal(depth_map):
        # Simple pseudo-normal from metric depth; enough to encourage locally
        # planar surfaces without adding a heavy geometric camera model.
        gx, gy = grad_xy(depth_map)
        gx = pad_last_dim_replicate(gx)
        gy = pad_second_last_dim_replicate(gy)
        nx = -gx
        ny = -gy
        nz = torch.ones_like(depth_map)
        channel_dim = depth_map.dim() - 3
        n = torch.cat([nx, ny, nz], dim=channel_dim)
        return F.normalize(n, dim=channel_dim, eps=1e-6)

    # 1) L1 in metric depth (near only): absolute scale anchor
    l1_z = torch.tensor(0.0, device=pred.device)
    if has_near:
        e = (pred - gt).abs() * near_mask
        l1_z = e.sum() / (near_mask.sum() + 1e-6)

    # 2) Charbonnier in inverse depth (all valid): dynamic range friendly
    inv_pred = 1.0 / (pred + eps)
    inv_gt = 1.0 / (gt + eps)
    charbonnier_eps = 1e-3
    e_inv = torch.sqrt(((inv_pred - inv_gt) ** 2 + charbonnier_eps**2))
    l1_s = (e_inv * weights * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    # 3) SILog (very low weight): scale-invariant alignment
    # d = log(pred) - log(gt)
    d = torch.log(pred) - torch.log(gt)
    d = d * valid_mask
    n = valid_mask.sum() + 1e-6
    silog = (d.pow(2).sum() / n) - 0.85 * (d.sum() / n).pow(
        2
    )  # λ=0.85 (표준), 전체 weight는 작게

    # 4) Gradient consistency in log-depth (structure preservation)
    log_pred = torch.log(pred)
    log_gt = torch.log(gt)
    gx_p, gy_p = grad_xy(log_pred)
    gx_g, gy_g = grad_xy(log_gt)
    # valid masks for gradients
    vmx = valid_mask[..., :, 1:] * valid_mask[..., :, :-1]
    vmy = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]
    gt_edge_x = gx_g.abs()
    gt_edge_y = gy_g.abs()
    edge_wx = (1.0 + 4.0 * (gt_edge_x / (gt_edge_x.mean().detach() + 1e-6))).clamp(1.0, 6.0)
    edge_wy = (1.0 + 4.0 * (gt_edge_y / (gt_edge_y.mean().detach() + 1e-6))).clamp(1.0, 6.0)
    grad_loss = (
        ((gx_p - gx_g).abs() * edge_wx * vmx).sum() / ((edge_wx * vmx).sum() + 1e-6)
        + ((gy_p - gy_g).abs() * edge_wy * vmy).sum() / ((edge_wy * vmy).sum() + 1e-6)
    )

    # 4b) Boundary-band loss: emphasize pixels near GT discontinuities so object
    # contours stay crisp instead of being averaged away.
    gx_g_full = pad_last_dim_replicate(gx_g.abs())
    gy_g_full = pad_second_last_dim_replicate(gy_g.abs())
    edge_mag = torch.sqrt(gx_g_full.pow(2) + gy_g_full.pow(2) + 1e-12)
    edge_seed = (edge_mag > 0.08).float()
    boundary_band = max_pool_spatial_5d(edge_seed, kernel_size=5) * valid_mask
    boundary_loss = ((pred - gt).abs() * boundary_band).sum() / (boundary_band.sum() + 1e-6)

    # 4c) Local surface orientation consistency: helps recover flat faces and
    # sharp transitions together.
    normal_pred = depth_to_normal(pred)
    with torch.no_grad():
        normal_gt = depth_to_normal(gt)
    channel_dim = normal_pred.dim() - 3
    normal_loss = (
        (1.0 - (normal_pred * normal_gt).sum(dim=channel_dim, keepdim=True)).clamp_min(0.0)
        * valid_mask
    ).sum() / (valid_mask.sum() + 1e-6)

    # 5) Far/sky hinge (encourage large depth for >max or sky)
    far_loss = torch.tensor(0.0, device=pred.device)
    if has_inf:
        # soft hinge: softplus(τ*(d_max - pred)) / τ
        tau = 5.0
        far_loss = (
            F.softplus((max_depth - pred) * inf_mask * tau).sum()
            / (inf_mask.sum() + 1e-6)
            / tau
        )

    # ---------- Weighted sum (초기 권장값) ----------
    w_l1_z = 1.0  # near 절대 스케일
    w_l1_s = 0.5  # 전체 범위 안정화
    w_grad = 0.5  # 구조 보존
    w_silog = 0.10  # flattening 방지 위해 작게
    w_far = 0.10  # 하늘/무한 영역
    w_boundary = 0.75  # GT edge band sharpening
    w_normal = 0.20  # local plane/orientation consistency

    total = (
        w_l1_z * l1_z
        + w_l1_s * l1_s
        + w_grad * grad_loss
        + w_boundary * boundary_loss
        + w_normal * normal_loss
        + w_silog * silog
        + w_far * far_loss
    )

    stats = dict(
        l1_depth=l1_z.item(),
        l1_inv=l1_s.item(),
        grad=grad_loss.item(),
        boundary=boundary_loss.item(),
        normal=normal_loss.item(),
        silog=silog.item(),
        far=far_loss.item(),
    )
    return total, stats


def depth_loss(
    pred_depth,
    gt_depth,
    valid_mask=None,
    max_depth=65.0,
    near_thr=15.0,
    n_log_bins=16,
    eps=1e-6,
    version="v2",
    log_weight=1.0,
    rel_weight=0.3,
    inv_weight=0.2,
    near_l1_weight=0.05,
    grad_weight=0.3,
    edge_recall_weight=0.4,
    normal_weight=0.1,
    scale_weight=0.5,
    shape_weight=0.2,
    far_weight=0.1,
    edge_radius=1,
):
    if version == "legacy":
        return _depth_loss_legacy(
            pred_depth,
            gt_depth,
            valid_mask=valid_mask,
            max_depth=max_depth,
            near_thr=near_thr,
            n_log_bins=n_log_bins,
            eps=eps,
        )

    if valid_mask is None:
        valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth)
    valid_mask = (
        valid_mask.float()
        * torch.isfinite(pred_depth).float()
        * torch.isfinite(gt_depth).float()
    ).float()
    valid_count = valid_mask.sum()
    zero = pred_depth.sum() * 0.0
    if valid_count < 1:
        stats = {
            "l1_depth": 0.0,
            "l1_inv": 0.0,
            "grad": 0.0,
            "boundary": 0.0,
            "normal": 0.0,
            "silog": 0.0,
            "far": 0.0,
            "depth_log": 0.0,
            "depth_rel": 0.0,
            "depth_inv": 0.0,
            "depth_shape": 0.0,
            "depth_scale": 0.0,
            "depth_edge_recall_loss": 0.0,
            "depth_edge_recall": 0.0,
            "depth_edge_precision": 0.0,
            "depth_near_absrel": 0.0,
            "depth_far_absrel": 0.0,
        }
        return zero, stats

    gt = torch.clamp(gt_depth, 0.05, max_depth)
    pred = torch.clamp(pred_depth, min=0.05, max=3.0 * max_depth)
    valid_mask = valid_mask * (gt_depth > 0).float()
    near_mask = ((gt_depth > 0.01) & (gt_depth < near_thr)).float() * valid_mask
    far_mask = ((gt_depth >= near_thr) & (gt_depth <= max_depth)).float() * valid_mask
    inf_mask = ((gt_depth >= max_depth) & torch.isfinite(gt_depth)).float() * valid_mask

    log_pred = torch.log(pred)
    log_gt = torch.log(gt)
    e_log = log_pred - log_gt
    l_log = masked_mean(depth_charbonnier(e_log), valid_mask)

    e_rel = (pred - gt) / gt.clamp_min(0.05)
    l_rel = masked_mean(depth_charbonnier(e_rel), valid_mask)

    inv_pred = 1.0 / pred.clamp_min(0.05)
    inv_gt = 1.0 / gt.clamp_min(0.05)
    inv_norm = masked_mean(inv_gt, valid_mask).detach().clamp_min(1e-6)
    l_inv = masked_mean(depth_charbonnier((inv_pred - inv_gt) / inv_norm), valid_mask)

    e_log_mean = masked_mean(e_log, valid_mask)
    l_scale = depth_charbonnier(e_log_mean)
    l_shape = masked_mean(depth_charbonnier(e_log - e_log_mean.detach()), valid_mask)

    l_near = zero
    if near_mask.sum() >= 1:
        l_near = masked_mean((pred - gt).abs(), near_mask)

    gx_p, gy_p = _depth_grad_xy(log_pred)
    gx_g, gy_g = _depth_grad_xy(log_gt)
    vmx = valid_mask[..., :, 1:] * valid_mask[..., :, :-1]
    vmy = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]
    grad_loss = (
        masked_mean(depth_charbonnier(gx_p - gx_g), vmx)
        + masked_mean(depth_charbonnier(gy_p - gy_g), vmy)
    )

    edge_loss, edge_stats = depth_edge_recall_loss(
        log_pred,
        log_gt.detach(),
        valid_mask,
        radius=edge_radius,
        recall_weight=1.0,
        precision_weight=0.2,
    )

    normal_pred = _depth_to_normal(log_pred)
    with torch.no_grad():
        normal_gt = _depth_to_normal(log_gt)
    channel_dim = normal_pred.dim() - 3
    normal_loss = masked_mean(
        (1.0 - (normal_pred * normal_gt).sum(dim=channel_dim, keepdim=True)).clamp_min(0.0),
        valid_mask,
    )

    far_loss = zero
    if inf_mask.sum() >= 1:
        tau = 5.0
        far_loss = masked_mean(F.softplus((max_depth - pred) * tau) / tau, inf_mask)

    total = (
        float(log_weight) * l_log
        + float(rel_weight) * l_rel
        + float(inv_weight) * l_inv
        + float(shape_weight) * l_shape
        + float(scale_weight) * l_scale
        + float(near_l1_weight) * l_near
        + float(grad_weight) * grad_loss
        + float(edge_recall_weight) * edge_loss
        + float(normal_weight) * normal_loss
        + float(far_weight) * far_loss
    )

    with torch.no_grad():
        absrel = (pred - gt).abs() / gt.clamp_min(0.05)
        near_absrel = masked_mean(absrel, near_mask).detach() if near_mask.sum() >= 1 else zero.detach()
        far_absrel = masked_mean(absrel, far_mask).detach() if far_mask.sum() >= 1 else zero.detach()

    stats = dict(
        l1_depth=float(l_near.detach().item()),
        l1_inv=float(l_inv.detach().item()),
        grad=float(grad_loss.detach().item()),
        boundary=float(edge_loss.detach().item()),
        normal=float(normal_loss.detach().item()),
        silog=float(l_shape.detach().item()),
        far=float(far_loss.detach().item()),
        depth_log=float(l_log.detach().item()),
        depth_rel=float(l_rel.detach().item()),
        depth_inv=float(l_inv.detach().item()),
        depth_shape=float(l_shape.detach().item()),
        depth_scale=float(l_scale.detach().item()),
        depth_edge_recall_loss=edge_stats["edge_recall_loss"],
        depth_edge_precision_loss=edge_stats["edge_precision_loss"],
        depth_edge_recall=edge_stats["edge_recall"],
        depth_edge_precision=edge_stats["edge_precision"],
        depth_near_absrel=float(near_absrel.item()),
        depth_far_absrel=float(far_absrel.item()),
    )
    return total, stats


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], use_l1=True, max_size=256):
        """
        layer_ids: VGG16 feature layer indices to extract
        use_l1: L1 loss vs MSE loss
        max_size: perceptual loss 계산 시 최대 해상도 (짧은 변 기준)
        """
        super().__init__()
        self.vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.layers = set(layer_ids)
        self.use_l1 = use_l1
        self.max_size = max_size

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _resize(self, x):
        """Perceptual loss 계산 시 입력 크기를 제한"""
        B, C, H, W = x.shape
        if min(H, W) > self.max_size:
            scale = self.max_size / min(H, W)
            newH, newW = int(H * scale), int(W * scale)
            x = F.interpolate(
                x, size=(newH, newW), mode="bilinear", align_corners=False
            )
        return x

    def forward(self, input, target):
        # [B,N,3,H,W] -> [B*N,3,H,W]
        if input.dim() == 5:
            B, N, C, H, W = input.shape
            input = input.reshape(B * N, C, H, W)
            target = target.reshape(B * N, C, H, W)

        # 0~1 clamp
        input = torch.clamp(input, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 해상도 줄이기
        input = self._resize(input)
        target = self._resize(target)

        # Normalize
        input_n = self._normalize(input)
        with torch.no_grad():
            target_n = self._normalize(target)

        loss = 0.0
        x, y = input_n, target_n
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            with torch.no_grad():
                y = layer(y)
            if i in self.layers:
                if self.use_l1:
                    loss = loss + F.l1_loss(x, y)
                else:
                    loss = loss + F.mse_loss(x, y)
        return loss


# Global perceptual loss will be lazily initialized per device
_perceptual_loss_cache = {}


class MuTonemap(nn.Module):
    def __init__(self, mu=10000):
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("den", torch.log1p(torch.tensor(float(mu))))

    def forward(self, x):
        return torch.log1p(self.mu * torch.clamp(x, min=0)) / self.den


# Global MuTonemap will be lazily initialized per device
_mu_tonemap_cache = {}


def get_perceptual_loss(device):
    """Get device-specific perceptual loss with lazy initialization"""
    if device not in _perceptual_loss_cache:
        _perceptual_loss_cache[device] = VGGPerceptualLoss().to(device)
    return _perceptual_loss_cache[device]


def tonemap_mu_law(hdr, mu=1000):
    """μ-law 톤 매핑 적용 with device-aware lazy initialization"""
    device = hdr.device

    # Lazy initialization per device
    if device not in _mu_tonemap_cache:
        _mu_tonemap_cache[device] = {}
    if mu not in _mu_tonemap_cache[device]:
        _mu_tonemap_cache[device][mu] = MuTonemap(mu=mu).to(device)

    mu_tonemap = _mu_tonemap_cache[device][mu]
    return mu_tonemap(hdr)


def weight_map(image, threshold=0.05, weight_high=0.5, weight_low=2.0):
    """어두운 영역을 강조하는 가중치 맵"""
    grayscale = (
        0.2126 * image[:, 0, :, :]
        + 0.7152 * image[:, 1, :, :]
        + 0.0722 * image[:, 2, :, :]
    )
    weight = torch.where(grayscale < threshold, weight_low, weight_high)
    return weight.unsqueeze(1)  # (B, 1, H, W)


def gaussian_kernel(window_size=11, sigma=1.5, channels=3):
    """
    가우시안 블러 커널 생성
    """
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    kernel = g[:, None] * g[None, :]
    kernel = kernel.repeat(channels, 1, 1, 1)  # (C, 1, H, W)
    return kernel


def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0, K1=0.01, K2=0.03):
    """
    SSIM (Structural Similarity Index Measure) 계산
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 가우시안 블러 커널 생성
    _, C, _, _ = img1.shape
    kernel = gaussian_kernel(window_size, sigma, C).to(img1.device)

    # 평균 계산 (가우시안 필터 적용)
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=C)

    # 분산 및 공분산 계산
    sigma1_sq = (
        F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=C) - mu1**2
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=C) - mu2**2
    )
    sigma12 = (
        F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=C) - mu1 * mu2
    )

    # SSIM 계산
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()
def sobel_grad(x):
    # x: [B*M, 3, H, W] or [B, M, 3, H, W]
    if x.dim() == 5:
        B,M,C,H,W = x.shape
        x = x.view(B*M, C, H, W)
    kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
    ky = x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
    kx = kx.repeat(x.shape[1],1,1,1); ky = ky.repeat(x.shape[1],1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return gx, gy

def charbonnier(x, eps=1e-3):
    """Robust L1 used for shift-tolerant HDR supervision."""
    return torch.sqrt(x * x + eps * eps)

def _flatten_bmchw(x):
    if x.dim() == 5:
        b, m, c, h, w = x.shape
        return x.reshape(b * m, c, h, w), (b, m, c, h, w)
    return x, None

def local_softmin_l1(pred, target, radius=1, tau=0.02, eps=1e-3):
    """Local alignment-tolerant robust L1 in tonemapped space.

    For each pred pixel, compare against a local target window and use a
    detached softmin over shifts. This rewards sharp but sub-pixel/one-pixel
    shifted edges without letting the network optimize the alignment selector.
    """
    pred4, _ = _flatten_bmchw(pred)
    target4, _ = _flatten_bmchw(target)
    if radius <= 0:
        return charbonnier(pred4 - target4, eps=eps).mean()

    b, c, h, w = pred4.shape
    k = 2 * int(radius) + 1
    pad = int(radius)
    padded = F.pad(target4, (pad, pad, pad, pad), mode="replicate")
    patches = F.unfold(padded, kernel_size=k).view(b, c, k * k, h, w)
    err = charbonnier(pred4.unsqueeze(2) - patches, eps=eps).mean(dim=1)  # [B,K,H,W]
    weights = torch.softmax((-err.detach() / max(float(tau), 1e-6)), dim=1)
    return (weights * err).sum(dim=1).mean()

def edge_recall_loss(
    pred_tm,
    target_tm,
    radius=1,
    recall_weight=1.0,
    precision_weight=0.2,
    margin=0.0,
):
    """Shift-tolerant edge loss.

    Exact Sobel matching punishes one-pixel shifted sharp edges. This loss asks
    that a GT edge has a nearby predicted edge, with a weaker reverse precision
    term to discourage hallucinated high frequency.
    """
    pred4, _ = _flatten_bmchw(pred_tm)
    target4, _ = _flatten_bmchw(target_tm)
    gx_p, gy_p = sobel_grad(pred4)
    gx_t, gy_t = sobel_grad(target4)
    gp = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-12).mean(dim=1, keepdim=True)
    gt = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-12).mean(dim=1, keepdim=True)
    k = 2 * int(radius) + 1
    pad = int(radius)
    gp_near = F.max_pool2d(gp, kernel_size=k, stride=1, padding=pad)
    gt_near = F.max_pool2d(gt, kernel_size=k, stride=1, padding=pad)

    recall = F.relu(gt - gp_near - float(margin))
    precision = F.relu(gp - gt_near - float(margin))
    edge_weight_gt = (gt / gt.detach().mean().clamp_min(1e-6)).detach().clamp(0.0, 5.0)
    edge_weight_p = (gp / gp.detach().mean().clamp_min(1e-6)).detach().clamp(0.0, 5.0)
    recall_loss = (recall * edge_weight_gt).mean()
    precision_loss = (precision * edge_weight_p).mean()
    loss = float(recall_weight) * recall_loss + float(precision_weight) * precision_loss
    with torch.no_grad():
        edge_mask_gt = gt > gt.mean().clamp_min(1e-6)
        edge_mask_p = gp > gp.mean().clamp_min(1e-6)
        edge_recall = ((gp_near + float(margin)) >= gt).float()
        edge_precision = ((gt_near + float(margin)) >= gp).float()
        edge_recall = (edge_recall * edge_mask_gt.float()).sum() / (
            edge_mask_gt.float().sum() + 1e-6
        )
        edge_precision = (edge_precision * edge_mask_p.float()).sum() / (
            edge_mask_p.float().sum() + 1e-6
        )
    metrics = {
        "edge_recall_loss": float(recall_loss.detach().item()),
        "edge_precision_loss": float(precision_loss.detach().item()),
        "edge_recall": float(edge_recall.detach().item()),
        "edge_precision": float(edge_precision.detach().item()),
        "edge_pred_grad": float(gp.detach().mean().item()),
        "edge_gt_grad": float(gt.detach().mean().item()),
    }
    return loss, metrics

def ms_ssim(
    img1, img2, data_range=1.0, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
):
    """
    Multi-Scale SSIM (MS-SSIM) 계산
    """
    levels = len(weights)
    msssim = []

    for i in range(levels):
        ssim_val = ssim(img1, img2, data_range=data_range)
        msssim.append(ssim_val * weights[i])

        if i < levels - 1:  # Downsample for next scale
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

    return sum(msssim)



def hdr_criterion(
    output,
    target,
    use_perception=True,
    multi_level=4,
    alignment_tolerant=False,
    align_radius=1,
    align_tau=0.02,
    align_weight=1.0,
    edge_recall_weight=0.0,
):
    """
    HDR 이미지 손실 함수 - anti-collapse regularization 추가
    output, target: (B, 3, H, W) or (B, N, 3, H, W)
    alpha: L1 손실과 SSIM 손실 간의 가중치
    beta: 가중 손실의 비율

    T1: This function MUST run in FP32 for numerical stability
    All HDR operations (log/exp/tonemap) require high precision
    """

    # T1: Force FP32 for ALL HDR computations (critical for stability)
    with autocast(enabled=False):
        # Ensure inputs are FP32
        output = output.float()
        target = target.float()

        if len(output.shape) == 5:
            b, n = output.shape[:2]
            output = output.reshape(-1, *output.shape[2:])
            if len(target.shape) == 5:
                target = target.reshape(-1, *target.shape[2:])
            else:
                target = target[:, None].expand(b, n, *target.shape[1:])
                target = target.reshape(-1, *target.shape[2:])

        # FIX-ULTRA-DARK: Allow very dark HDR values (GT has values down to 1e-10 or lower)
        # Old: clamp(1e-10, 20.0) prevented learning ultra-dark regions
        # New: Only prevent exact zeros and extreme values
        # This allows model to learn full HDR range including deep shadows
        output = torch.clamp(
            output, min=1e-12, max=20.0
        )  # Lower bound: 1e-12 instead of 1e-10
        target = torch.clamp(target, min=1e-12, max=20.0)  # Preserve more dynamic range

        # Reduced anti-collapse regularization
        # output_mean = output.mean()
        # if output_mean < 1e-10:
        #     zero_penalty = F.l1_loss(output, torch.full_like(output, 0.02))
        #     return zero_penalty * 10.0, {
        #         "l1_loss_br": 0.0,
        #         "l1_loss_hdr": 0.0,
        #         "perceptual_br": 0.0,
        #         "zero_penalty": zero_penalty.item(),
        #         "diversity_loss": 0.0,
        #     }

        output_tm_br = tonemap_mu_law(output, mu=50000)  # Reduced mu for stability
        with torch.no_grad():
            target_tm_br = tonemap_mu_law(target, mu=50000)

        # Tone-Mapped L1 Loss - primary loss. In alignment-tolerant mode,
        # keep a small exact term but let local softmin carry the sharp edge
        # supervision under sub-pixel/one-pixel GT alignment noise.
        l1_loss_br_exact = F.l1_loss(output_tm_br, target_tm_br)
        align_loss_br = torch.tensor(0.0, device=output.device)
        if alignment_tolerant and align_weight > 0:
            align_loss_br = local_softmin_l1(
                output_tm_br, target_tm_br, radius=align_radius, tau=align_tau
            )
            l1_loss_br = 0.35 * l1_loss_br_exact + float(align_weight) * align_loss_br
        else:
            l1_loss_br = l1_loss_br_exact

        # SSIM loss - reduced weight to prevent gradient issues
        ssim_loss = 1 - ms_ssim(output_tm_br, target_tm_br)

        # HDR space L1 loss - reduced weight
        l1_loss_hdr = F.l1_loss(output, target) * 0.1

        # output_tm_md = mertens_merge_differentiable(output, num_exposures=13, step=2, gamma=2.2)
        # with torch.no_grad():
        #     target_tm_md = mertens_merge_differentiable(target, num_exposures=13, step=2, gamma=2.2)
        output_tm_md = torch.concat([
            tonemap_mu_law(output , mu=1000 * (10 ** i)) for i in range(multi_level)
        ])
        with torch.no_grad():
            target_tm_md = torch.concat([
                tonemap_mu_law(target , mu=1000 * (10 ** i)) for i in range(multi_level)
            ])

        l1_loss_md_exact = F.l1_loss(output_tm_md, target_tm_md)
        align_loss_md = torch.tensor(0.0, device=output.device)
        if alignment_tolerant and align_weight > 0:
            align_loss_md = local_softmin_l1(
                output_tm_md, target_tm_md, radius=align_radius, tau=align_tau
            )
            l1_loss_md = 0.25 * l1_loss_md_exact + 0.75 * float(align_weight) * align_loss_md
        else:
            l1_loss_md = l1_loss_md_exact
        ssim_loss_md = 1 - ms_ssim(output_tm_md, target_tm_md)
        
        # Gradient L1 (edge alignment)
        gx1, gy1 = sobel_grad(output_tm_md); gx2, gy2 = sobel_grad(target_tm_md)
        l_grad = (gx1-gx2).abs().mean() + (gy1-gy2).abs().mean()
        l_edge = torch.tensor(0.0, device=output.device)
        edge_metrics = {}
        if edge_recall_weight > 0:
            l_edge, edge_metrics = edge_recall_loss(
                output_tm_br,
                target_tm_br,
                radius=align_radius,
                recall_weight=1.0,
                precision_weight=0.2,
            )

        # Wide-DR fidelity: multi-mu tonemaps are good perceptual proxies, but
        # they can under-report multiplicative errors in deep shadows and exact
        # absolute errors in highlights. Add light-weight log/tail terms so the
        # extremes remain visible to both training and TensorBoard.
        log_output = torch.log(output + 1e-12)
        log_target = torch.log(target + 1e-12)
        log_abs = (log_output - log_target).abs()
        l_log = log_abs.mean()
        with torch.no_grad():
            lum = (
                0.2126 * target[:, 0:1]
                + 0.7152 * target[:, 1:2]
                + 0.0722 * target[:, 2:3]
            )
            flat_lum = lum.reshape(lum.shape[0], -1)
            q_dark = torch.quantile(flat_lum, 0.05, dim=1).view(-1, 1, 1, 1)
            q_bright = torch.quantile(flat_lum, 0.95, dim=1).view(-1, 1, 1, 1)
            dark_mask = (lum <= q_dark).float()
            bright_mask = (lum >= q_bright).float()
        l_log_dark = (log_abs * dark_mask).sum() / (dark_mask.sum() * output.shape[1] + 1e-6)
        l_lin_bright = ((output - target).abs() * bright_mask).sum() / (
            bright_mask.sum() * output.shape[1] + 1e-6
        )

        # Standard supervision-focused loss
        loss = (
            l1_loss_br * 1.0  # Strong supervision
            + l1_loss_hdr * 1.0  # Strong HDR supervision
            + ssim_loss * 3.0  # Moderate SSIM
            + l1_loss_md * 0.5
            + ssim_loss_md * 3.0
            + l_grad * (1.0 if alignment_tolerant else 5.0)
            + l_edge * float(edge_recall_weight)
            + l_log * 0.03
            + l_log_dark * 0.05
            + l_lin_bright * 0.02
        )

        metric = {
            "l1_loss_br": l1_loss_br.item(),
            "l1_loss_br_exact": l1_loss_br_exact.item(),
            "align_loss_br": align_loss_br.item(),
            "l1_loss_hdr": l1_loss_hdr.item(),
            "ssim_loss": ssim_loss.item(),
            "ssim_loss_md": ssim_loss_md.item(),
            "l1_loss_md": l1_loss_md.item(),
            "l1_loss_md_exact": l1_loss_md_exact.item(),
            "align_loss_md": align_loss_md.item(),
            "l_grad": l_grad.item(),
            "edge_loss": l_edge.item(),
            "log_loss": l_log.item(),
            "log_dark_loss": l_log_dark.item(),
            "lin_bright_loss": l_lin_bright.item(),
        }
        metric.update(edge_metrics)
        if use_perception:
            # Perceptual loss - much reduced weight
            perceptual_loss = get_perceptual_loss(output_tm_br.device)
            perceptual_br = perceptual_loss(output_tm_br, target_tm_br)
            loss = loss + perceptual_br * 0.05
            metric["perceptual_br"] = perceptual_br.item()
        return loss, metric
    
    


def confidence_validity_loss(
    D0_half,  # [B,N,1,Hh,Wh]
    gt_depths_in_half,  # [B,N,1,Hh,Wh]
    C,  # [B,N,1,Hh,Wh], sigmoid 확률(0~1)
    *,
    rel_thr=0.15,
    abs_thr=0.20,
    eps=1e-6,
    use_soft_target=True,
    soft_k=6.0,
    focal_gamma=0.0,
    reg_entropy_weight=0.05,
    neutral_target=0.5,
    label_smooth=1e-4,
    budget_target=0.5,
    budget_weight=0.3,
    always_entropy=True,
    view_diversity_weight=0.05,
    saturation_threshold=0.85,
    saturation_weight=0.3,
):
    # ---------- 0) Sanitize inputs ----------
    D0 = torch.nan_to_num(D0_half, nan=0.0, posinf=0.0, neginf=0.0)
    GT = torch.nan_to_num(gt_depths_in_half, nan=0.0, posinf=0.0, neginf=0.0)
    C = torch.nan_to_num(C, nan=0.0, posinf=1.0, neginf=0.0)

    # finite + GT>0 만 학습에 사용
    # resolution half
    B, M, _, _, _  = D0.shape
    D0 = D0.reshape(B*M, 1, D0.shape[3], D0.shape[4])
    GT = GT.reshape(B*M, 1, GT.shape[3], GT.shape[4])
    D0 = F.interpolate(D0, scale_factor=0.5, mode="bilinear", align_corners=False)
    GT = F.interpolate(GT, scale_factor=0.5, mode="bilinear", align_corners=False)
    C = C.reshape(B*M, 1, C.shape[3], C.shape[4])
    
    finite_mask = torch.isfinite(D0) & torch.isfinite(GT) & torch.isfinite(C)
    valid = (GT > 0) & finite_mask
    valid = valid.float()
    valid_cnt = valid.sum()

    # ---------- 1) Target 만들기 ----------
    abs_err = (D0 - GT).abs()
    rel_err = abs_err / GT.clamp_min(1e-3)

    if use_soft_target:
        t_rel = torch.exp(-soft_k * rel_err).clamp(0.0, 1.0)
        t_abs = torch.exp(-soft_k * (abs_err / abs_thr)).clamp(0.0, 1.0)
        tgt = 1.0 - (1.0 - t_rel) * (1.0 - t_abs)
        tgt = torch.where(abs_err > 4.0 * abs_thr, torch.zeros_like(tgt), tgt)
    else:
        tgt = ((rel_err < rel_thr) | (abs_err < abs_thr)).float()

    # invalid → 중립, 그리고 sanitize
    tgt = torch.where(valid > 0.5, tgt, torch.full_like(tgt, neutral_target))
    tgt = torch.nan_to_num(tgt, nan=neutral_target, posinf=0.0, neginf=0.0).clamp(
        0.0, 1.0
    )

    # ---------- 2) 동적 클래스 가중치 (valid만 집계) ----------
    pos_cnt = (tgt * valid).sum().detach()
    neg_cnt = ((1.0 - tgt) * valid).sum().detach()
    total = (pos_cnt + neg_cnt).clamp_min(1.0)
    w_pos = (neg_cnt / total).clamp_min(0.05)
    w_neg = (pos_cnt / total).clamp_min(0.05)

    # ---------- 3) 안정적인 BCE(+focal) ----------
    # label smoothing + clamp 로 log(0) 방지
    if label_smooth > 0:
        C = (1.0 - label_smooth) * C + label_smooth * 0.5
    C = C.clamp(eps, 1.0 - eps)

    # --- 안전한 log 준비 (FP32) ---
    C32 = C.float()  # AMP라도 로그는 FP32로
    C32 = torch.nan_to_num(C32, nan=0.5, posinf=1.0, neginf=0.0)
    C32 = C32.clamp(eps, 1.0 - eps)  # 반드시 min/max 둘 다 클램프

    # 기본 경로
    logC = torch.log(C32)  # log(C)
    log1mC = torch.log1p(-C32)  # log(1 - C)

    # 만약 수치 이슈가 남아있다면(드물지만) 대체 경로로 강제 교체
    bad = ~torch.isfinite(logC) | ~torch.isfinite(log1mC)
    if bad.any():
        # 1-C 계산을 먼저 하고 clamp → log 로 가는 더 보수적인 경로
        one_minus_C = (1.0 - C32).clamp(eps, 1.0)
        logC = torch.where(bad, torch.log(C32), logC)
        log1mC = torch.where(bad, torch.log(one_minus_C), log1mC)

    # 이후 BCE
    bce = -(tgt * logC + (1.0 - tgt) * log1mC)

    # 필요하면 원래 dtype으로
    bce = bce.to(C.dtype)

    if focal_gamma > 0.0:
        pt = torch.where(tgt > 0.5, C, 1.0 - C)
        bce = bce * (1.0 - pt).pow(focal_gamma)

    class_w = torch.where(tgt > 0.5, w_pos, w_neg)
    loss_map = bce * class_w * valid

    # 비정상값 zero-out (디버깅 로그 남기고 진행)
    if not torch.isfinite(loss_map).all():
        # print("Warning: Non-finite values in loss_map")  # 필요하면 로깅
        loss_map = torch.where(
            torch.isfinite(loss_map), loss_map, torch.zeros_like(loss_map)
        )

    # ---------- 4) 리덕션 & fallback ----------
    if valid_cnt > 0:
        loss = loss_map.sum() / valid_cnt
    else:
        # 유효픽셀 전무 → 약한 엔트로피 정규화(그래프 유지 + DDP 동기화 보장)
        Ce = C.clamp(eps, 1.0 - eps)
        entropy = -(Ce * torch.log(Ce) + (1.0 - Ce) * torch.log1p(-Ce))
        # 그래프 경로 유지(제로 그라드): 0과의 곱/합으로 구성
        loss = reg_entropy_weight * entropy.mean()

    # ---------- 4b) Anti-collapse regularizers (always active) ----------
    # Why: when D0 quality is similar across views, C cancels in per-view normalization
    # of w_base = exp(-α·rerr)·C·valid_geo, leaving little gradient pressure on C.
    # The all-1 solution is locally optimal; explicit regularizers are needed.
    Ce_full = C.clamp(eps, 1.0 - eps)
    if always_entropy and reg_entropy_weight > 0.0:
        ent_full = -(Ce_full * torch.log(Ce_full) + (1.0 - Ce_full) * torch.log1p(-Ce_full))
        loss = loss + reg_entropy_weight * ent_full.mean()

    if budget_weight > 0.0:
        # Per-pixel budget: penalize each pixel's |C - 0.5| (asymmetric on the
        # saturation side). Why per-pixel: a global mean-only term has gradient
        # 1/numel per pixel, which is negligible — verified empirically that C
        # stayed at 1.0000 / view_std=0.0000 with the global form. The per-pixel
        # form delivers a real per-location push.
        loss = loss + budget_weight * (Ce_full - budget_target).pow(2).mean()

    if saturation_weight > 0.0:
        # Asymmetric anti-saturation: only penalize when C exceeds the threshold
        # (default 0.85). Lets confident pixels stay confident up to that point
        # while strongly punishing trivial collapse to 1.
        over = (Ce_full - saturation_threshold).clamp_min(0.0)
        loss = loss + saturation_weight * over.pow(2).mean()

    if view_diversity_weight > 0.0:
        # Reward per-pixel variance of C across N source views — collapse-to-1
        # gives zero variance, so this directly counters it. C was reshaped to
        # [B*M, 1, Hh', Wh'] earlier (line 464); recover [B, M, 1, Hh', Wh'] for
        # var-across-views.
        try:
            C5 = Ce_full.view(B, M, 1, Ce_full.shape[-2], Ce_full.shape[-1])
            var_n = C5.var(dim=1, unbiased=False).mean()
            loss = loss + view_diversity_weight * (-var_n)
        except Exception:
            pass  # shape mismatch fallback — skip silently

    # 최종 가드
    if not torch.isfinite(loss):
        # 그래프를 유지하면서 0 스칼라를 만들기
        loss = (C * 0.0).sum()

    # ---------- 5) 메트릭 (valid 위치에서만) ----------
    with torch.no_grad():
        C_bin = (C > 0.5).float()
        tgt_bin = (tgt > 0.5).float()

        if valid_cnt > 0:
            tp = (C_bin * tgt_bin * valid).sum()
            fp = (C_bin * (1.0 - tgt_bin) * valid).sum()
            fn = (((1.0 - C_bin) * tgt_bin) * valid).sum()
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            acc = ((C_bin == tgt_bin).float() * valid).sum() / valid_cnt
            pos_ratio = (tgt * valid).sum() / valid_cnt
        else:
            precision = recall = f1 = acc = pos_ratio = torch.tensor(
                0.0, device=C.device
            )

        metrics = {
            "conf_loss": float(loss.detach().item()),
            "conf_acc": float(acc.detach().item()),
            "conf_precision": float(precision.detach().item()),
            "conf_recall": float(recall.detach().item()),
            "conf_f1": float(f1.detach().item()),
            "conf_pos_ratio": float(pos_ratio.detach().item()),
            "conf_valid_count": int(valid_cnt.detach().item()),
        }

    return loss, metrics
