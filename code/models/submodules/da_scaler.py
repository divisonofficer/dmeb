import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_stats(x, mask, eps=1e-6):
    # x, mask: (B,1,H,W). returns mean,p90 as scalars per-sample
    B = x.shape[0]
    flat = x.view(B, -1)
    mflat = mask.view(B, -1).float()
    cnt = mflat.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (flat * mflat).sum(dim=1, keepdim=True) / cnt
    # p90 with masked sort
    xmasked = torch.where(mflat > 0, flat, torch.full_like(flat, -1e9))
    q_idx = (cnt.long().clamp_min(1) - 1).squeeze(1)  # ~max for stability
    topk = torch.topk(xmasked, k=1, dim=1).values  # crude approx p~100
    p90 = topk  # simple surrogate; can implement true masked quantile if needed
    return mean.squeeze(1), p90.squeeze(1).clamp_min(1e-3)


class ResidualScaleHead(nn.Module):
    def __init__(self, in_ch=3, feat=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, feat, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(feat, feat, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(feat + 6, 64), nn.ReLU(True), nn.Linear(64, 2)  # Δa, Δb
        )

    def forward(self, rel_disp, sparse_u, mask, a0, b0):
        B, _, H, W = rel_disp.shape
        x = torch.cat([rel_disp, sparse_u, mask], dim=1)  # (B,3,H,W)
        f = self.enc(x).view(B, -1)  # (B,feat)
        # global masked stats as aux features
        m = (mask > 0).float()
        rd_mean, rd_scale = masked_stats(rel_disp.abs(), torch.ones_like(m))
        su_mean, _ = masked_stats(sparse_u.abs(), m)
        valid_ratio = m.view(B, -1).float().mean(dim=1, keepdim=False)
        ab0 = torch.stack([a0, b0], dim=1)
        aux = torch.stack([rd_mean, rd_scale, su_mean, valid_ratio], dim=1)
        g = torch.cat([f, ab0, aux], dim=1)
        d_ab = self.mlp(g)  # (B,2)

        # CRITICAL: Clamp residuals to prevent gradient explosion
        # During early training, unconstrained MLPs can output extreme values
        da = torch.clamp(d_ab[:, 0], -2.0, 2.0)  # Limit residual magnitude
        db = torch.clamp(d_ab[:, 1], -2.0, 2.0)

        return da, db


class SoftRANSACScaler:
    """
    Soft RANSAC-based scale estimation (no learnable parameters)
    Estimates affine transformation: disparity = a * rel_disp + b
    where disparity = 1/depth
    """

    @staticmethod
    def estimate_scale(
        rel_disp, sparse_depth, mask, n_iterations=100, inlier_threshold=0.1, eps=1e-6
    ):
        """
        rel_disp:   (B,1,H,W)   -- relative disparity from backbone
        sparse_depth:(B,1,H,W)  -- absolute depth from sensor (0=invalid)
        mask:       (B,1,H,W)   -- bool/0-1 valid mask
        n_iterations: number of RANSAC iterations
        inlier_threshold: threshold for inlier classification (in disparity space)

        Returns: metric_depth (B,1,H,W)
        """
        B, _, H, W = rel_disp.shape
        device = rel_disp.device

        # Normalize rel_disp for stability
        rd_abs = rel_disp.abs().view(B, -1)
        scale_norm = (
            torch.quantile(rd_abs, q=0.9, dim=1).clamp_min(1e-3).view(B, 1, 1, 1)
        )
        rel_disp_norm = rel_disp / scale_norm

        metric_depth_batch = []

        for batch_idx in range(B):
            # Get valid sparse points
            valid_mask = mask[batch_idx, 0] > 0
            if valid_mask.sum() < 10:
                # Not enough points - use simple median scaling
                valid_depths = sparse_depth[batch_idx, 0][valid_mask]
                valid_rel_disp = rel_disp_norm[batch_idx, 0][valid_mask]
                if len(valid_depths) > 0:
                    median_depth = valid_depths.median()
                    median_rel_disp = valid_rel_disp.abs().median()
                    scale_a = 1.0 / (median_depth * median_rel_disp.clamp_min(1e-3))
                    scale_b = 0.0
                else:
                    scale_a = 1.0
                    scale_b = 0.0
            else:
                # Soft RANSAC
                valid_depths = sparse_depth[batch_idx, 0][valid_mask]  # (N,)
                valid_rel_disp = rel_disp_norm[batch_idx, 0][valid_mask]  # (N,)
                valid_disparity = 1.0 / valid_depths.clamp_min(1e-6)  # (N,)

                N = len(valid_depths)
                best_score = -1
                best_a, best_b = 1.0, 0.0

                for _ in range(n_iterations):
                    # Randomly sample 2 points
                    if N < 2:
                        break
                    idx = torch.randperm(N, device=device)[:2]
                    rd1, rd2 = valid_rel_disp[idx[0]], valid_rel_disp[idx[1]]
                    u1, u2 = valid_disparity[idx[0]], valid_disparity[idx[1]]

                    # Solve for a, b: u = a*rd + b
                    # u1 = a*rd1 + b
                    # u2 = a*rd2 + b
                    denom = (rd1 - rd2).abs()
                    if denom < 1e-6:
                        continue

                    a_candidate = (u1 - u2) / (rd1 - rd2)
                    b_candidate = u1 - a_candidate * rd1

                    # Compute residuals
                    predicted_u = a_candidate * valid_rel_disp + b_candidate
                    residuals = (predicted_u - valid_disparity).abs()

                    # Soft inlier scoring with exponential weights
                    weights = torch.exp(-residuals / inlier_threshold)
                    score = weights.sum().item()

                    if score > best_score:
                        best_score = score
                        best_a = a_candidate.item()
                        best_b = b_candidate.item()

                scale_a, scale_b = best_a, best_b

            # Clamp to reasonable range
            scale_a = max(0.001, min(100.0, scale_a))
            scale_b = max(-10.0, min(10.0, scale_b))

            # Convert back to original scale
            scale_a = scale_a / scale_norm[batch_idx, 0, 0, 0].item()

            # Compute metric depth for entire image
            disparity = scale_a * rel_disp[batch_idx, 0] + scale_b
            disparity = disparity.clamp_min(1e-6)
            metric_depth = 1.0 / disparity
            metric_depth_batch.append(metric_depth)

        metric_depth_batch = torch.stack(metric_depth_batch).unsqueeze(
            1
        )  # (B, 1, H, W)
        return metric_depth_batch


class ScaleNet(nn.Module):
    def __init__(self, allow_spatial_affine=False):
        super().__init__()
        self.head = ResidualScaleHead(in_ch=3, feat=32)
        self.allow_spatial = allow_spatial_affine
        if allow_spatial_affine:
            # 저차 공간 보정: a(x,y)=a+ax*xn+ay*yn (b도 동일)
            self.axay = nn.Parameter(torch.zeros(4))  # ax, ay, bx, by

    @staticmethod
    def ls_init(rel_disp, sparse_depth, mask, eps=1e-6):
        """
        rel_disp:   (B,1,H,W)   -- relative disparity \hat{u}
        sparse_depth:(B,1,H,W)  -- absolute depth z (0=invalid)
        mask:       (B,1,H,W)   -- bool/0-1 valid mask (depth>0)
        returns: a0 (B,), b0 (B,)
        회귀모형: u = 1/z ≈ a * \hat{u} + b
        """
        B, _, H, W = rel_disp.shape
        m = mask[:, 0].float()  # (B,H,W)
        uhat = rel_disp[:, 0]  # (B,H,W)
        z = sparse_depth[:, 0].clamp_min(1e-6)  # (B,H,W)
        u = 1.0 / z  # (B,H,W)

        # 유효 포인트 수
        S1 = m.view(B, -1).sum(dim=1)  # (B,)
        has_enough = S1 >= 10  # Increased from 2 to 10 for more stable LS

        # 마스크된 합산(정규방정식 항)
        Sx = (m * uhat).view(B, -1).sum(dim=1)  # \sum \hat{u}
        Sy = (m * u).view(B, -1).sum(dim=1)  # \sum u
        Sxx = (m * uhat * uhat).view(B, -1).sum(dim=1)  # \sum \hat{u}^2
        Sxy = (m * uhat * u).view(B, -1).sum(dim=1)  # \sum \hat{u} u

        # a = (S1*Sxy - Sx*Sy) / (S1*Sxx - Sx^2)
        denom = S1 * Sxx - Sx * Sx
        a0 = (S1 * Sxy - Sx * Sy) / denom.clamp_min(eps)
        b0 = (Sy - a0 * Sx) / S1.clamp_min(eps)

        # Better fallback: use median-based initialization
        # If not enough points, estimate scale from available data
        fallback_a = torch.ones_like(a0)
        fallback_b = torch.zeros_like(b0)

        for i in range(B):
            if not has_enough[i] and S1[i] >= 1:
                # Use median of sparse depths for better initialization
                valid_depths = sparse_depth[i, 0][mask[i, 0] > 0]
                valid_disps = rel_disp[i, 0][mask[i, 0] > 0]
                if len(valid_depths) > 0:
                    median_depth = valid_depths.median()
                    median_disp = valid_disps.median()
                    # Rough estimate: a ≈ 1/(median_depth * median_disp)
                    fallback_a[i] = 1.0 / (
                        median_depth * median_disp.abs().clamp_min(1e-3)
                    )
                    fallback_b[i] = 0.0

        a0 = torch.where(has_enough, a0, fallback_a)
        b0 = torch.where(has_enough, b0, fallback_b)

        # Clamp to reasonable range to avoid extreme values
        a0 = torch.clamp(a0, 0.001, 100.0)
        b0 = torch.clamp(b0, -10.0, 10.0)

        return a0, b0

    def forward(self, rel_disp, sparse_depth, mask):
        """
        rel_disp: (B,1,H,W)  [relative disparity]
        sparse_depth: (B,1,H,W)  [absolute depth; 0 invalid]
        mask: (B,1,H,W)  bool/float {0,1}
        returns: metric_depth (B,1,H,W), a, b
        """
        # T1: Force FP32 for quantile operation (required by torch.quantile)
        with torch.cuda.amp.autocast(enabled=False):
            rel_disp = rel_disp.float()
            sparse_depth = sparse_depth.float()
            mask = mask.float()

            # normalize inputs (per-sample)
            B, _, H, W = rel_disp.shape
            # p90로 scale normalize
            rd_abs = rel_disp.abs().view(B, -1)
            s = torch.quantile(rd_abs, q=0.9, dim=1).clamp_min(1e-3).view(B, 1, 1, 1)
            rd = rel_disp / s

            sparse_u = torch.zeros_like(sparse_depth)
            m = mask > 0
            sparse_u[m] = (1.0 / sparse_depth.clamp_min(1e-6))[m]

            # LS init (stop-grad)
            a0, b0 = self.ls_init(
                rd.detach(), sparse_depth.detach(), m.detach()
            )  # init on normalized rel_disp

            # Safety check: ensure a0 and b0 are reasonable
            # Extreme values here will cause gradient explosion
            a0 = torch.clamp(a0, 0.01, 10.0)  # More conservative clamping
            b0 = torch.clamp(b0, -5.0, 5.0)

            # residual head (can run in AMP, but keep in FP32 for consistency)
            da, db = self.head(rd, sparse_u, m.float(), a0, b0)
            a = a0 + da
            b = b0 + db

            # Clamp final a and b to prevent extreme disparity values
            a = torch.clamp(a, 0.001, 100.0)
            b = torch.clamp(b, -10.0, 10.0)

            # (선택) 저차 공간 보정
            if self.allow_spatial:
                yy = (
                    torch.linspace(-1, 1, H, device=rel_disp.device)
                    .view(1, 1, H, 1)
                    .expand(B, 1, H, W)
                )
                xx = (
                    torch.linspace(-1, 1, W, device=rel_disp.device)
                    .view(1, 1, 1, W)
                    .expand(B, 1, H, W)
                )
                ax, ay, bx, by = self.axay
                A_map = a.view(B, 1, 1, 1) + ax * xx + ay * yy
                B_map = b.view(B, 1, 1, 1) + bx * xx + by * yy
            else:
                A_map = a.view(B, 1, 1, 1)
                B_map = b.view(B, 1, 1, 1)

            # 역정규화: 실제 disparity = A*(rel_disp/s) + B  => A' = A/s, B'=B
            A_map = A_map / s
            disp = (A_map * rel_disp + B_map).clamp_min(1e-6)
            metric_depth = 1.0 / disp

            # Safety check: clamp metric_depth to reasonable range (0.1m ~ 1000m)
            # This prevents infinity/NaN from causing training instability
            metric_depth = torch.clamp(metric_depth, 0.1, 1000.0)

            return metric_depth  # , a, b
