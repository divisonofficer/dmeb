import torch
import torch.nn as nn
import torch.nn.functional as F

class RANSACScaleAligner(nn.Module):
    """
    Pure non-learnable scale alignment module.
    Aligns monocular relative disparity to LiDAR metric depth using Soft-RANSAC.
    No trainable parameters.
    """

    def __init__(self, n_iterations=100, inlier_threshold=0.1):
        super().__init__()
        self.n_iterations = n_iterations
        self.inlier_threshold = inlier_threshold

    @torch.no_grad()
    def forward(self, rel_disp, sparse_depth, mask):
        """
        Args:
            rel_disp: (B,1,H,W) relative disparity (1/depth-like output from monocular network)
            sparse_depth: (B,1,H,W) sparse LiDAR depth (0=invalid)
            mask: (B,1,H,W) binary mask of valid LiDAR points
        Returns:
            metric_depth: (B,1,H,W) scaled depth aligned to metric scale
            a_list, b_list: estimated global affine coefficients per batch
        """
        B, _, H, W = rel_disp.shape
        device = rel_disp.device

        metric_depths = []
        a_list, b_list = [], []

        for i in range(B):
            valid_mask = mask[i, 0] > 0
            if valid_mask.sum() < 5:
                # fallback: simple median scaling
                rd = rel_disp[i, 0]
                sd = sparse_depth[i, 0]
                if valid_mask.sum() == 0:
                    metric_depths.append((1.0 / rd.clamp_min(1e-6)).unsqueeze(0))
                    a_list.append(torch.tensor(1.0, device=device))
                    b_list.append(torch.tensor(0.0, device=device))
                    continue
                valid_depths = sd[valid_mask]
                valid_disp = rd[valid_mask]
                median_d = valid_depths.median()
                median_rd = valid_disp.abs().median()
                a_est = 1.0 / (median_d * median_rd.clamp_min(1e-3))
                b_est = 0.0
            else:
                # Soft RANSAC
                valid_depths = sparse_depth[i, 0][valid_mask]  # (N,)
                valid_rel_disp = rel_disp[i, 0][valid_mask]    # (N,)
                valid_disp = 1.0 / valid_depths.clamp_min(1e-6)

                N = len(valid_depths)
                best_score = -1.0
                best_a, best_b = 1.0, 0.0

                for _ in range(self.n_iterations):
                    if N < 2:
                        break
                    idx = torch.randperm(N, device=device)[:2]
                    x1, x2 = valid_rel_disp[idx]
                    y1, y2 = valid_disp[idx]

                    if abs(x1 - x2) < 1e-6:
                        continue

                    a_cand = (y1 - y2) / (x1 - x2)
                    b_cand = y1 - a_cand * x1

                    # residuals in disparity space
                    pred_y = a_cand * valid_rel_disp + b_cand
                    residuals = (pred_y - valid_disp).abs()
                    weights = torch.exp(-residuals / self.inlier_threshold)
                    score = weights.sum().item()

                    if score > best_score:
                        best_score = score
                        best_a, best_b = a_cand.item(), b_cand.item()

                a_est, b_est = best_a, best_b

            # Clamp to safe range
            a_est = max(0.001, min(100.0, a_est))
            b_est = max(-10.0, min(10.0, b_est))

            # Compute full metric depth
            disp = (a_est * rel_disp[i, 0] + b_est).clamp_min(1e-6)
            metric_depth = 1.0 / disp
            metric_depth = metric_depth.clamp(0.1, 1000.0)

            metric_depths.append(metric_depth.unsqueeze(0))
            a_list.append(torch.tensor(a_est, device=device))
            b_list.append(torch.tensor(b_est, device=device))

        metric_depths = torch.stack(metric_depths, dim=0)
        a_list = torch.stack(a_list)
        b_list = torch.stack(b_list)
        return metric_depths#, a_list, b_list
