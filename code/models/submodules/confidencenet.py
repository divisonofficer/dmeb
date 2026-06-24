import torch
import torch.nn as nn
import torch.nn.functional as F
# ===== (간단 Confidence Head) =====
class ConfidenceHead(nn.Module):
    """
    T5: Enhanced Confidence Head with merge-aware features

    Input features:
    - rgb (3): RGB appearance
    - sat (1): Saturation mask
    - pnorm (1): Prompt density/norm
    - log_consensus (1): Log-domain consensus strength
    - geo_reliability (1): Geometry reliability from reprojection
    - dark_mask (1): Dark region indicator
    - exposure (1): NEW - Exposure value (noise indicator)
    - info_sufficiency (1): NEW - SNR-based information score

    Total: 3+1+1+1+1+1+1+1 = 10 channels (was 8)
    """

    def __init__(self, in_ch=13, mid=32):  # Increased from 11 to 13 for exposure+info
        super().__init__()
        # Slightly deeper network for better expressiveness (minimal param increase)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid, mid // 2, 3, 1, 1),  # Additional layer for capacity
            nn.ReLU(),
            nn.Conv2d(mid // 2, 1, 1, 1, 0),
            nn.Sigmoid(),
        )
        # Neutral initialization (sigmoid(0)=0.5) so loss decides where C should saturate.
        # Why: bias=2.0 starts C≈0.88 everywhere; combined with weak gradient from depth fusion
        # (C cancels in per-view normalization when D0 quality is uniform across views),
        # the network has no incentive to leave the saturated region.
        with torch.no_grad():
            if hasattr(self.net[-2], "bias") and self.net[-2].bias is not None:
                self.net[-2].bias.fill_(0.0)

    def forward(
        self,
        rgb,
        sat,
        pnorm,
        log_consensus=None,
        geo_reliability=None,
        dark_mask=None,
        exposure=None,  # NEW
        info_sufficiency=None,  # NEW
    ):
        """
        T5: Forward pass with merge-aware features

        Args:
            rgb: [B,N,3,H,W] - RGB appearance
            sat: [B,N,1,H,W] - Saturation mask
            pnorm: [B,N,1,H,W] - Prompt density
            log_consensus: [B,N,1,H,W] - Log-domain consensus (optional)
            geo_reliability: [B,N,1,H,W] - Geometry reliability (optional)
            dark_mask: [B,N,1,H,W] - Dark region mask (optional)
            exposure: [B,N,1,H,W] - Exposure value (optional)
            info_sufficiency: [B,N,1,H,W] - SNR-based info score (optional)

        Returns:
            out: [B,N,1,H,W] - Confidence scores
        """
        # Build feature list
        feature_list = [rgb, sat, pnorm]  # 3+1+1 = 5 channels

        # Add optional features (with fallbacks)
        B, N, _, H, W = rgb.shape

        if log_consensus is not None:
            feature_list.append(log_consensus)
        else:
            # Fallback: use ones (no consensus info available)
            feature_list.append(torch.ones(B, N, 1, H, W, device=rgb.device))

        if geo_reliability is not None:
            feature_list.append(geo_reliability)
        else:
            # Fallback: use ones
            feature_list.append(torch.ones(B, N, 1, H, W, device=rgb.device))

        if dark_mask is not None:
            feature_list.append(dark_mask)
        else:
            # Fallback: use zeros (not dark)
            feature_list.append(torch.zeros(B, N, 1, H, W, device=rgb.device))

        # NEW: Exposure channel (normalized to ~[0,1] range)
        if exposure is not None:
            # exposure is typically [0.01, 1.0], log-normalize for better distribution
            exp_norm = torch.log(exposure.clamp_min(0.001) + 1.0) / 2.0  # ~[0, 0.35]
            feature_list.append(exp_norm)
        else:
            feature_list.append(torch.ones(B, N, 1, H, W, device=rgb.device) * 0.5)

        # NEW: Info sufficiency channel (already [0,1] from sigmoid in model)
        if info_sufficiency is not None:
            feature_list.append(info_sufficiency)
        else:
            feature_list.append(torch.ones(B, N, 1, H, W, device=rgb.device) * 0.5)

        x = torch.cat(feature_list, dim=2)  # [B,N,10,H,W]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        out = self.net(x).view(B, N, 1, H, W)  # [B,N,1,H,W]
        return out

