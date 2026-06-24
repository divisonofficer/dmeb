# flow_estimator.py
# [FLOW] Optical flow estimation for depth-free alignment
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_ch, out_ch, k=3, s=1, p=1):
    """Simple conv block with LeakyReLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p),
        nn.LeakyReLU(0.1, inplace=True)
    )


class FlowEstimatorTiny(nn.Module):
    """
    PWC-lite style minimal flow estimator.
    
    Input: concat([I_src, I_tgt]) in [B,6,H,W] (RGB-RGB, linear recommended)
    Output: flow in pixels [B,2,H,W] (u,v)
    
    Single-scale architecture (coarse->fine replaced with single scale)
    for simplicity and speed. Fine-tuning during training recommended.
    """
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        self.f1 = conv(in_ch, base, 3, 1, 1)
        self.f2 = conv(base, base, 3, 2, 1)        # 1/2
        self.f3 = conv(base, base*2, 3, 2, 1)      # 1/4
        self.f4 = conv(base*2, base*2, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.head = nn.Conv2d(base*2, 2, 3, 1, 1)
        
        # Initialize head to produce near-zero flow
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, src, tgt):
        """
        Args:
            src: [B,3,H,W] source image (linear)
            tgt: [B,3,H,W] target image (linear)
        
        Returns:
            flow: [B,2,H,W] optical flow in pixels (u,v)
        """
        x = torch.cat([src, tgt], dim=1)  # [B,6,H,W]
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        flow_low = self.head(x)            # [B,2,H/4,W/4] (pixels @low scale)
        flow = self.up(flow_low)           # [B,2,H,W]     (pixels @full)
        return flow


def flow_to_grid(flow):
    """
    Convert flow (pixels) to normalized grid ([-1,1]).
    
    Args:
        flow: [B,2,H,W] optical flow where:
            - flow[:,0,:,:] = u (x direction, horizontal)
            - flow[:,1,:,:] = v (y direction, vertical)
    
    Returns:
        grid: [B,H,W,2] normalized coordinates in [-1,1] range
    """
    B, _, H, W = flow.shape
    
    # Create base grid: [-1,1] normalized coordinates
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow.device),
        torch.linspace(-1, 1, W, device=flow.device),
        indexing="ij"
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, H, W, 2)  # [B,H,W,2]
    
    # Convert flow(px) to [-1,1] scale delta
    # dx_norm = 2 * (u / (W-1))
    # dy_norm = 2 * (v / (H-1))
    dx = 2.0 * (flow[:, 0] / max(W-1, 1)).unsqueeze(-1)  # [B,H,W,1]
    dy = 2.0 * (flow[:, 1] / max(H-1, 1)).unsqueeze(-1)  # [B,H,W,1]
    delta = torch.cat([dx, dy], dim=-1)                   # [B,H,W,2]
    
    grid = base + delta
    return grid


def warp_with_flow(img, flow):
    """
    Warp image according to optical flow.
    
    Args:
        img: [B,3,H,W] or [B,C,H,W] input image
        flow: [B,2,H,W] optical flow in pixels
    
    Returns:
        warped: [B,C,H,W] warped image
        valid: [B,1,H,W] validity mask (1 if grid in [-1,1], 0 otherwise)
    """
    grid = flow_to_grid(flow)  # [B,H,W,2]
    
    # Warp using grid_sample
    warped = F.grid_sample(
        img, 
        grid, 
        mode="bilinear", 
        padding_mode="border", 
        align_corners=True
    )
    
    # Validity mask: 1 if grid is within [-1,1] range, 0 otherwise
    valid = (
        (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) &
        (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    ).float().unsqueeze(1)  # [B,1,H,W]
    
    return warped, valid
