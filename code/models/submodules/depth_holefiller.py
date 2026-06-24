import torch
import torch.nn as nn
import torch.nn.functional as F

def smoothstep(edge0, edge1, x):
    t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1)
    return t * t * (3 - 2 * t)

class DepthHoleFiller(nn.Module):
    """
    Non-learnable LiDAR depth hole filler (no trainable params).
    Iteratively fills missing (0) pixels with neighborhood propagation.
    """
    def __init__(self, iterations=5, kernel_size=15, max_depth=100.0):
        super().__init__()
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.max_depth = max_depth

    def forward(self, depth):
        """
        Args:
            depth: [B,1,H,W] tensor with 0 for invalid pixels.
        Returns:
            dense_depth: [B,1,H,W] filled depth map.
        """
        B, _, H, W = depth.shape
        mask = (depth > 0).float()
        current = depth.clone()
        current_mask = mask.clone()

        pad = self.kernel_size // 2

        for i in range(self.iterations):
            holes = (current_mask < 0.5).float()
            if holes.sum() < 1:
                break

            # Local neighborhood statistics
            num = F.avg_pool2d(current * current_mask, self.kernel_size, 1, pad)
            den = F.avg_pool2d(current_mask, self.kernel_size, 1, pad).clamp_min(1e-6)
            avg_neighbors = num / den

            # Local maximum propagation (preserve far structure)
            max_neighbors = F.max_pool2d(
                current * current_mask + (1 - current_mask) * (-1e9),
                self.kernel_size, 1, pad
            )

            # Density-based blending: sparse → MAX, dense → AVG
            neighbor_density = den
            max_weight = 1.0 - smoothstep(0.15, 0.35, neighbor_density)
            avg_weight = 1.0 - max_weight
            filled = max_weight * max_neighbors + avg_weight * avg_neighbors

            # Update holes
            current = current * current_mask + filled * holes

            # Expand mask gradually
            current_mask = torch.clamp(current_mask + (den > 0.005).float(), 0, 1)

        return current.clamp(0, self.max_depth)
