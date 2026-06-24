import numpy as np
import cv2
import torch
import torch.distributed as dist


def depth_colorize(depth, max_depth=24, use_disparity=True, eps=1e-6):
    """
    Depth map을 컬러 이미지로 변환하여 TensorBoard용 텐서 반환

    Args:
        depth: [H, W] or [1, H, W] depth map in meters
        max_depth: maximum depth for normalization (default 24m)
        use_disparity: if True, visualize as disparity (1/depth) for better near detail
        eps: small value to avoid division by zero

    Returns:
        [3, H, W] RGB tensor for TensorBoard
    """
    depth = depth.squeeze().cpu().numpy()
    depth = np.clip(depth, eps, max_depth)

    if use_disparity:
        # Convert to disparity (1/depth) for better visualization
        # Near regions (important!) get brighter colors
        disparity = 1.0 / (depth + eps)
        # Normalize disparity: [1/max_depth, 1/eps] → [0, 255]
        disp_min = 1.0 / max_depth  # Far region
        disp_max = 0.1  # Near region
        disparity[depth < 0.1] = disp_min
        disparity_normalized = (
            (disparity.clip(0, disp_max) - disp_min) / (disp_max - disp_min + eps) * 255
        ).astype(np.uint8)
        depth_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_MAGMA)
    else:
        # Original depth visualization
        depth_normalized = (depth / max_depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

    depth_colored = depth_colored[..., ::-1]  # BGR to RGB
    # Convert to torch tensor and change from (H, W, 3) to (3, H, W)
    depth_tensor = (
        torch.from_numpy(depth_colored.copy()).permute(2, 0, 1).float() / 255.0
    )
    return depth_tensor


def safe_tensorboard_image(writer, tag, img_tensor, global_step, max_val=None):
    """TensorBoard에 안전하게 이미지를 추가하는 함수"""
    if writer is None:
        return

    # Only rank 0 should actually write images
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    try:
        # Ensure tensor is on CPU
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.cpu()

        # Handle different tensor shapes
        if len(img_tensor.shape) == 4:  # [B, C, H, W]
            img_tensor = img_tensor[0]  # Take first batch
        elif len(img_tensor.shape) == 2:  # [H, W]
            # For grayscale, keep as HW format
            pass
        elif len(img_tensor.shape) == 3:  # [C, H, W] or [H, W, C]
            if img_tensor.shape[0] == 3 or img_tensor.shape[0] == 1:  # [C, H, W]
                pass  # Already in correct format
            elif img_tensor.shape[2] == 3 or img_tensor.shape[2] == 1:  # [H, W, C]
                img_tensor = img_tensor.permute(2, 0, 1)  # Convert to [C, H, W]

        # Normalize if needed
        if max_val is not None:
            img_tensor = img_tensor / max_val

        # Clamp values to valid range
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Determine dataformats
        if len(img_tensor.shape) == 2:
            dataformats = "HW"
        elif len(img_tensor.shape) == 3:
            dataformats = "CHW"
        else:
            raise ValueError(f"Unsupported tensor shape: {img_tensor.shape}")

        writer.add_image(tag, img_tensor, global_step, dataformats=dataformats)

    except Exception as e:
        print(f"Warning: Failed to log image {tag}: {e}")
        print(
            f"Image tensor shape: {img_tensor.shape if hasattr(img_tensor, 'shape') else 'N/A'}"
        )
        print(
            f"Image tensor dtype: {img_tensor.dtype if hasattr(img_tensor, 'dtype') else 'N/A'}"
        )
