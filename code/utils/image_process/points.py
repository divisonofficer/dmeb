from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def transform_points(points: np.ndarray, transform_mtx: np.ndarray):
    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: Transformed points
    """
    points = points.reshape(-1, 3)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = transform_mtx @ points.T
    return points[:3].T


def transform_point_inverse(points: np.ndarray, transform_mtx: np.ndarray):
    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns
        np.ndarray: Transformed points
    """
    transform_mtx = np.linalg.pinv(transform_mtx)
    return transform_points(points, transform_mtx)


def lidar_points_to_disparity_with_cal(
    points: np.ndarray,
    transform_mtx: np.ndarray,
    calibration_dict: dict,
    points_scale=1000,
):
    points = points.reshape(-1, 3) * points_scale
    fx = calibration_dict["mtx_left"][0, 0]
    cx = calibration_dict["mtx_left"][0, 2]
    cx_r = calibration_dict["mtx_right"][0, 2]
    cy = calibration_dict["mtx_left"][1, 2]
    baseline = np.linalg.norm(calibration_dict["T"])

    points = transform_point_inverse(points, transform_mtx)

    w = int(round(cx / 360) * 720)
    h = int(round(cy / 270) * 540)

    cx = round(cx / 360) * 360
    cy = round(cy / 270) * 270

    print(w, h, cx, cy)

    points = project_points_on_camera(points, fx, cx, cy, w, h)
    points[:, 2] = fx * baseline / points[:, 2]
    return points


def depth_points_to_disparity_with_cal(
    points: np.ndarray,
    calibration_dict: dict,
):
    fx = calibration_dict["mtx_left"][0, 0]
    baseline = np.linalg.norm(calibration_dict["T"])
    points[:, 2] = (
        fx * baseline / points[:, 2]
        + calibration_dict["mtx_left"][0, 2]
        - calibration_dict["mtx_right"][0, 2]
    )
    return points


def disparity_points_to_depth_with_cal(
    points: Union[np.ndarray, torch.Tensor],
    calibration_dict: dict,
    width=720,
    height=540,
):
    points = points[
        (points[:, 0] < width)
        & (points[:, 1] < height)
        & (points[:, 2] > 0)
        & (points[:, 0] >= 0)
        & (points[:, 1] >= 0)
    ]
    fx = calibration_dict["mtx_left"][0, 0]
    baseline = np.linalg.norm(calibration_dict["T"])

    points[:, 2] = (
        fx
        * baseline
        / (
            points[:, 2]
            - (calibration_dict["mtx_left"][0, 2] - calibration_dict["mtx_right"][0, 2])
        )
    )
    return points


def refine_disparity_points(points: torch.Tensor, thresh_dist=0.5, thresh_disp=0.85):
    # u, v, d 좌표 분리
    u = points[:, 0]
    v = points[:, 1]
    d = points[:, 2]

    # 거리 계산 함수
    def calculate_distances(u, v):
        # (N, 1) - (1, N) 으로 브로드캐스팅하여 모든 쌍의 유클리드 거리 계산
        dist_u = u.unsqueeze(1) - u.unsqueeze(0)
        dist_v = v.unsqueeze(1) - v.unsqueeze(0)
        distances = torch.sqrt(dist_u**2 + dist_v**2)
        return distances

    # 거리 행렬 계산
    distances = calculate_distances(u, v)

    # 각 포인트의 거리 d 내에서 다른 포인트 찾기
    mask = distances <= d.unsqueeze(1) * thresh_dist

    # 2배 이상 큰 d 값을 가진 포인트 필터링
    d_ratio = d.unsqueeze(1) / d.unsqueeze(0)
    remove_mask = (d_ratio < thresh_disp) & mask

    # remove_mask를 통해 제거할 포인트를 남기지 않는 새로운 인덱스 계산
    keep_indices = ~(remove_mask.any(dim=1))

    # 최종 남은 포인트들
    filtered_points = points[keep_indices]

    return filtered_points

def refine_disparity_points_optimized(points: torch.Tensor,
                                      thresh_dist=0.5,
                                      thresh_disp=0.85,
                                      cell_size_factor=0.5):
    """
    Memory-efficient disparity refinement.

    points: (N, 3) tensor of (u, v, d)
    thresh_dist: local distance threshold multiplier
    thresh_disp: disparity ratio threshold
    cell_size_factor: cell size = mean(d) * cell_size_factor
    """

    # 좌표 분리
    u, v, d = points[:, 0], points[:, 1], points[:, 2]
    device = points.device

    # --- 1️⃣ Spatial grid 생성 ---
    cell_size = d.mean() * cell_size_factor
    grid_u = torch.floor(u / cell_size).long()
    grid_v = torch.floor(v / cell_size).long()
    grid_keys = grid_u * 73856093 + grid_v * 19349663  # simple spatial hash (avoids collisions)

    # --- 2️⃣ 각 cell별 인덱스 그룹화 ---
    unique_keys, inverse_indices = torch.unique(grid_keys, return_inverse=True)
    cell_points = [[] for _ in range(len(unique_keys))]
    for i, cell_id in enumerate(inverse_indices.tolist()):
        cell_points[cell_id].append(i)

    keep_mask = torch.ones(len(points), dtype=torch.bool, device=device)

    # --- 3️⃣ 이웃 cell 정의 (8방향 + 자기 자신) ---
    neighbor_offsets = torch.tensor(
        [[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1]], device=device
    )

    # --- 4️⃣ 각 cell 내부 및 주변만 비교 ---
    for idx, key in enumerate(unique_keys):
        if not keep_mask.any():
            break

        cell_u, cell_v = grid_u[inverse_indices == idx][0], grid_v[inverse_indices == idx][0]

        # 주변 9개 셀 탐색
        neighbor_cells = (cell_u + neighbor_offsets[:, 0]) * 73856093 + \
                         (cell_v + neighbor_offsets[:, 1]) * 19349663

        # 이웃 cell의 point indices 수집
        neighbor_indices = []
        for neighbor_key in neighbor_cells:
            mask = (grid_keys == neighbor_key)
            if mask.any():
                neighbor_indices.append(torch.nonzero(mask, as_tuple=False).squeeze(1))
        if len(neighbor_indices) == 0:
            continue

        neighbor_indices = torch.cat(neighbor_indices)
        local_indices = torch.tensor(cell_points[idx], device=device)

        u_local, v_local, d_local = u[local_indices], v[local_indices], d[local_indices]
        u_neigh, v_neigh, d_neigh = u[neighbor_indices], v[neighbor_indices], d[neighbor_indices]

        # 거리 계산
        dist_u = u_local.unsqueeze(1) - u_neigh.unsqueeze(0)
        dist_v = v_local.unsqueeze(1) - v_neigh.unsqueeze(0)
        distances = torch.sqrt(dist_u ** 2 + dist_v ** 2)

        mask_dist = distances <= d_local.unsqueeze(1) * thresh_dist
        d_ratio = d_local.unsqueeze(1) / d_neigh.unsqueeze(0)
        remove_mask = (d_ratio < thresh_disp) & mask_dist

        # 제거 대상 표시
        local_remove = remove_mask.any(dim=1)
        keep_mask[local_indices[local_remove]] = False

    # --- 5️⃣ 최종 결과 ---
    filtered_points = points[keep_mask]
    return filtered_points
def project_points_on_camera(
    points: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    image_width: float = 0,
    image_height: float = 0,
):
    """
    Project 3D points to 2D image plane
    Args:
        points (np.ndarray): 3D points to project
        focal_length (float): Focal length of the camera
        cx (float): Principal point x-coordinate
        cy (float): Principal point y-coordinate
        image_width (float): Image width, Optional
        image_height (float): Image height, Optional
    Returns:
        np.ndarray: Projected points
    """
    points[:, 0] = points[:, 0] * focal_length / points[:, 2] + cx
    points[:, 1] = points[:, 1] * focal_length / points[:, 2] + cy

    if image_width > 0 and image_height > 0:
        points = points[
            (points[:, 0] >= 0)
            & (points[:, 0] <= image_width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= image_height - 1)
            & (points[:, 2] > 0)
        ]
    return points


def depth_points_to_depth_map(points: np.ndarray, width=720, height=540):
    points = points[
        (points[:, 0] < width)
        & (points[:, 1] < height)
        & (points[:, 2] > 0)
        & (points[:, 0] >= 0)
        & (points[:, 1] >= 0)
    ]
    depth_map = np.zeros((height, width), dtype=np.float32)
    u, v, d = points.T
    u = u.astype(int)
    v = v.astype(int)
    depth_map[v, u] = d
    return depth_map


def torch_depth_points_to_depth_map(points: torch.Tensor, width=720, height=540):
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    points = points[
        (points[:, 0] < width)
        & (points[:, 1] < height)
        & (points[:, 2] > 0)
        & (points[:, 0] >= 0)
        & (points[:, 1] >= 0)
    ]
    u, v, d = points.T
    u = u.int()
    v = v.int()
    depth_map[v, u] = d
    return depth_map


def render_depth_map(
    points: np.ndarray, width: int = 0, height: int = 0, max_depth=10000
):
    if width == 0:
        width = int(points[:, 0].max()) + 1
    if height == 0:
        height = int(points[:, 1].max()) + 1
    canvas = np.zeros((height, width), dtype=np.uint8)

    for u, v, depth in points:
        depth = depth / max_depth * 255
        depth = np.clip(depth, 0, 255).asdtype(np.uint8)
        canvas[int(v), int(u)] = depth
    return canvas


def points_sampled_disparity(points: np.ndarray, disparity_map: np.ndarray):
    points = points[
        points[:, 1]
        < disparity_map.shape[0] & points[:, 0]
        < disparity_map.shape[1] & points[:, 1]
        >= 0 & points[:, 0]
        >= 0
    ]
    u, v, d = points.T
    d = disparity_map[v.astype(int), u.astype(int)]
    points[:, 2] = d
    return points


def lidar_points_to_disparity(
    points: np.ndarray,
    transform_mtx: np.ndarray,
    focal_length: float,
    baseline: float,
    cx: float,
    cy: float,
):
    points = transform_point_inverse(points, transform_mtx)
    points = project_points_on_camera(points, focal_length, cx, cy, 720, 540)
    points[:, 2] = focal_length * baseline / points[:, 2] - 1
    return points


def pad_lidar_points(lidar_projected_points, target_size=5000):
    current_size = len(lidar_projected_points)

    if current_size >= target_size:
        return lidar_projected_points[:target_size]

    # 필요한 포인트 수 계산
    needed = target_size - current_size

    # 기존 포인트에서 랜덤하게 샘플링 (복원 추출)
    # 샘플링할 포인트 수가 현재 포인트 수보다 많을 경우, 여러 번 반복할 수 있음
    # NumPy의 random.choice를 사용하여 인덱스를 랜덤하게 선택
    sampled_indices = np.random.choice(current_size, size=needed, replace=True)
    sampled_points = lidar_projected_points[sampled_indices]

    # 기존 포인트와 샘플링된 포인트를 결합
    padded_lidar_projected_points = np.concatenate(
        [lidar_projected_points, sampled_points], axis=0
    )

    return padded_lidar_projected_points


def combine_block(
    lidar_points: np.ndarray,
    disparity_rgb: np.ndarray,
    disparity_nir: np.ndarray,
    combined_disparity: np.ndarray,
    criteria: Callable[
        [np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]], bool
    ],
    blk_w=24,
    blk_h=24,
):
    width = disparity_rgb.shape[-1]
    height = disparity_rgb.shape[-2]
    n_blk_u = (width + blk_w - 1) // blk_w  # Ceiling division
    n_blk_v = (height + blk_h - 1) // blk_h  # Ceiling division
    u, v, z = lidar_points.T
    for blk_v_idx in range(n_blk_v):
        for blk_u_idx in range(n_blk_u):
            # Define the vertical block boundaries
            st_v = blk_v_idx * blk_h
            en_v = min((blk_v_idx + 1) * blk_h, height)
            st_u = blk_u_idx * blk_w
            en_u = min((blk_u_idx + 1) * blk_w, width)

            # Identify LiDAR points within the current vertical block
            in_block = (u >= st_u) & (u < en_u) & (v >= st_v) & (v < en_v)

            if not np.any(in_block):
                # No points in this vertical block; retain the horizontal-based disparity
                continue

            # Get the indices of points in the current block
            bu, bv, bz = lidar_points[in_block].T

            # Ensure u and v are within image bounds
            valid = (bu >= 0) & (bu < width) & (bv >= 0) & (bv < height)
            bu, bv, bz = np.stack([bu, bv, bz], axis=1)[valid].T

            critic = criteria(
                bu.astype(np.int32), bv.astype(np.int32), bz, (st_u, en_u, st_v, en_v)
            )

            # Choose the disparity map with lower loss for this block
            if critic:
                chosen_disparity = disparity_rgb[st_v:en_v, st_u:en_u]
            else:
                chosen_disparity = disparity_nir[st_v:en_v, st_u:en_u]

            # Assign the chosen disparity to the combined map
            combined_disparity[st_v:en_v, st_u:en_u] = chosen_disparity

    return combined_disparity


def refine_disparity(
    disparity_map: torch.Tensor,
    image_left: torch.Tensor,
    image_right: torch.Tensor,
    disp_thresh=4.0,
):
    disparity_map = disparity_map.unsqueeze(0)
    image_left = image_left.unsqueeze(0)
    image_right = image_right.unsqueeze(0)
    reprojected_right = reproject_disparity(disparity_map, image_left)
    ssim_loss = ssim_torch(reprojected_right, image_right).mean(dim=1)

    # Create a mask for conditions where disparity_map <= 4.0 and ssim_loss >= 0.98
    ssim_loss = F.pad(ssim_loss, [0, 2, 0, 2])
    mask = (disparity_map <= disp_thresh) & (
        ssim_loss.unsqueeze(1) >= 0.98
    )  # Adjust shape of ssim_loss for broadcasting

    # Set the corresponding pixels in disparity_map to 0
    disparity_map[mask] = 0

    return disparity_map[0]


def refine_disparity_with_monodepth(disparity_map: np.ndarray, mono_depth: np.ndarray):
    mask = (mono_depth <= 1).astype(np.float32)
    mask = gaussian_filter(mask, 9)
    disparity_map = disparity_map * (1 - mask) + mono_depth * mask
    return disparity_map


def ssim_torch(x: torch.Tensor, y: torch.Tensor):
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return SSIM


def reproject_disparity(
    disparity_map: torch.Tensor, left_image: torch.Tensor, max_disparity=128
):
    batch_size, channels, height, width = left_image.shape
    # Create a mesh grid for pixel coordinates
    x_coords, y_coords = torch.meshgrid(
        torch.arange(width, device=left_image.device),
        torch.arange(height, device=left_image.device),
        indexing="xy",
    )

    x_coords = x_coords.unsqueeze(0).expand(batch_size, -1, -1).float()
    y_coords = y_coords.unsqueeze(0).expand(batch_size, -1, -1).float()

    # Compute the new x coordinates based on disparity
    disparity_map = F.pad(
        disparity_map, (1, 1, 1, 1), mode="constant", value=0
    )  # Pad to handle boundary
    disparity_map = F.interpolate(
        disparity_map, size=(height, width), mode="bilinear", align_corners=False
    )  # Resample disparity map

    # Convert disparity map to float type
    disparity_map = disparity_map.squeeze(1)

    x_new_coords = x_coords - disparity_map
    y_new_coords = y_coords

    # Create grid tensor with shape [N, H, W, 2]
    grid = torch.stack([x_new_coords, y_new_coords], dim=-1)

    # Normalize the grid to the range [-1, 1]
    grid = (
        2.0 * grid / torch.tensor([width - 1, height - 1], device=left_image.device)
        - 1.0
    )

    # Perform bilinear interpolation for the reprojected image
    reprojected_image = F.grid_sample(
        left_image, grid, mode="bilinear", align_corners=False
    )

    return reprojected_image


def chamfer_loss_pc(pc1, pc2):
    """
    Chamfer Loss를 순수 PyTorch로 구현
    Args:
        pc1: [B, N, 3] 텐서
        pc2: [B, M, 3] 텐서
        batch_size: 메모리 최적화를 위한 배치 크기 (기본값: 1)
        device: 'cuda' 또는 'cpu'
    Returns:
        loss: 스칼라 텐서
    """
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape
    loss = 0.0

    # 각 배치에 대해 Chamfer Loss 계산
    for b in range(B):
        # 포인트 클라우드 A와 B
        A = pc1[b]  # [N, 3]
        B_pc = pc2[b]  # [M, 3]

        # 메모리 최적화를 위해 N과 M을 작은 배치로 나누기
        # 예: N_chunk와 M_chunk를 설정하여 한 번에 처리할 포인트 수를 제한
        N_chunk_size = 1024
        M_chunk_size = 1024

        # A to B
        min_dist_A_to_B = []
        for i in range(0, N, N_chunk_size):
            A_chunk = A[i : i + N_chunk_size].unsqueeze(1)  # [chunk, 1, 3]
            B_expand = B_pc.unsqueeze(0)  # [1, M, 3]
            dists = torch.norm(A_chunk - B_expand, dim=2)  # [chunk, M]
            min_dists, _ = torch.min(dists, dim=1)  # [chunk]
            min_dist_A_to_B.append(min_dists)
        min_dist_A_to_B = torch.cat(min_dist_A_to_B, dim=0)  # [N]
        loss_A_to_B = min_dist_A_to_B.mean()

        # B to A
        min_dist_B_to_A = []
        for i in range(0, M, M_chunk_size):
            B_chunk = B_pc[i : i + M_chunk_size].unsqueeze(1)  # [chunk, 1, 3]
            A_expand = A.unsqueeze(0)  # [1, N, 3]
            dists = torch.norm(B_chunk - A_expand, dim=2)  # [chunk, N]
            min_dists, _ = torch.min(dists, dim=1)  # [chunk]
            min_dist_B_to_A.append(min_dists)
        min_dist_B_to_A = torch.cat(min_dist_B_to_A, dim=0)  # [M]
        loss_B_to_A = min_dist_B_to_A.mean()

        loss += loss_A_to_B + loss_B_to_A

    # 평균을 내기 위해 배치 크기로 나누기
    loss = loss / B
    return loss


def disparity_map_to_point_cloud(
    disparity_map: torch.Tensor, fx=860, cx=360, cy=270, bs=135
):
    depth = torch.where(
        disparity_map > 0, fx * bs / disparity_map, torch.zeros_like(disparity_map)
    )

    B, _, H, W = disparity_map.shape
    device = depth.device

    # 생성할 좌표 그리드
    u = torch.arange(0, W, device=device).view(1, 1, W).expand(B, 1, H, W)
    v = torch.arange(0, H, device=device).view(1, H, 1).expand(B, 1, H, W)
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fx
    z = depth
    point_cloud = torch.stack([x, y, z], dim=1).reshape(B, 3, -1).permute(0, 2, 1)
    return point_cloud


def downsample_point_cloud(point_cloud, num_samples):
    """
    포인트 클라우드를 무작위로 다운샘플링
    Args:
        point_cloud: [B, N, 3] 텐서
        num_samples: 샘플링할 포인트 수
    Returns:
        downsampled_pc: [B, num_samples, 3] 텐서
    """
    B, N, _ = point_cloud.shape
    if N <= num_samples:
        return point_cloud
    indices = torch.randint(0, N, (B, num_samples), device=point_cloud.device)
    downsampled_pc = torch.gather(point_cloud, 1, indices.unsqueeze(-1).repeat(1, 1, 3))
    return downsampled_pc


def disparity_points_loss(
    disparity_map: torch.Tensor,
    point_disparity: torch.Tensor,
    fx=860,
    cx=360,
    cy=270,
    bs=135,
):
    point_depth = torch.where(
        point_disparity > 0,
        fx * bs / point_disparity,
        torch.zeros_like(point_disparity),
    )
    point_depth[..., 0] = (point_depth[..., 0] - cx) * point_depth[..., 2] / fx
    point_depth[..., 1] = (point_depth[..., 1] - cy) * point_depth[..., 2] / fx
    pred_point_cloud = disparity_map_to_point_cloud(disparity_map, fx, cx, cy, bs)
    pred_point_cloud = downsample_point_cloud(pred_point_cloud, 10000)
    loss = chamfer_loss_pc(pred_point_cloud, point_depth)
    return loss


import numpy as np
from scipy.spatial import cKDTree


def lidar_occlusion_filter(lidar, cal_tuple, transform, uv_max=16):
    # 변환 과정: 실제 좌표계를 계산
    fx, cx, cy = cal_tuple
    u, v, z = lidar.T
    x = (u - cx) * z / fx
    y = (v - cy) * z / fx
    lidar_virtual = np.column_stack((x, y, z))

    transform_virtual = transform.copy()
    transform_virtual[:2, 3] = -transform_virtual[:2, 3] * 0
    lidar_virtual = transform_points(lidar_virtual, transform)
    lidar_virtual = transform_point_inverse(lidar_virtual, transform_virtual)

    x, y, z = lidar_virtual.T
    u = (x * fx / z) + cx
    v = (y * fx / z) + cy
    lidar_virtual = np.column_stack((u, v, z))

    del_indices = set()
    # uv 좌표(첫 두 열)를 기준으로 KD-Tree 구축
    tree = cKDTree(lidar[:, :2])
    transform_v = np.linalg.inv(transform)
    for i in range(len(lidar)):
        if i in del_indices:
            continue
        # 현재 포인트의 uv 좌표에서 uv_max 이내에 있는 이웃 포인트들만 검색
        neighbor_indices = tree.query_ball_point(lidar[i, :2], uv_max)
        neighbor_indices.remove(i)
        if len(neighbor_indices) == 0:
            continue

        neighbor_indices = np.array(neighbor_indices)
        neighbor_indices = neighbor_indices[
            lidar[neighbor_indices, 2] > lidar[i, 2] * 1.1
        ]

        diff = lidar_virtual[neighbor_indices, :2] - lidar_virtual[i, :2]
        norm = np.linalg.norm(diff, axis=1)
        direction = diff / norm[:, None]
        # 변환된 포인트 간의 차이가 가까운 방향과 일치하면 occlusion 처리

        occlusion = (
            np.sum(
                np.dot(lidar[neighbor_indices, :2] - lidar[i, :2], direction.T), axis=1
            )
            <= 0
        )
        del_indices.update(neighbor_indices[occlusion])

    lidar_filtered = np.delete(lidar, list(del_indices), axis=0)
    return lidar_filtered

