import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import yaml
import traceback
from utils.image_process.points import (
    project_points_on_camera,
)
import re
from tqdm import tqdm
from utils.image_process.points_v2 import occlusion_aware_splat


import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Any
import json
import math
import os
try:
    from method.depth_warp_hdr.real_calibration_fallback import (
        resolve_lidar_transform_path,
        resolve_multi_cam_calibration_path,
    )
except ModuleNotFoundError:
    from real_calibration_fallback import (
        resolve_lidar_transform_path,
        resolve_multi_cam_calibration_path,
    )

cv2.setNumThreads(0)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")




@torch.no_grad()
def transform_points_torch(points: torch.Tensor,
                           transform_mtx: torch.Tensor) -> torch.Tensor:
    """
    3D points를 4x4 SE(3) homogeneous transform으로 변환.
    - points: (N, 3) 혹은 (3,) float Tensor (권장 device: cuda)
    - transform_mtx: (4, 4) float Tensor (권장 device: cuda)
    반환: (N, 3) float Tensor
    참고: 원본 NumPy 구현과 동일하게 w-분할(perspective divide)은 수행하지 않음.
    """
    # (N,3)로 정규화
    points = points.reshape(-1, 3)

    # homogeneous 좌표 생성: (N,4)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homo = torch.cat([points, ones], dim=1)
    out = transform_mtx @ homo.T

    return out.T[...,:3]


@torch.no_grad()
def depth_points_to_depth_map_torch(points: torch.Tensor,
                                    width: int = 720,
                                    height: int = 540) -> torch.Tensor:
    """
    (u, v, d) 포인트를 depth map으로 rasterization.
    - points: (N, 3) float Tensor, columns = (u, v, depth), 권장 device: cuda
    - width, height: 출력 맵 크기
    반환: (H, W) float32 Tensor (입력과 동일 device)
    구현 규약:
      * 화면 범위 밖/음수/비양수 depth는 제거
      * u,v를 int로 캐스팅해 픽셀 인덱스로 사용 (np.astype(int)와 동일하게 0쪽으로 truncation)
      * 동일 픽셀로 다수 포인트가 들어오면 "마지막 값이 기록"됨 (원본 NumPy와 동일)
    """
    # 유효 마스크
    m = (
        (points[:, 0] < float(width)) &
        (points[:, 1] < float(height)) &
        (points[:, 2] > 0) &
        (points[:, 0] >= 0) &
        (points[:, 1] >= 0)
    )
    pts = points[m]

    depth_map = torch.zeros((height, width), dtype=torch.float32, device=points.device)
    if pts.numel() == 0:
        return depth_map

    # 정수 인덱스 (np.astype(int)와 동일한 truncation)
    u = pts[:, 0].to(torch.int64)
    v = pts[:, 1].to(torch.int64)
    d = pts[:, 2].to(depth_map.dtype)

    # 원소별 쓰기 (CUDA 지원)
    depth_map[v, u] = d
    return depth_map


def gain_to_linear(
    gain_value: float, mode: str = "dB", db_rule: str = "20log10"
) -> float:
    """
    mode: 'dB' or 'linear'
    db_rule: '20log10' (6 dB ≈ ×2) 또는 '10log10' (3 dB ≈ ×2) — 카메라 문서로 확인
    """
    if mode.lower() == "linear":
        return float(gain_value)
    elif mode.lower() == "db":
        if db_rule == "20log10":
            return float(10.0 ** (gain_value / 20.0))
        elif db_rule == "10log10":
            return float(10.0 ** (gain_value / 10.0))
        else:
            raise ValueError(f"Unknown db_rule: {db_rule}")
    else:
        raise ValueError(f"Unknown gain mode: {mode}")


@dataclass
class DarkNoiseParams:
    camera_id: str
    channel: str
    mu0: float  # [0..1] 스케일 기준으로 피팅 권장
    kq: float  # 분위수 상수까지 흡수된 스케일 팩터
    a0: float  # read-noise 상수항 (>=0)
    b0: float  # dark-current 계수 (>=0)  -> sqrt(e) 항에 곱해짐
    gain_mode: str = "dB"  # 'dB' or 'linear'
    db_rule: str = "20log10"  # dB를 linear로 바꾸는 규칙
    use_residual: bool = False
    residual_x: Optional[list] = None  # exposure grid
    residual_y: Optional[list] = None  # residual grid (same length as residual_x)
    # meta
    n_points_used: Optional[int] = None
    n_points_total: Optional[int] = None
    full_scale: Optional[float] = None  # (선택) DN→[0,1] 변환용. 예: 4095.0

    def _residual_1d(self, exposure: float) -> float:
        if not (self.use_residual and self.residual_x and self.residual_y):
            return 0.0
        x, y = self.residual_x, self.residual_y
        if exposure <= x[0]:
            return float(y[0])
        if exposure >= x[-1]:
            return float(y[-1])
        lo, hi = 0, len(x) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if x[mid] <= exposure:
                lo = mid
            else:
                hi = mid
        t = (exposure - x[lo]) / (x[hi] - x[lo])
        return float((1 - t) * y[lo] + t * y[hi])

    def min_intensity(self, exposure: float, gain: float) -> float:
        if exposure < 0:
            raise ValueError("exposure must be non-negative")
        # 만약 파라미터가 DN 기반으로 피팅되었다면, 우선 [0,1]로 변환 (권장: 애초에 [0,1]로 재피팅)
        scale = 1.0
        if self.full_scale and self.full_scale > 1.0:
            scale = 1.0 / float(self.full_scale)

        g_lin = gain_to_linear(gain, mode=self.gain_mode, db_rule=self.db_rule)
        sigma_like = math.sqrt(
            max(self.a0, 0.0) ** 2 + max(self.b0, 0.0) ** 2 * max(exposure, 0.0)
        )
        base = self.mu0 + self.kq * g_lin * sigma_like
        base *= scale  # DN→[0,1] 보정 (필요 시)

        return max(0.0, min(1.0, base + self._residual_1d(exposure)))


def _load_dark_noise_table(json_path: str) -> Dict[str, DarkNoiseParams]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"dark noise table not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    table: Dict[str, DarkNoiseParams] = {}
    for row in data:
        p = DarkNoiseParams(
            camera_id=str(row["camera_id"]),
            channel=row.get("channel", "Unknown"),
            mu0=float(row["mu0"]),
            kq=float(row["k"]),
            a0=float(row["a"]),
            b0=float(row["b"]),
            gain_mode=row.get("gain_mode", "dB"),
            use_residual=bool(row.get("use_residual", False)),
            residual_x=row.get("residual_spline_knots"),
            residual_y=row.get("residual_spline_s"),
            n_points_used=row.get("n_points_used"),
            n_points_total=row.get("n_points_total"),
        )
        table[p.camera_id] = p
    return table


def _make_min_fn(params: DarkNoiseParams) -> Callable[[float, float], float]:
    """
    반환 함수 시그니처: f(exposure: float, gain: float) -> float
    - exposure 단위와 gain 단위는 params 설정(gain_mode)에 맞춰 넣어야 합니다.
    """

    def f(exposure: float, gain: float) -> float:
        return params.min_intensity(exposure, gain)

    f.__name__ = f"min_intensity_{params.camera_id}"
    f.__doc__ = (
        f"Min intensity for camera_id={params.camera_id} (channel={params.channel}, "
        f"gain_mode={params.gain_mode}). "
        "Args: exposure (>=0, same unit as model), gain (dB or linear)."
    )
    return f


def _opencv_matrix_constructor(loader, node):
    try:
        mapping = loader.construct_mapping(node, deep=True)
        return mapping
    except Exception:
        return {}


for _tag in ("!opencv-matrix", "!!opencv-matrix", "tag:yaml.org,2002:opencv-matrix"):
    try:
        yaml.SafeLoader.add_constructor(_tag, _opencv_matrix_constructor)
    except Exception:
        # ignore if loader doesn't support adding constructors
        pass


def load_calibration_yaml(calib_path: str) -> dict:
    """Load OpenCV-style calibration YAML and return as dict.
    Tolerate OpenCV-style YAML that starts with '%YAML:...' by stripping that line.
    """
    if not calib_path or not os.path.exists(calib_path):
        return {}
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Remove YAML directive lines like "%YAML:1.0" or "%YAML 1.0"
        text = re.sub(r"^%YAML[:\s].*\n", "", text)
        # Remove leading document marker '---' if present
        text = re.sub(r"^\s*---\s*\n", "", text)
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        print(f"[calib] load fail {calib_path}: {e}")
        return {}


def _np_load(path):
    return np.load(path, allow_pickle=False, mmap_mode='r')

import time
class LucidHexItem:
    def __init__(
        self,
        scene_idx,
        frame_path,
        camera_intrinsics,
        camera_extrinsics,
        cam_lidar_transform,
        fn_hdr_dn=None,
        fn_ldr_dn=None,
        output_aspect_ratio=(3, 4),
        rescale_input=None,
        rescale_output=None,
        compiled_splat_fn=None,
        refine_depth = True,
        input_random_crop: bool = False,
        input_random_crop_max: float = 0.1,
        intel=False,
    ):
        self.scene_idx = scene_idx
        # New dataset layout: files are directly under the frame folder (no `post` subfolder)
        self.frame_raw = frame_path
        self.frame_path = frame_path
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics = camera_extrinsics
        self.cam_lidar_transform = cam_lidar_transform
        # optional rescale sizes: tuples (H, W) or None
        self.rescale_input = rescale_input
        self.rescale_output = rescale_output
        self.ldr_sides = ["rear", "left", "right", "rear_sub", "left_sub", "right_sub"]
        self.hdr_sides = ["left", "right"]
        self.fn_hdr_dn = fn_hdr_dn
        self.fn_ldr_dn = fn_ldr_dn
        self.output_aspect_ratio = output_aspect_ratio  # (H,W)
        self.splat_fn = occlusion_aware_splat if compiled_splat_fn is None else compiled_splat_fn
        self.refine_depth = refine_depth
        # random crop augmentation for input (applies to LDR / '12_' sides)
        self.input_random_crop = bool(input_random_crop)
        # maximum fraction of width/height that may be cropped from each side
        # (sampled independently for top/bottom/left/right in [0, max_fraction])
        self.input_random_crop_max = float(input_random_crop_max)
        self.intel = intel
      

    def unpack(self):
        """
        Simplified unpack for the new dataset layout.
        - Loads {base}_raw.exr from the frame folder for every side (base: strip '12_' prefix).
        - Loads lidar points from {frame}/ouster/points.npy (if present).
        - Returns for each side: rgb_{side}, lidar_{side}, K_{side}, E_{side} (torch tensors).
        Exposure / dark-noise fields are omitted because the raw EXR is GT HDR.
        """
        return_dict = {}

        # Load lidar points (single point cloud used for all sides)
        points_path = os.path.join(self.frame_raw, "ouster", "points.npy")
        if os.path.exists(points_path):
            try:
                lidar_pts = _np_load(points_path).reshape(-1, 3).astype(np.float32)
            except Exception:
                lidar_pts = np.zeros((0, 3), dtype=np.float32)
        else:
            lidar_pts = np.zeros((0, 3), dtype=np.float32)

        # Helper to map side key to base filename (strip '12_' prefix)

        # For each side, load the raw EXR and attach lidar, K, E
        sides_union = self.hdr_sides + [f"12_{x}" for x in self.ldr_sides]
        
        # hdr images
        hdr_as_r = self.output_aspect_ratio[0] / self.output_aspect_ratio[1]
        if hdr_as_r > 928 / 1440:
            target_w = (int(928 / hdr_as_r) // 2) * 2
            target_h = 928
        else:
            target_h = (int(1440 * hdr_as_r) // 2) * 2
            target_w = 1440
        t_rct = (
            1440 // 2 - target_w // 2,
            928 // 2 - target_h // 2,
            target_w,
            target_h,
        )
        
        for side in sides_union:
            
            raw_path = os.path.join(self.frame_path, f"{side}_raw.exr")
            img = None
            if os.path.exists(raw_path):
                try:
                    img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
                    # some EXR readers return BGR; keep as-is but convert to float32
                    if img is not None:
                        img = img.astype(np.float32)
                except Exception as e:
                    print(f"Error loading raw EXR {raw_path}: {e}")
                    img = None

            if img is None:
                # default empty image (small reasonable size); keep channels-first
                img = np.zeros((3, 256, 342), dtype=np.float32)
                img = np.transpose(img, (1, 2, 0))  # to HWC for consistency below
            # If image loaded as HxWxC, ensure shape
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            K = np.asarray(self.camera_intrinsics[side]).copy()
            E = np.asarray(self.camera_extrinsics[side]).copy()
            # Optionally rescale if requested (preserve original centering)
            
            if "12_" in side and self.input_random_crop:
                # perform a random crop on LDR ('12_' sides) as augmentation
                try:
                    h_img, w_img = img.shape[0], img.shape[1]
                    max_h_crop = int(round(self.input_random_crop_max * float(h_img)))
                    max_w_crop = int(round(self.input_random_crop_max * float(w_img)))
                    if max_h_crop > 0 or max_w_crop > 0:
                        left = int(np.random.randint(0, max_w_crop + 1)) if max_w_crop > 0 else 0
                        right = int(np.random.randint(0, max_w_crop + 1)) if max_w_crop > 0 else 0
                        top = int(np.random.randint(0, max_h_crop + 1)) if max_h_crop > 0 else 0
                        bottom = int(np.random.randint(0, max_h_crop + 1)) if max_h_crop > 0 else 0
                        # ensure we don't crop away the entire image
                        if left + right >= w_img:
                            left = min(left, max(0, w_img - 1))
                            right = 0
                        if top + bottom >= h_img:
                            top = min(top, max(0, h_img - 1))
                            bottom = 0
                        x0 = left
                        y0 = top
                        x1 = w_img - right
                        y1 = h_img - bottom
                        img = img[y0:y1, x0:x1, :]
                        # principal point shifts by the amount cropped from left/top
                        K[0, 2] -= float(x0)
                        K[1, 2] -= float(y0)
                except Exception:
                    # if anything goes wrong, fall back to no crop
                    pass

            if "12_" in side and self.rescale_input is not None:
                H_si, W_si = self.rescale_input
                
                K[0, 0] *= W_si / img.shape[1]
                K[1, 1] *= H_si / img.shape[0]
                K[0, 2] *= W_si / img.shape[1]
                K[1, 2] *= H_si / img.shape[0]
                img = cv2.resize(img, (W_si, H_si), interpolation=cv2.INTER_LINEAR)
            if not "12_" in side:
                # crop to target aspect ratio
                x, y, w, h = t_rct
                img = img[y : y + h, x : x + w, :]
                K[0, 2] -= x
                K[1, 2] -= y
                # rescale to output size if requested
                if self.rescale_output is not None:
                    H_so, W_so = self.rescale_output
                    img = cv2.resize(img, (W_so, H_so), interpolation=cv2.INTER_LINEAR)
                    K[0, 0] *= W_so / w
                    K[1, 1] *= H_so / h
                    K[0, 2] *= W_so / w
                    K[1, 2] *= H_so / h
           
                
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).contiguous().float()
            return_dict[f"rgb_{side}"] = img_t

            
            #return_dict[f"lidar_{side}"] = torch.from_numpy(lidar_pts.copy()).float()
            
            if self.intel:
                lidar = np.load(os.path.join(self.frame_raw, "realsense", "depth.npy"))
                lidar = cv2.resize(lidar, (img_t.shape[2], img_t.shape[1]), interpolation=cv2.INTER_NEAREST)
                lidar_depthmap = torch.from_numpy(lidar.copy()).float()
                return_dict[f"lidar_{side}"] = lidar_depthmap
            else:
                lidar_to_cam = torch.from_numpy(lidar_pts.copy()).contiguous().float().reshape(-1,3)
                lidar_to_cam = transform_points_torch(lidar_to_cam * 1000.0, self.cam_lidar_transform[side])
                fx = K[0,0]
                cx = K[0,2]
                cy = K[1,2]
                width = img_t.shape[2]
                height = img_t.shape[1]
                lidar_to_cam = project_points_on_camera(
                    lidar_to_cam, fx, cx, cy, width, height
                )
                
                
                lidar_depthmap = depth_points_to_depth_map_torch(lidar_to_cam, 
                                                                width=width, 
                                                                height=height)

                return_dict[f"lidar_{side}"] = lidar_depthmap
            # intrinsics/extrinsics precomputed in dataset constructor
            
            return_dict[f"K_{side}"] = torch.from_numpy(K).float()
            return_dict[f"E_{side}"] = torch.from_numpy(E).float()

        return return_dict


class LucidHexUltraDataset(Dataset):

    sides = [
        "left",
        "right",
        "12_rear",
        "12_left",
        "12_right",
        "12_rear_sub",
        "12_left_sub",
        "12_right_sub",
    ]

    def __init__(
        self,
        scenes=[],
        rescale_input=(384, 512),
        rescale_output=(480, 736),
        cache_new=False,
        compiled_splat_fn=None,
        output_aspect_ratio=(3, 4),
        input_random_crop: bool = False,
        input_random_crop_max: float = 0.1,
        random_batch_mode = False,
        intel=False,
    ):
        super().__init__()
        self.scenes = scenes
        self.frames = []
        self.output_aspect_ratio = output_aspect_ratio
        
        self.calibration_dict = {}
        
        self.random_batch_mode = random_batch_mode
        self.cache_new = cache_new
    # input random crop augmentation for LDR inputs (12_ sides)
        self.input_random_crop = bool(input_random_crop)
        self.input_random_crop_max = float(input_random_crop_max)
        self.intel = intel
        if isinstance(self.scenes, str):
            self.scenes = [self.scenes]
        for i, scene in enumerate(self.scenes):
            if not os.path.isabs(scene):
                scene_new = os.path.join(os.environ.get("LUCID_ROOT", "/bean/lucid"), scene)
                self.scenes[i] = scene_new
        if len(self.scenes) == 0:
            self.search_scenes()
        # self.camera_calibration = load_calibration_yaml(camera_calibration)
        # self.lidar_transform = _np_load(lidar_transform)

        # self.camera_intrinsics, self.camera_extrinsics, self.cam_lidar_transform = (
        #     self.precompute_lidar_transform(
        #         self.lidar_transform, self.camera_calibration
        #     )
        # )
        self.fn_hdr_dn, self.fn_ldr_dn = self.get_dark_noise_function()
        self.rescale_input = rescale_input
        self.rescale_output = rescale_output
        self.compiled_splat_fn = compiled_splat_fn
        for i, scene in enumerate(tqdm(self.scenes, desc="Preparing frames")):
            frames = self.retrieve_frames(scene)
            camera_calibration_dict = load_calibration_yaml(
                resolve_multi_cam_calibration_path(scene)
            )
            lidar_transform = _np_load(resolve_lidar_transform_path(scene))
            camera_intrinsics, camera_extrinsics, cam_lidar_transform = (
                self.precompute_lidar_transform(
                    lidar_transform, camera_calibration_dict
                )
            )
            self.calibration_dict[i] = {
                "camera_calibration": camera_calibration_dict,
                "lidar_transform": lidar_transform,
                "camera_intrinsics": camera_intrinsics,
                "camera_extrinsics": camera_extrinsics,
                "cam_lidar_transform": cam_lidar_transform,
            }
            
            for frame in frames:
                self.frames.append((i, frame))
        

    def precompute_lidar_transform(self, lidar_transform, camera_calibration):
        cam_intrinsics = {}
        cam_extrinsics = {}
        cam_lidar_transform = {}
        for side in self.sides:
            cam_K = np.asarray(camera_calibration[f"k_{side}"]["data"]).reshape(3, 3)
            
            #cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
            cam_D = np.asarray(
                camera_calibration[f"d_{side}"]["data"]
            ).reshape(-1, 1)
            if "12_" in side:
                w, h = 1024, 768
            else:
                w, h = 1440, 928
            new_K, _ = cv2.getOptimalNewCameraMatrix(cam_K, cam_D, (w, h), 0)
            cam_K = new_K
            
            cam_ext = np.asarray(
                camera_calibration[f"t_{side}_transform_in_12_rear"]["data"]
            ).reshape(4, 4)
            cam_intrinsics[side] = cam_K
            cam_extrinsics[side] = cam_ext
            # lidar transform is lidar_to_rear. compute per-side lidar to cam

            lidar_to_cam = np.linalg.pinv(cam_ext) @ lidar_transform
            cam_lidar_transform[side] = torch.from_numpy(lidar_to_cam.copy()).float()
        return cam_intrinsics, cam_extrinsics, cam_lidar_transform

    def search_scenes(self, root=None):
        if root is None:
            root = os.environ.get("LUCID_ROOT", "/bean/lucid")
        # search all scenes in root
        # scene folder follows pattern MM_DD_HH_MM
        # scene folder should contains frame subfolders, follows pattern HH_MM_SS_mmm

        scenes = []
        for date_folder in os.listdir(root):
            date_path = os.path.join(root, date_folder)
            if not os.path.isdir(date_path):
                continue
            for scene_folder in os.listdir(date_path):
                scene_path = os.path.join(date_path, scene_folder)
                if not os.path.isdir(scene_path):
                    continue
                scenes.append(scene_path)
        self.scenes = scenes

    def retrieve_frames(self, scene_path):
        # frame should contain "post" folder in
        if not self.cache_new and os.path.exists(
            os.path.join(scene_path, "frames_cache.txt")
        ):
            # load from cache
            frames = []
            with open(
                os.path.join(scene_path, "frames_cache.txt"), "r", encoding="utf-8"
            ) as f:
                for line in f:
                    frame_path = line.strip()
                    # if os.path.exists(frame_path):
                    frames.append(frame_path)
            frames = sorted(frames)
            if len(frames) > 0:
                return frames

        frames = []
        # New layout: no `post` folder. Require metadata.json and per-side raw EXR files.
 

        for frame_folder in tqdm(
            os.listdir(scene_path),
            desc=f"Searching frames in {os.path.basename(scene_path)}",
        ):
            frame_path = os.path.join(scene_path, frame_folder)
            if not os.path.isdir(frame_path):
                print(f"[frame] {frame_path} is not a directory, skipping")
                continue
            # metadata.json is still required
            if not os.path.exists(os.path.join(frame_path, "metadata.json")):
                print(f"[frame] missing metadata.json in {frame_path}, skipping")
                continue

            # require raw EXR for every side (strip '12_' prefix when checking filenames)
            ok = True
            for side in self.sides:
                base = side
                raw_path = os.path.join(frame_path, f"{base}_raw.exr")
                if not os.path.exists(raw_path):
                    ok = False
                    print(f"[frame] missing raw EXR in {frame_path} : {raw_path}, skipping")
                    break
            if not ok:
                
                continue

            frames.append(frame_path)
        frames = sorted(frames)

        cache_file = os.path.join(scene_path, "frames_cache.txt")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                for frame_path in frames:
                    f.write(frame_path + "\n")
        except Exception as e:
            print(f"[cache] failed to write cache file {cache_file}: {e}")
            traceback.print_exc()

        return frames

    def update_cache_remove_frame(self, scene_idx, frame_path):
        scene_path = self.scenes[scene_idx]
        cache_file = os.path.join(scene_path, "frames_cache.txt")
        if not os.path.exists(cache_file):
            return
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(cache_file, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip() != frame_path:
                        f.write(line)
        except Exception as e:
            print(f"[cache] failed to update cache file {cache_file}: {e}")
            traceback.print_exc()

    def __len__(self):
        if self.random_batch_mode:
            return 30000  # large number for random sampling
        return len(self.frames)

    def __getitem__(self, idx):
        if self.random_batch_mode:
            idx = np.random.randint(0, len(self.frames))
        scene_idx, frame_path = self.frames[idx]
        try:
            item = LucidHexItem(
                scene_idx,
                frame_path,
                
                self.calibration_dict[scene_idx]["camera_intrinsics"],
                self.calibration_dict[scene_idx]["camera_extrinsics"],
                self.calibration_dict[scene_idx]["cam_lidar_transform"],
                self.fn_hdr_dn,
                self.fn_ldr_dn,
                output_aspect_ratio=self.output_aspect_ratio,
                rescale_input=self.rescale_input,
                rescale_output=self.rescale_output,
                input_random_crop=self.input_random_crop,
                input_random_crop_max=self.input_random_crop_max,
                compiled_splat_fn=self.compiled_splat_fn,
                intel=self.intel
            ).unpack()
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            traceback.print_exc()
            self.update_cache_remove_frame(scene_idx, frame_path)
            return None
        return item

    def get_dark_noise_function(
        self,
        dark_noise_static: str | None = None,
        cams: Dict[str, str] = {
            "hdr": "224201564",
            "ldr": "253200221",
        },
    ) -> Tuple[Callable[[float, float], float], Callable[[float, float], float]]:
        """
        dark_noise_static: dark_min.json 경로
        cams: {"hdr": <camera_id>, "ldr": <camera_id>}
        return: (function_hdr, function_ldr)
        """
        if dark_noise_static is None:
            dark_noise_static = os.environ.get("LUCID_DARK_MIN", "/bean/lucid/dark_min.json")
        table = _load_dark_noise_table(dark_noise_static)

        def pick(cam_key: str) -> Callable[[float, float], float]:
            cam_id = str(cams[cam_key])
            if cam_id not in table:
                raise KeyError(
                    f"camera_id '{cam_id}' not found in {dark_noise_static} (key='{cam_key}')"
                )
            return _make_min_fn(table[cam_id])

        fn_hdr = pick("hdr")
        fn_ldr = pick("ldr")
        return fn_hdr, fn_ldr
