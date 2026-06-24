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
    ):
        self.scene_idx = scene_idx
        self.frame_raw = frame_path
        self.frame_path = frame_path + "/post"
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
        

    def unpack(self):
        # ldr images
        # ldr images are in post/rgb_lucid_12_{side}.exr, load all 6 sides, if one side is missing, fill with zeros
        return_dict = {}
      
        for side in self.ldr_sides:
            ldr = None
            if os.path.exists(
                os.path.join(self.frame_path, f"rgb_lucid_12_{side}_4096.tiff")
            ):
                try:
                    ldr = cv2.imread(
                        os.path.join(
                            self.frame_path, f"rgb_lucid_12_{side}_4096.tiff"
                        ),
                        cv2.IMREAD_UNCHANGED,
                    )[...,::-1]
                    
                    ldr = ldr.astype(np.float32) / 4095.0
                except Exception as e:
                    print(f"Error loading LDR image {side}: {e}")
                    ldr = None
            if ldr is None and os.path.exists(
                os.path.join(self.frame_path, f"rgb_lucid_12_{side}.exr")
            ):
                try:
                    ldr = cv2.imread(
                        os.path.join(self.frame_path, f"rgb_lucid_12_{side}.exr"),
                        cv2.IMREAD_UNCHANGED,
                    )
                except Exception as e:
                    print(f"Error loading LDR image {side}: {e}")
                    ldr = None
            if ldr is None:
                ldr = np.zeros((768, 1024, 3), dtype=np.float32)
            
                # apply input rescale if requested
            if self.rescale_input is not None:
           
                H_si, W_si = self.rescale_input
                ldr = cv2.resize(ldr, (W_si, H_si), interpolation=cv2.INTER_LINEAR)
              
            ldr = torch.from_numpy(ldr.copy()).permute(2, 0, 1).contiguous()  # [C,H,W]

            return_dict[f"rgb_12_{side}"] = ldr
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
       
        for side in self.hdr_sides:
            
            try:
                hdr_path = os.path.join(self.frame_path, f"rgb_lucid_{side}.exr")
                if os.path.exists(hdr_path):
                    hdr = cv2.imread(
                        hdr_path,
                        cv2.IMREAD_UNCHANGED,
                    )
                else:
                    hdr = None
            except Exception as e:
                print(f"Error loading HDR image {side}: {e}")
                hdr = None
            if hdr is None:
                hdr = np.zeros((928, 1440, 3), dtype=np.float32)
            #hdr = torch.from_numpy(hdr).permute(2, 0, 1).float()  # [C,H,W]
            hdr = hdr[
                t_rct[1] : t_rct[1] + t_rct[3], t_rct[0] : t_rct[0] + t_rct[2]
            ]

            # apply output rescale if requested
            if self.rescale_output is not None:
                H_so, W_so = self.rescale_output
                hdr = cv2.resize(hdr, (W_so, H_so), interpolation=cv2.INTER_LINEAR)
               
            hdr = torch.from_numpy(hdr.copy()).permute(2, 0, 1).contiguous()  # [C,H,W]
            return_dict[f"rgb_{side}"] = hdr

        # lidar points
        lidar_gt = None
        lidar_sf = None
        #points_path = os.path.join(self.frame_raw, "ouster", "points.npy")
        points_path = os.path.join(self.frame_raw,"post","points_compressed_ouster.npy")
        if not os.path.exists(points_path):
            lidar = np.zeros((128, 1024, 3), dtype=np.float32)
            #return_dict["lidar"] = lidar
        else:
            if os.path.exists(
                os.path.join(self.frame_raw, "ouster", "points_kiss_5.npy")
            ):
                lidar = _np_load(
                    os.path.join(self.frame_raw, "ouster", "points_kiss_5.npy")
                )
            else:
                lidar = _np_load(
                    points_path
                )
            lidar = lidar.reshape(-1, 3)

            if os.path.exists(
                os.path.join(self.frame_raw, "ouster", "points_kiss_10_post.npy")
            ):
                lidar_gt = _np_load(
                    os.path.join(self.frame_raw, "ouster", "points_kiss_10_post.npy")
                )
                lidar_gt = lidar_gt.reshape(-1, 3)
            else:
                lidar_gt = lidar.copy()
            lidar_sf = _np_load(points_path)
            #return_dict["lidar"] = lidar_sf
            lidar_sf = lidar_sf.copy().reshape(-1, 3)
    
        sides_concat = [f"12_{side}" for side in self.ldr_sides] + self.hdr_sides

        meta_json = json.loads(
            open(os.path.join(self.frame_raw, "metadata.json"), "r").read()
        )

        for side in sides_concat:
            if not f"lucid_{side}" in meta_json["cameras"]:
                continue
            meta = meta_json["cameras"][f"lucid_{side}"]
            # original intrinsics (numpy array)
            K_orig = np.asarray(self.camera_intrinsics[side]).copy()
            # current image size from returned tensor
            w = int(return_dict[f"rgb_{side}"].shape[2])
            h = int(return_dict[f"rgb_{side}"].shape[1])

            # original image size assumptions (the dataset images are 1440x928 by default)

            # If this side is an LDR (12_) use rescale_input, otherwise hdr uses rescale_output
            if not side.startswith("12_"):
                K_orig[0, 2] = K_orig[0, 2] - (1440 - t_rct[2]) / 2
                K_orig[1, 2] = K_orig[1, 2] - (928 - t_rct[3]) / 2

            # default adjusted intrinsics start from original
            K_adj = K_orig.copy()

            if side.startswith("12_") and self.rescale_input is not None:
                H_si, W_si = self.rescale_input
                sx = float(W_si) / float(1024)
                sy = float(H_si) / float(768)
                K_adj[0, 0] = K_orig[0, 0] * sx
                K_adj[1, 1] = K_orig[1, 1] * sy
                K_adj[0, 2] = K_orig[0, 2] * sx
                K_adj[1, 2] = K_orig[1, 2] * sy
            elif (not side.startswith("12_")) and self.rescale_output is not None:
                H_so, W_so = self.rescale_output
                sx = float(W_so) / float(target_w)
                sy = float(H_so) / float(target_h)
                K_adj[0, 0] = K_orig[0, 0] * sx
                K_adj[1, 1] = K_orig[1, 1] * sy
                K_adj[0, 2] = K_orig[0, 2] * sx
                K_adj[1, 2] = K_orig[1, 2] * sy

            fx = float(K_adj[0, 0])
            cx = float(K_adj[0, 2])
            cy = float(K_adj[1, 2])
            sf_pyramid = None
            
            for lidar_points, lidar_type in [
                (lidar_sf, "lidar_sf"),
                (lidar, "lidar"),
                (lidar_gt, "lidar_gt"),
            ]:
                if lidar_points is None:
                    continue

                lidar_to_cam = torch.from_numpy(lidar_points.copy()).float()  # [N,3]
                
                lidar_to_cam = transform_points_torch(
                    lidar_to_cam * 1000, self.cam_lidar_transform[side]
                )
                

                if not self.refine_depth or "12" in side or lidar_type == "lidar_sf":
                    lidar_to_cam = project_points_on_camera(
                        lidar_to_cam,
                        focal_length=fx,
                        cx=cx,
                        cy=cy,
                        image_width=w,
                        image_height=h,
                    )
                    lidar_to_cam = depth_points_to_depth_map_torch(
                        lidar_to_cam,
                        width=w,
                        height=h,
                    )
                    

                else:
                    delta = 2 * 3.14159265 / 1024  # 2 degrees in radians
                    
                    splat_output = self.splat_fn(
                        pts_cam=lidar_to_cam / 1000.0,
                        K=torch.from_numpy(K_adj),
                        H=h,
                        W=w,
                        delta_theta=delta,
                        single_frame_depth=return_dict["lidar_sf_" + side].clone()
                        / 1000.0,
                        sf_pyramid=sf_pyramid,
                        tau_front=0.03,
                        tau0=0.010,  # base occlusion margin (m)
                        tau1=0.3,  # slope per meter (m/m), 원거리일수록 여유↑
                        k1=0.8,
                        k2=1e-4,
                        r_min=4,
                        r_max=10,
                        use_inverse_depth=True,
                        r_clean=4,
                        tau_clean0=0.01,
                        tau_clean1=0.02,
                    )
                    lidar_to_cam, sf_pyramid = splat_output
                    lidar_to_cam = lidar_to_cam * 1000

                return_dict[f"{lidar_type}_{side}"] = lidar_to_cam

            return_dict[f"K_{side}"] = torch.from_numpy(K_adj).float()  # [3,3]
            return_dict[f"E_{side}"] = torch.from_numpy(
                self.camera_extrinsics[side]
            ).float()  # [4,4]
            # dark noise compute
            if "12_" in side:
                fn = self.fn_ldr_dn
            else:
                fn = self.fn_hdr_dn

            if not "lucid_" + side in meta_json["cameras"]:
                print(f"[dark noise] camera lucid_{side} not in metadata.json")
                exposure_ms = -1
                gain_db = -1
                min_dn = 0.001
                return_dict[f"min_dn_{side}"] = min_dn
                return_dict[f"exposure_{side}"] = exposure_ms
                return_dict[f"shutter_{side}"] = exposure_ms
                return_dict[f"gain_{side}"] = gain_db
                continue

            
            exposure_ms = meta["ExposureTime"]
            gain_db = meta["Gain"]

            min_dn = fn(exposure_ms, gain_db)
            return_dict[f"min_dn_{side}"] = min_dn

            computed_exposure = exposure_ms * gain_to_linear(
                gain_db, mode="dB", db_rule="20log10"
            )
            return_dict[f"exposure_{side}"] = computed_exposure
            return_dict[f"shutter_{side}"] = exposure_ms
            return_dict[f"gain_{side}"] = gain_db
            

        for side in self.ldr_sides:
            if side == "rear":
                continue
            # if failed to load "side' image, copy with rear one
            if torch.sum(return_dict[f"rgb_12_{side}"]) == 0:
                return_dict[f"rgb_12_{side}"] = return_dict[f"rgb_12_rear"].clone()
                return_dict[f"lidar_12_{side}"] = return_dict[f"lidar_12_rear"].clone()
                return_dict[f"lidar_gt_12_{side}"] = return_dict[
                    f"lidar_gt_12_rear"
                ].clone()
                return_dict[f"lidar_sf_12_{side}"] = return_dict[
                    f"lidar_sf_12_rear"
                ].clone()
                return_dict[f"K_12_{side}"] = return_dict[f"K_12_rear"].clone()
                return_dict[f"E_12_{side}"] = return_dict[f"E_12_rear"].clone()
                return_dict[f"min_dn_12_{side}"] = return_dict[f"min_dn_12_rear"]
                return_dict[f"exposure_12_{side}"] = return_dict[f"exposure_12_rear"]
                return_dict[f"shutter_12_{side}"] = return_dict[f"shutter_12_rear"]
                return_dict[f"gain_12_{side}"] = return_dict[f"gain_12_rear"]
        # if hdr_right is missing, copy from hdr_left
        if torch.sum(return_dict[f"rgb_right"]) == 0:
            return_dict[f"rgb_right"] = return_dict[f"rgb_left"].clone()
            return_dict[f"lidar_right"] = return_dict[f"lidar_left"].clone()
            return_dict[f"lidar_gt_right"] = return_dict[f"lidar_gt_left"].clone()
            return_dict[f"lidar_sf_right"] = return_dict[f"lidar_sf_left"].clone()
            return_dict[f"K_right"] = return_dict[f"K_left"].clone()
            return_dict[f"E_right"] = return_dict[f"E_left"].clone()
            return_dict[f"min_dn_right"] = return_dict[f"min_dn_left"]
            return_dict[f"exposure_right"] = return_dict[f"exposure_left"]
            return_dict[f"shutter_right"] = return_dict[f"shutter_left"]
            return_dict[f"gain_right"] = return_dict[f"gain_left"]

        return return_dict


class LucidHexDataset(Dataset):

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
    ):
        super().__init__()
        self.scenes = scenes
        self.frames = []
        self.output_aspect_ratio = output_aspect_ratio
        
        self.calibration_dict = {}
        
        
        self.cache_new = cache_new
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
        for frame_folder in tqdm(
            os.listdir(scene_path),
            desc=f"Searching frames in {os.path.basename(scene_path)}",
        ):
            frame_path = os.path.join(scene_path, frame_folder)
            if not os.path.isdir(frame_path):
                continue
            post_path = os.path.join(frame_path, "post")
            if not os.path.isdir(post_path):
                continue
            if not os.path.exists(os.path.join(frame_path, "metadata.json")):
                continue

            if os.path.exists(os.path.join(scene_path,"test_subsets")) and not os.path.exists(
                os.path.join(scene_path,"test_subsets",frame_folder+"_thumbnail.jpg")
            ) and not os.path.exists(
                os.path.join(scene_path,"test_subset",frame_folder+"_thumbnail.jpg")
            ):
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
        return len(self.frames)

    def __getitem__(self, idx):
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
                compiled_splat_fn=self.compiled_splat_fn,
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
