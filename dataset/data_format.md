# Data Format

## Robot subsets (Modest DR / Ultra DR)

Each archive expands to a tree of **sessions → timestamps**:

```
<session>/                         # e.g. 10_29_18_34
  <timestamp>/                     # e.g. 18_26_53_930  (HH_MM_SS_mmm)
    lucid_12_left/image.tiff       # camera "left",  main exposure  (LDR input)
    lucid_12_left_sub/image.tiff   # camera "left",  sub  exposure  (LDR input)
    lucid_12_right/image.tiff      # camera "right", main exposure
    lucid_12_right_sub/image.tiff  # camera "right", sub  exposure
    lucid_12_rear/image.tiff       # camera "rear",  main exposure
    lucid_12_rear_sub/image.tiff   # camera "rear",  sub  exposure
    lucid_left/image.tiff          # HDR reference camera (high bit depth)
    lucid_right/image.tiff         # HDR reference camera (high bit depth)
    ouster/                        # LiDAR
      points.npy  lidar_ts.npy  imu_av.npy  imu_la.npy  imu_ts.npy
    realsense/                     # active stereo
      depth.npy  rgb.png
    metadata.json                  # per-camera ExposureTime, Gain, timestamps, pixel_format
    post/                          # processed / derived products
      calib_undist_<cam>.yaml      # undistorted intrinsics/extrinsics (8 cameras)
      rgb_lucid_12_<cam>_4096.tiff  # undistorted LDR (4096 px)
      rgb_lucid_left.exr           # HDR pseudo-GT, depth-registered  (reference)
      rgb_lucid_right.exr          # HDR pseudo-GT, depth-registered
      points_compressed_ouster.npy # LiDAR points projected/compressed
      realsense_depth.npy          # active-stereo depth (post-processed)
      ouster_points3d.png, realsense_depth_magma.png   # visualizations
```

For **Ultra DR**, the LDR inputs and GT are stored as `.exr` (linear HDR) rather
than `.tiff`; the directory layout is otherwise identical.

### File types
| Extension | Meaning | Notes |
|---|---|---|
| `.tiff` | LDR camera image | `lucid_12_*` 12-bit Bayer→RGB; `*_4096` undistorted |
| `.exr`  | Linear HDR | reference GT (`rgb_lucid_*`) and Ultra DR inputs |
| `.npy`  | depth / point clouds / IMU | float arrays, see below |
| `.png`  | preview/visualization only | not used for metrics |
| `.yaml` | calibration | see [`calibration.md`](calibration.md) |
| `metadata.json` | per-frame sensor metadata | exposure, gain, timestamps |

### `metadata.json` (per timestamp)
```json
{
  "frame_timestamp": 1761730020.10,
  "frame_timestamp_ns": 1761730020103848948,
  "cameras": {
    "lucid_left":   { "ExposureTime": 10002.9, "Gain": 41.2, "pixel_format": "BayerRG24", ... },
    "lucid_12_left":{ "ExposureTime": 59545.5, "Gain": 42.5, "pixel_format": "BayerRG12", ... },
    ...
  }
}
```
`ExposureTime` is in **microseconds**, `Gain` in **dB**. `timestamp_ns` is the
sensor clock; `timestamp_ns_pc` is the host clock (used for synchronization).

### Depth arrays
- `realsense/depth.npy`, `post/realsense_depth.npy` — `float32`, **meters**;
  invalid pixels are `0` (or `NaN` after post-processing — check `np.isfinite`).
- `ouster/points.npy` — `(N, 3)` or `(N, 4)` LiDAR points in the LiDAR frame;
  `post/points_compressed_ouster.npy` is projected to the reference camera.
- Coordinate frames and the transform to the reference camera are in the
  `post/calib_undist_*.yaml` files.

---

## Synthetic (CARLA)

```
<scene>/                           # e.g. example_hex_type1_extended_00
  hdr_<view>/<idx>.exr             # rendered LDR/HDR input, main exposure
  hdr_<view>_sub/<idx>.exr         # sub exposure
  ground_truth_depth_<view>/<idx>.exr
  ground_truth_depth_<view>_sub/<idx>.exr
  calibration.npz                  # intrinsics/extrinsics for all views
```
Views: `left`, `right`, `rear`, and `mid` (the `mid` view exists only in
`type1_extended`). 50 frames per scene; main + sub exposure per view.

---

## iPhone

```
<scene>/
  wide/<idx>.<ext>  ultrawide/<idx>.<ext>  telephoto/<idx>.<ext>
  lidar/<idx>.npy
  hdr_gt/<idx>.exr                 # bracket-merged HDR reference
  overlap_mask/<idx>.png           # common-overlap valid region
  calibration.json
```

---

## Reading examples
See [`../code/dataloader.py`](../code/dataloader.py) for canonical loaders.
EXR is read with `OpenEXR`/`imageio`; TIFF with `tifffile`/`cv2`; `.npy` with
`numpy.load`.
