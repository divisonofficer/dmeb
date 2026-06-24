# Calibration, Synchronization, and Exposure Metadata

## Cameras

Each robot timestamp provides undistorted calibration for all eight cameras in
`post/calib_undist_<cam>.yaml`, where `<cam>` is one of:

```
lucid_12_left   lucid_12_left_sub   lucid_12_right  lucid_12_right_sub
lucid_12_rear   lucid_12_rear_sub   lucid_left      lucid_right
```

Each YAML contains:
- **Intrinsics** `K` (3×3): focal lengths `fx, fy`, principal point `cx, cy`
  (pixels), for the **undistorted** image (`rgb_lucid_12_*_4096.tiff`).
- **Extrinsics** `T` (4×4): rigid transform placing the camera in the common rig
  frame. Use these to warp between views. The reference camera defines the output
  frame (see [`benchmark_protocol.md`](benchmark_protocol.md)).
- **Distortion** is already removed in the `*_4096` images; raw distortion
  coefficients (if needed) are retained in the YAML.

Synthetic scenes store the equivalent in `calibration.npz`
(`K`, `T` per view); iPhone in `calibration.json`.

## Depth sensor extrinsics
The LiDAR (Ouster) and active-stereo (RealSense) frames are related to the
reference camera by transforms stored in the same calibration files. The
post-processed `post/points_compressed_ouster.npy` and `post/realsense_depth.npy`
are already expressed in / projectable to the reference camera.

## Synchronization
Each camera entry in `metadata.json` carries two clocks:
- `timestamp_ns` — sensor hardware clock.
- `timestamp_ns_pc` — host PC clock at receipt.

Frames within a timestamp directory are grouped by host-clock proximity. The
folder name `HH_MM_SS_mmm` is the frame group's nominal capture time. Use
`timestamp_ns_pc` to verify per-camera skew when strict alignment matters.

## Exposure / gain
- `ExposureTime`: **microseconds** (µs).
- `Gain`: **decibels** (dB).

The "main" and "sub" streams of each camera differ in exposure (and possibly
gain); read the exact values per frame from `metadata.json`. Exposure spacing is
**not** assumed constant across scenes — always read it from metadata rather than
hard-coding.

## Putting it together (warp a source view into the reference view)
```python
# pseudo-code; see code/dataloader.py + code/models for the real implementation
K_ref, T_ref = load_calib(ref_cam)
K_src, T_src = load_calib(src_cam)
depth_ref    = load_depth_in_ref_frame(timestamp)   # meters
# back-project ref pixels to 3D, transform into src, project with K_src
```
