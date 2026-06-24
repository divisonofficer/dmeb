<!-- TODO FINAL: fill [N] from the v1.0 upload before camera-ready -->

# DMEB Dataset Card

A multi-view, varying-exposure HDR benchmark for robot and consumer vision.
This is a **processed benchmark release (v1.0)**: synchronized, calibrated, and
undistorted data sufficient to reproduce every table and figure in the paper.
Raw ROS bags and raw sensor streams are **not** included.

- **License:** datasets CC BY-NC 4.0; code MIT (see [`../LICENSE`](../LICENSE)).
- **Access:** direct download, no request form, no login/VPN required.
- **Citation:** see [`../CITATION.cff`](../CITATION.cff).

---

## 1. Release-Scope Matrix

| Subset | Public | Scenes/Sessions | Frames | Sensors / inputs | HDR GT | Depth | Archive | Not included |
|---|---|---|---|---|---|---|---|---|
| Robot — Modest DR | ✅ v1.0 | [N] sessions | [N] | 3 cam × 2 exp (TIFF) + HDR ref cam (EXR) | depth-registered EXR | LiDAR, active-stereo, iToF | [Google Drive](https://drive.google.com/drive/folders/1DJ6mOw3Y5VVNtvdoLCfY2SMBwGZkZk1z?usp=drive_link) | raw ROS bags |
| Robot — Ultra DR | ✅ v1.0 | [N] sessions | [N] | single-shot LDR (EXR) | burst-based EXR | LiDAR, active-stereo | [Google Drive](https://drive.google.com/drive/folders/16HsC-HKVJ110RjrT_gNQ9ewEp5HLL7SQ?usp=drive_link) | raw burst frames |
| Synthetic — CARLA | ✅ v1.0 | [N] scenes | [N] | rendered multi-view LDR (EXR) | rendered EXR (extreme DR) | metric + sensor-like | [Google Drive](https://drive.google.com/drive/folders/14broXSegL1aa204akX_ft8NZkQH60Jq-?usp=drive_link) | CARLA engine assets |
| iPhone | ✅ v1.0 | [N] scenes | [N] | wide/ultrawide/telephoto | bracket-generated | LiDAR | [Google Drive](https://drive.google.com/drive/folders/1vW8jDhCQcoK5klad48aCyhtoI0oyttze?usp=drive_link) | — |

**External (not redistributed):** *Choi et al., CVPR'25* — obtain from the
original authors; we provide only the evaluation protocol and adapter.

Every archive ships with a `MANIFEST.md5` (checksums) and a `meta.json`
(scene/frame counts, sensor list).

The full dataset root is available as a
[Google Drive folder](https://drive.google.com/drive/folders/1t1Hy2oellaAWjeIeyXrrq6RlCGXcOPB-?usp=drive_link).

---

## 2. Subset details

### Robot — Modest DR / Ultra DR
Captured with a multi-camera robot rig. Three machine-vision cameras
(`left`, `right`, `rear`), each with a **main** and **sub** exposure stream, plus
two high-bit-depth HDR **reference** cameras (`lucid_left`, `lucid_right`) used to
form pseudo-GT. Depth is provided simultaneously from an Ouster LiDAR, a
RealSense (active stereo), and a Helios iToF sensor.

- Modest DR uses TIFF LDR inputs with moderate exposure spacing.
- Ultra DR uses EXR single-shot inputs whose GT (from exposure bursts) can exceed
  the dynamic range of any single LDR input — by construction, an observation-based
  method cannot hallucinate radiance that no input observed (a documented limit).

### Synthetic — CARLA
Rendered hexagonal multi-view scenes with controllable exposure spacing, depth
noise, and motion, reaching dynamic ranges difficult to obtain in hardware. Both
metric depth and sensor-like (noisy) depth are provided, with full camera
configuration and generation metadata.

### iPhone
Consumer **proof-of-concept** subset: wide / ultrawide / telephoto inputs with
LiDAR and bracket-generated HDR GT. Because the lenses have different fields of
view, evaluation is restricted to the **common-overlap region** with valid masks
(see [`benchmark_protocol.md`](benchmark_protocol.md)). Not a primary benchmark.

---

## 3. Modality availability

| Modality | Modest | Ultra | Synthetic | iPhone |
|---|:--:|:--:|:--:|:--:|
| Multi-view LDR inputs | ✅ | ✅ | ✅ | ✅ |
| HDR ground truth | ✅ | ✅ | ✅ | ✅ |
| LiDAR depth | ✅ | ✅ | ✅(sim) | ✅ |
| Active-stereo depth | ✅ | ✅ | — | — |
| iToF depth | ✅ | — | — | — |
| Metric GT depth | — | — | ✅ | — |
| Intrinsics / extrinsics | ✅ | ✅ | ✅ | ✅ |
| Exposure / gain metadata | ✅ | ✅ | ✅ | ✅ |
| Valid masks | ✅ | ✅ | ✅ | ✅ |

---

## 4. Evaluation Partition

- **Robot test holdout:** `10_29_18_34` (most recent session).
- **Robot train/val:** GT-bearing sessions `10_29_15_*` / `10_29_17_*` /
  `10_29_18_*`; two most-recent sessions held out as val.
- Sessions without pseudo-GT are excluded from benchmark metrics.

All baselines and DMEB are trained/evaluated on the **same** data partition,
resolution, exposure setting, valid mask, and metric domain.

---

## 5. How GT is generated

- **Robot:** the high-bit-depth reference cameras provide depth-registered HDR
  references; Ultra DR additionally fuses exposure bursts.
- **Synthetic:** HDR radiance is rendered directly by the engine.
- **iPhone:** exposure brackets are merged to an HDR reference.

Quantitative metrics use a separate 24-bit HDR reference, **not** the tone-mapped
visualizations shown in the paper figures.

---

## 6. Pointers

- Directory layout & file formats → [`data_format.md`](data_format.md)
- Calibration, sync, exposure/gain units → [`calibration.md`](calibration.md)
- Reference-view definition, masks, metrics → [`benchmark_protocol.md`](benchmark_protocol.md)
- Limitations & excluded data → [`known_issues.md`](known_issues.md)
