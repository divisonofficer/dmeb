# Known Issues, Limitations, and Exclusions

## Scope of this release
This is a **processed benchmark release**. It contains synchronized, calibrated,
undistorted data sufficient to reproduce the paper. It does **not** include raw
ROS bags or raw per-sensor streams.

## Limitations
- **Ultra DR ceiling.** Burst-based GT can contain radiance outside the valid
  exposure range of every single-shot LDR input. An observation-based method
  cannot recover unobserved radiance, so improvement margins on Ultra DR are
  smaller by construction — this is a property of the setting, not a bug.
- **iPhone FoV mismatch.** Wide/ultrawide/telephoto lenses differ in field of
  view; evaluation is limited to the common-overlap region. iPhone is a consumer
  proof-of-concept, not a primary benchmark.
- **Low-light depth.** Monocular depth priors degrade in low light; metric LiDAR
  anchoring is used to stabilize them. Residual depth error remains in extreme
  dark regions.
- **Robot rig is a superset.** The multi-camera/multi-depth rig is a benchmark
  superset for comparing camera-count and depth-modality subsets, **not** a
  deployment minimum. The practical minimum is: calibrated cameras with
  complementary exposures + overlapping FoV + depth.

## Sessions excluded from robot benchmark metrics
Sessions without populated pseudo-GT (`post/rgb_lucid_*.exr`) are excluded:
`10_28_20_05`, `10_28_20_07`, `10_29_14_56`, `10_29_14_59`, and all `11_11_*`.

## Privacy review

Robot and iPhone data were captured **inside the university campus during
off-hours with no people present**, by design. As a result no personally
identifiable subjects appear in the release.

- [x] Faces — none present (off-hours capture, no people in frame).
- [x] Vehicle license plates — none present in the captured scenes.
- [x] Indoor monitor/screen content — N/A (campus exterior/interior, no PII screens).
- [x] iPhone EXIF / GPS — stripped from released images.
- [x] CARLA assets — engine assets not redistributed (see LICENSE).
- [x] Choi et al. data — not redistributed (link + protocol only).

No scenes/frames required masking or removal for privacy.

## Numbers to re-verify against source logs
<!-- TODO FINAL: cross-check before camera-ready -->
- Reference ckpt = epoch 466; paper headline single-shot multi-camera =
  **39.40 dB** (consistent across project page, `benchmark_protocol.md`, and
  `code/checkpoints/README.md`).
- 2-view ablation value (rebuttal Table R3 lists 31.67 dB) vs. final table.
- Synthetic dynamic-range claim (body "~100 dB" vs. rebuttal ">150 dB") —
  unify under one DR definition.
