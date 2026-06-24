# Benchmark Protocol

This defines exactly how predictions are evaluated so that numbers are
reproducible and comparable across methods.

## 1. Task and output domain
Given multiple **input views at varying exposures** (and, where available, depth),
reconstruct the **linear HDR radiance** in the **reference-camera field of view**.

- The output is **not** a stitched panorama; it is defined in the reference
  camera's FoV (default reference view: `left`).
- A source view contributes to fusion only where its projection is valid;
  invalid/unreliable observations receive zero/low weight, followed by refinement.
- An observation-based reconstruction does not hallucinate radiance that no input
  view observed — relevant for the Ultra DR subset, where GT can exceed every
  single-shot input's range.

## 2. Valid regions and masks
Metrics are restricted to valid pixels. Three masks (see
[`../code/common/metrics.py`](../code/common/metrics.py)):

- **valid_mask** — pixels with valid reference HDR GT (provided / caller-supplied).
- **saturated_mask** — pixels where *any* input view is saturated (`max ≥ 0.98`).
- **dark_mask** — pixels underexposed in every input (`max ≤ 0.02`).

For **iPhone**, additionally intersect with the **common-overlap mask**
(`overlap_mask/`) because the wide/ultrawide/telephoto lenses have different FoVs.

Metrics are reported on `full` (valid), and decomposed into `saturated` and
`dark` regions via `metric_by_region(...)`.

## 3. Metrics
All primary metrics are computed in the **linear HDR domain**:

| Metric | Definition | Code |
|---|---|---|
| **PSNR** | linear-domain PSNR over MSE, `max_val=1.0`, masked | `metrics.psnr` |
| **PSNR-μ** (tone-mapped) | Reinhard global tone-map, then PSNR | `metrics.psnr_tonemapped` |
| **PU-PSNR** | PU21-style perceptually-uniform encoding, then PSNR | `metrics.pu_psnr` |
| **SSIM** | Reinhard tone-map, then SSIM (scikit-image) | `metrics.ssim` |
| **LPIPS** | Reinhard tone-map to [-1,1], AlexNet | `metrics.lpips` |

Tone-mapping (Reinhard global, γ=2.2) is used **only** for SSIM/LPIPS and for
the PSNR-μ variant; PSNR/PU-PSNR are linear. HDR-VDP, if reported, is computed
with the external VDP3 toolbox (not bundled).

> **Figures vs. metrics.** Qualitative figures show tone-mapped predictions (not
> GT); quantitative metrics use the separate 24-bit HDR reference.

## 4. Training/eval fairness
All baselines (HDR-Transformer, AFUNet, HDRFlow, SAFNet) and DMEB are trained
**from scratch** under the **same** data partition, resolution, exposure setting,
valid mask, and metric domain.

## 5. Reference numbers (robot held-out evaluation set)
PSNR (dB):

| Method | Multi-shot single-camera | Single-shot multi-camera |
|---|--:|--:|
| HDR-Transformer | 31.26 | 33.61 |
| AFUNet | 29.64 | 32.25 |
| DMEB (reference) | 30.72 | **39.40** |

Depth-modality ablation (DMEB): flow 35.84 → mono 37.09 → mono+LiDAR (SV) 37.90
→ mono+LiDAR (MV) **39.40**.

Cross-platform (PSNR, dB): iPhone — HDR-T 23.35 / AFUNet 25.21 / DMEB 26.12;
Choi et al. (zero-shot) — 26.33 / 25.39 / 33.00.

## 6. Reproduce
```bash
cd ../code
python inference.py --data_root <subset> --ckpt <dmeb_ref.pth> --output out/
python eval.py --pred out/ --gt <subset>
```
