# DMEB — Reference Code

Inference + evaluation for the DMEB benchmark reference pipeline. This package
reproduces the main result table; it is the **reference method**, not the central
contribution (the dataset/benchmark is). Training code is not included.

## Contents
```
code/
├── inference.py         # run the DMEB model over a session → HDR EXR predictions
├── eval.py              # score predictions vs GT (PSNR/PU-PSNR/SSIM/LPIPS, region-aware)
├── dataloader_real.py   # LucidHexDataset (robot subsets)
├── dataloader_real_ultra.py
├── models/              # PromptDA_MV_All + submodules (the reference model)
├── common/              # metrics, collation, model factory, tonemap
├── utils/               # image/point-cloud helpers (vendored, scipy-only)
├── configs/             # evaluation and ablation configs
├── checkpoints/         # reference checkpoint (download — see checkpoints/README.md)
└── third_party/         # external backbones for inference (PromptDA, DAv2, AFUNet)
```

## Install
```bash
pip install -r requirements.txt
```

## Evaluate predictions (self-contained, no GPU/backbones needed)
```bash
python eval.py --pred /path/to/predictions --gt /path/to/subset/10_29_18_34 \
               --pred_glob '**/*.exr' --csv results.csv
```
`eval.py` is the canonical scorer and matches the paper's metric definitions
(linear-domain PSNR; Reinhard tone-map for SSIM/LPIPS; full/saturated/dark
regions). See `../dataset/benchmark_protocol.md`.

## Run the reference model (needs GPU + checkpoint + backbones)
```bash
# one-time: arrange third-party backbones and set PYTHONPATH
bash third_party/setup_third_party.sh
export PYTHONPATH="$PWD/third_party:$PWD:$PYTHONPATH"

# download the reference checkpoint -> checkpoints/dmeb_ref.pth (see checkpoints/README.md)
export DMEB_CKPT="$PWD/checkpoints/dmeb_ref.pth"

python inference.py --data_root /path/to/robot_modest_dr --session 10_29_18_34 \
                    --output out/
python eval.py --pred out/ --gt /path/to/robot_modest_dr/10_29_18_34
```

`--mode rs` (default) is the depth+LiDAR reference; `--mode flow` is the
no-depth ablation.

## Notes
- The model is constructed only via `common/dmeb_factory.make_dmeb_model` so the
  architecture always matches the checkpoint.
- Paths are passed as arguments; nothing is hard-coded to the authors' machine.
