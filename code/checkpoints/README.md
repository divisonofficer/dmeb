# DMEB reference checkpoint

<!-- TODO FINAL: fill [DRIVE-LINK] + [MD5] from the uploaded checkpoint -->

Reference checkpoint: **epoch 466** (teacher-anchored real fine-tune), merged and
directly loadable. Under the benchmark protocol it produces the paper's headline
single-shot multi-camera result of **39.40 dB** on the robot test split.

Download it and place it here as `dmeb_ref.pth` (or point `--ckpt` / `$DMEB_CKPT`
at it):

| File | Epoch | PSNR (single-shot multi-camera) | Link | md5 |
|---|---|---|---|---|
| `dmeb_ref.pth` | 466 | 39.40 dB | [DRIVE-LINK] | [MD5] |

> Source: merged `dmeb_real_466_anchor_merged_step562600.pth`. Evaluate under
> `../../dataset/benchmark_protocol.md` to reproduce 39.40 dB.

## Load

```python
import torch
from common.dmeb_factory import make_dmeb_model

model = make_dmeb_model(mode="rs").cuda().eval()
sd = torch.load("checkpoints/dmeb_ref.pth", map_location="cpu")
sd = sd.get("model_state_dict", sd)
model.load_state_dict(sd, strict=False)
```

The canonical constructor kwargs live in `common/dmeb_factory.py` (`DMEB_KWARGS`)
and must match the checkpoint's training architecture, or keys will mismatch.

## Verify
```bash
md5sum checkpoints/dmeb_ref.pth   # compare against the table above
```
