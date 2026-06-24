# Third-party modules (for `inference.py`)

The DMEB **reference model** (`models/PromptDAMulti.py`) imports a few external
depth/HDR backbones. These are released by their original authors under their own
licenses and are **not** redistributed in this repository. `eval.py` does **not**
need them — only `inference.py` (running the model) does.

The model expects these import paths to resolve:

| Import path | Upstream project | License |
|---|---|---|
| `modules.depth_densify.PromptDA.promptda.promptda` | PromptDA | see upstream |
| `modules.monodepth.DepthAnythingV2.depth_anything_v2.dpt` | Depth-Anything-V2 | Apache-2.0 |
| `modules.hdr.AFUNet.models.AFUNet` | AFUNet | see upstream |
| `modules.depth_densify.DepthPrompting.model_list` | DepthPrompting (optional) | see upstream |
| `modules.depth_densify.BPNet` | BPNet (optional) | see upstream |

## Setup

Arrange the upstream repositories under `third_party/modules/` to match the
import paths above, then add `third_party/` to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/third_party:$PYTHONPATH"
bash third_party/setup_third_party.sh   # clones the repos into the expected layout
```

Pretrained backbone weights (PromptDA, Depth-Anything-V2) are downloaded by their
own scripts; see each upstream README. Place the DMEB reference checkpoint per
`../checkpoints/README.md`.

> If you only need to score predictions you already have, skip this entirely and
> use `eval.py`.
