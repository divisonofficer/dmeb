#!/usr/bin/env bash
# Arrange third-party depth/HDR backbones under third_party/modules/ so that the
# DMEB reference model's imports resolve. Edit the URLs/refs if upstream moves.
#
# Usage:  bash third_party/setup_third_party.sh
#         export PYTHONPATH="$PWD/third_party:$PYTHONPATH"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MROOT="$HERE/modules"
mkdir -p "$MROOT/depth_densify" "$MROOT/monodepth" "$MROOT/hdr"

clone() {  # clone <url> <dest>
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then echo "[skip] $dest exists"; return; fi
  echo "[clone] $url -> $dest"
  git clone --depth 1 "$url" "$dest"
}

# TODO: pin these to the exact upstream URLs/commits you used for the paper.
clone https://github.com/DepthAnything/PromptDA.git          "$MROOT/depth_densify/PromptDA"
clone https://github.com/DepthAnything/Depth-Anything-V2.git "$MROOT/monodepth/DepthAnythingV2"
# AFUNet / DepthPrompting / BPNet: add their upstream URLs here if available.
# clone <AFUNet-url>        "$MROOT/hdr/AFUNet"
# clone <DepthPrompting-url> "$MROOT/depth_densify/DepthPrompting"
# clone <BPNet-url>          "$MROOT/depth_densify/BPNet"

# Package markers so `modules.*` is importable.
for d in "$MROOT" "$MROOT/depth_densify" "$MROOT/monodepth" "$MROOT/hdr"; do
  touch "$d/__init__.py"
done

echo "[done] add to PYTHONPATH:  export PYTHONPATH=\"$HERE:\$PYTHONPATH\""
