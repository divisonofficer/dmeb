# Ablation Study Configuration Files

이 디렉토리에는 각 depth 모듈의 ablation study를 위한 설정 파일이 있습니다.

## 사용 가능한 Config 파일

### 1. da_scalenet.yaml
- **Mode**: rs (DepthAnything + ScaleNet)
- **설명**: Learnable MLP로 monocular depth를 metric scale로 변환
- **특징**: 빠르고 효율적, 현재 baseline
- **Weight**: DepthAnythingV2 ViT-B (자동 로드)

### 2. da_ransac.yaml
- **Mode**: ransac (DepthAnything + RANSAC)
- **설명**: Geometric soft RANSAC으로 scale fitting
- **특징**: 비학습 방법, MLP와 비교용
- **Weight**: DepthAnythingV2 ViT-B (자동 로드)

### 3. promptda_vitb.yaml
- **Mode**: promptda (PromptDA)
- **설명**: DINOv2 + DPT 기반 sparse-to-dense depth completion
- **특징**: 전통적인 depth completion 방법과 비교
- **Weight**: `/jarvis/modules/depth_densify/PromptDA/pretrained/promptda_small_t.ckpt`

### 4. depthprompt.yaml
- **Mode**: depthprompt (DepthPrompting)
- **설명**: Monocular backbone + CSPN iterative refinement
- **특징**: 가장 정확하지만 느림
- **Weight**: DepthAnythingV2 backbone + CSPN (자동 로드)

### 5. bpnet.yaml
- **Mode**: bpnet (BPNet)
- **설명**: Bilateral propagation network
- **특징**: Intrinsics 필요, CUDA 컴파일 필요
- **Weight**: `/jarvis/modules/depth_densify/BPNet/checkpoints/kitty.pth`

## 사용 방법

### Training
```bash
# Config 파일 사용
python train_depth_hex.py --config configs/ablation/promptda_vitb.yaml

# 또는 커맨드라인 인자로
python train_depth_hex.py --depth_mode promptda
```

### Inference Test
```bash
# Config 파일 사용
python test_ablation_inference.py \
    --depth_mode promptda \
    --test_data /path/to/test

# BPNet (CUDA 컴파일 후)
python test_ablation_inference.py \
    --depth_mode bpnet \
    --bpnet_ckpt modules/depth_densify/BPNet/checkpoints/kitty.pth \
    --test_data /path/to/test
```

### 모든 모드 테스트
```bash
bash run_quick_test.sh /path/to/test/data
```

## Pretrained Weights 위치

실제로 사용 가능한 weight 파일들:

```
/jarvis/
├── modules/
│   ├── monodepth/
│   │   └── DepthAnythingV2/
│   │       └── pretrained/
│   │           └── depth_anything_v2_vitb.pth  ✓ (rs, ransac, depthprompt에서 사용)
│   └── depth_densify/
│       ├── PromptDA/
│       │   └── pretrained/
│       │       └── promptda_small_t.ckpt  ✓ (promptda에서 사용)
│       ├── DepthPrompting/
│       │   └── (DepthAnythingV2 backbone 사용)
│       └── BPNet/
│           └── checkpoints/
│               └── kitty.pth  ✓ (bpnet에서 사용)
```

## Config 수정

각 config 파일을 복사해서 수정할 수 있습니다:

```bash
# 예: PromptDA prop_time 조정
cp configs/ablation/depthprompt.yaml configs/ablation/depthprompt_fast.yaml

# depthprompt_fast.yaml 수정:
# depthprompt_prop_time: 3 → 1  (더 빠르게)
```

## 성능 비교 (예상)

| Config | 속도 | RMSE↓ | MAE↓ | 특징 |
|--------|------|-------|------|------|
| da_scalenet.yaml | ⚡⚡⚡⚡ | ~0.82 | ~0.51 | Learnable, 빠름 |
| da_ransac.yaml | ⚡⚡⚡⚡ | ~0.91 | ~0.57 | Non-learnable, 매우 빠름 |
| promptda_vitb.yaml | ⚡⚡⚡ | ~0.75 | ~0.48 | DINOv2 기반, 정확 |
| depthprompt.yaml | ⚡⚡ | ~0.78 | ~0.50 | CSPN refinement |
| bpnet.yaml | ⚡⚡ | ~0.80 | ~0.52 | Bilateral propagation |

*실제 성능은 데이터셋에 따라 다를 수 있습니다.*

## Troubleshooting

### BPNet CUDA 컴파일 오류
```bash
cd /jarvis/modules/depth_densify/BPNet/exts
python setup.py install
```

### PromptDA weight 없음
```bash
ls /jarvis/modules/depth_densify/PromptDA/pretrained/promptda_small_t.ckpt
```

### Config 로딩 실패
YAML 파일 형식 확인 (들여쓰기 주의)
