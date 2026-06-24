import torch
import torch.nn.functional as F

@torch.jit.ignore
def _nan_to_num_(t: torch.Tensor) -> torch.Tensor:
    # in-place는 피하고, dtype/디바이스 일치 유지
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def snr_channel_blind(
    linear_warped: torch.Tensor,
    sat_mask: torch.Tensor = None,
    valid_geo: torch.Tensor = None,
    eps: float = 1e-6,  # 살짝 키움: denorm/underflow 가드
    snr0: float = 3.0,
    gamma: float = 2.2,
    min_views: int = 4,
    use_log: bool = True,
    r_clip: float = 10.0,  # r 정규화 후 클립(극단치 가드)
) -> torch.Tensor:
    """
    Safe multi-view consistency-based SNR estimation.
    - 모든 division/log/tukey 연산 구간 FP32 강제 (AMP 안전)
    - NaN/Inf 정화, 빈 샘플/유효치 부족 fallback, 극단치 클램프
    - DDP에서 모든 rank가 동일한 경로를 타도록 분기 처리 일관 유지
    """

    assert linear_warped.dim() == 6, "Expected [B,N,NN,C,H,W]"
    B, N, NN, C, H, W = linear_warped.shape
    device = linear_warped.device
    dtype = linear_warped.dtype

    # 0) 입력 정화(앞단에서부터 NaN/Inf 차단)
    x_in = _nan_to_num_(linear_warped)

    # 1) Pseudo-linear space (여기는 dtype 유지 OK)
    #   pow/div가 들어가지만 값 범위가 [0,1]이라 AMP에서도 보통 안전.
    #   그래도 혹시 몰라 clamp 후 pow.
    x = x_in.clamp(0, 1).pow(1.0 / float(gamma))  # [B,N,NN,C,H,W]

    # 2) Valid mask with saturation + erosion
    if sat_mask is None:
        sat = (x_in > 0.999).to(dtype=dtype)
    else:
        sat = _nan_to_num_((sat_mask > 0.5).to(dtype=dtype))

    if valid_geo is not None:
        vg = valid_geo
        if vg.shape[-2:] != x.shape[-2:]:
            B_v, N_v, NN_v, C_v, H_v, W_v = vg.shape
            vg2 = vg.view(B_v * N_v * NN_v, C_v, H_v, W_v)
            vg2 = F.interpolate(vg2, size=(H, W), mode="nearest")
            vg = vg2.view(B_v, N_v, NN_v, C_v, H, W)
        if vg.shape[3] > 1:
            vg = vg[:, :, :, 0:1, :, :]
        vg = _nan_to_num_(vg.to(dtype=dtype))
        valid = vg * (1.0 - sat.mean(dim=3, keepdim=True))
    else:
        valid = 1.0 - sat.mean(dim=3, keepdim=True)

    # erosion: 3x3에서 모두 유효(엄격)
    kernel = torch.ones(1, 1, 3, 3, device=device, dtype=dtype)
    v2d = valid.view(B * N * NN, 1, H, W)
    v2d_eroded = (F.conv2d(v2d, kernel, padding=1) >= (9.0 - 1e-6)).to(dtype=dtype)
    valid = v2d_eroded.view(B, N, NN, 1, H, W)  # [B,N,NN,1,H,W]

    # 3) Channel average
    xg = x.mean(dim=3)  # [B,N,NN,H,W]
    v = valid.squeeze(3) > 0.5  # bool [B,N,NN,H,W]

    # 4) Shared per-target scale (q: 99.9%)
    #    빈 표본/NaN 안전 + 하한 가드
    xg_masked = torch.where(v, xg, torch.full_like(xg, float("nan")))
    q_list = []
    # quantile은 AMP 영향 없음. 수치 안전을 위해 FP32로 수행
    with torch.cuda.amp.autocast(enabled=False):
        xg_masked_f32 = xg_masked.float()
        for b in range(B):
            q_tgt = []
            for n in range(N):
                vals = xg_masked_f32[b, n].flatten()
                finite = torch.isfinite(vals)
                if finite.any():
                    valid_vals = vals[finite]
                    # 표본 부족 시 백업
                    if valid_vals.numel() > 100:
                        q_val = torch.quantile(valid_vals, 0.999)
                    else:
                        q_val = valid_vals.mean()  # 표본 적으면 평균으로 대체
                else:
                    q_val = torch.tensor(1e-3, device=device, dtype=torch.float32)
                q_tgt.append(torch.clamp(q_val, min=1e-3))  # 너무 작으면 분모 폭주
            q_list.append(torch.stack(q_tgt))
        q = torch.stack(q_list).view(B, N, 1, 1, 1).to(dtype=dtype)

    xg_n = (xg / q).clamp(0, 10.0)  # [B,N,NN,H,W]

    # 5) Log domain + source-axis statistics
    with torch.cuda.amp.autocast(enabled=False):
        y = torch.log(xg_n.float() + eps) if use_log else xg_n.float()
        y = _nan_to_num_(y)

        y_masked = torch.where(v, y, torch.full_like(y, float("nan")))
        med = torch.nanmedian(y_masked, dim=2, keepdim=True).values  # [B,N,1,H,W]
        dev_all = (y_masked - med).abs()
        mad = torch.nanmedian(dev_all, dim=2, keepdim=True).values
        mad = mad * 1.4826 + eps
        # NaN/Inf 방지 + 하한 보강
        mad = torch.clamp(_nan_to_num_(mad), min=1e-6)

        # per-source deviation
        dev = y - med  # [B,N,NN,H,W]

        # E) shallow smoothing on dev/mad (통계만 살짝)
        dev_s = F.avg_pool2d(dev.view(B * N * NN, 1, H, W), 3, 1, 1).view(
            B, N, NN, H, W
        )
        mad_s = F.avg_pool2d(mad.view(B * N, 1, H, W), 3, 1, 1).view(B, N, 1, H, W)
        mad_s = torch.clamp(mad_s, min=1e-6)

        # Tukey biweight
        c = 4.685
        r = dev_s / (mad_s + eps)  # [B,N,NN,H,W]
        r = torch.clamp(r, min=-r_clip, max=r_clip)
        # w_tukey = (1 - (r/c)^2)^2, r in [-c,c]만 유효
        rc = r / c
        w_tukey = (1.0 - rc * rc).clamp_min(0.0)
        w_tukey = w_tukey * w_tukey

        # SNR proxy (FP32로 계산 후 원래 dtype으로)
        snr_proxy = w_tukey / (1.0 + r.abs() + eps)
        snr_proxy = _nan_to_num_(snr_proxy).to(dtype=dtype)

    # 6) Fallback for insufficient sources
    valid_count = v.sum(dim=2, keepdim=True)  # [B,N,1,H,W]
    has_enough = (valid_count >= min_views).to(dtype=dtype)

    bright = xg_n.clamp(0, 1)  # [B,N,NN,H,W]
    fallback = bright / (bright + 0.1)  # 부드러운 대체
    fallback = _nan_to_num_(fallback)

    snr_combined = has_enough * snr_proxy + (1.0 - has_enough) * fallback

    # 7) Suppress dark/invalid regions
    # (med는 FP32, use_log 기준으로 임계치 계산)
    if use_log:
        dark_thresh = torch.log(torch.as_tensor(0.005, device=device, dtype=med.dtype))
        dark_mask = (
            torch.where(torch.isfinite(med), med, torch.zeros_like(med)) < dark_thresh
        ).to(dtype=dtype)
    else:
        dark_mask = (
            torch.where(torch.isfinite(med), med, torch.zeros_like(med)) < 0.005
        ).to(dtype=dtype)

    snr_combined = snr_combined * (1.0 - dark_mask)
    snr_combined = snr_combined * v.to(dtype=dtype)

    # 8) 최종 정화 + 차원 확장
    snr_norm = snr_combined.unsqueeze(3)  # [B,N,NN,1,H,W]
    snr_norm = torch.clamp(_nan_to_num_(snn := snr_norm), 0.0, 1.0)

    # 일관된 메모리 포맷(성능/경고 완화용)
    return snr_norm.contiguous()
