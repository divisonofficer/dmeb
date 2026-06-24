import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1) 간단 피처 인코더 ----------
class FeatEncoder(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.ReLU(inplace=True),
            )
        self.l1 = block(in_ch, base)          # 1/1
        self.d1 = nn.Conv2d(base, base, 3, 2, 1)  # down
        self.l2 = block(base, base*2)         # 1/2
        self.d2 = nn.Conv2d(base*2, base*2, 3, 2, 1)
        self.l3 = block(base*2, base*4)       # 1/4
        self.d3 = nn.Conv2d(base*4, base*4, 3, 2, 1)
        self.l4 = block(base*4, base*4)       # 1/8

    def forward(self, x):
        f1 = self.l1(x)            # [B,32,H,W]
        f2 = self.l2(self.d1(f1))  # [B,64,H/2,W/2]
        f3 = self.l3(self.d2(f2))  # [B,128,H/4,W/4]
        f4 = self.l4(self.d3(f3))  # [B,128,H/8,W/8]
        return [f1, f2, f3, f4]

# ---------- 2) 캐노니컬 생성기 (deformable-ish gating) ----------
class CanonicalBuilder(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats_list, masks_list):
        # feats_list: list of [B,Ch,H/4,W/4] across N sources (same target m)
        # masks_list: list of [B,1,H/4,W/4] validity/conf
        # 간단 게이팅 평균
        gates = [self.gate(f) * (m if m is not None else 1.0) for f, m in zip(feats_list, masks_list)]
        num = sum(g * f for g, f in zip(gates, feats_list))
        den = sum(g for g in gates) + 1e-6
        canon = num / den
        return self.post(canon)  # [B,Ch,H/4,W/4]

class CorrBlock(nn.Module):
    """
    Low-memory local correlation.
    - unfold 없이 오프셋을 순회하며 rolling shift로 상관을 계산
    - 결과 K=(2r+1)^2 채널로 쌓아서 반환
    """
    def __init__(self, radius=4, chunk_k=None, detach_ref=True, use_amp=True):
        super().__init__()
        self.r = radius
        self.chunk_k = chunk_k  # ex) 9 -> 오프셋 9개씩 청킹. None이면 전체 루프
        self.detach_ref = detach_ref
        self.use_amp = use_amp

    @staticmethod
    def _shift2d(x, dy: int, dx: int):
        """
        x: [B,C,H,W]
        dy,dx: 정수 픽셀 시프트 (위/왼쪽이 음수, 아래/오른쪽이 양수)
        replicate padding으로 경계 보정
        """
        B, C, H, W = x.shape
        # 필요한 쪽에만 패딩
        pad_t, pad_b = max(dy, 0), max(-dy, 0)
        pad_l, pad_r = max(dx, 0), max(-dx, 0)

        # [left, right, top, bottom] 순서
        x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='replicate')

        # 패딩 뒤에는 시작좌표 = pad - shift 가 맞습니다!
        y0 = pad_t - dy
        x0 = pad_l - dx
        y1 = y0 + H
        x1 = x0 + W

        # 안전장치(이 범위가 어긋나면 빈 텐서가 나와 폭=0/높이=0이 됩니다)
        # 그래도 혹시 모를 잘못된 dy/dx를 막기 위해 clamp
        y0 = max(y0, 0); x0 = max(x0, 0)
        y1 = min(y1, x_pad.shape[2]); x1 = min(x1, x_pad.shape[3])

        return x_pad[:, :, y0:y1, x0:x1]

    def forward(self, src, ref):
        """
        src, ref: [B, C, H, W]  (1/4 또는 1/8 해상도)
        return  : [B, K, H, W], K=(2r+1)^2
        """
        if self.detach_ref:
            ref = ref.detach()

        B, C, H, W = src.shape
        r = self.r
        offsets = [(dy, dx) for dy in range(-r, r+1) for dx in range(-r, r+1)]
        K = len(offsets)

        outs = []
        # 청킹(선택): 메모리를 더 줄이고 싶으면 offsets를 chunk로 나눠 처리
        if self.chunk_k is None or self.chunk_k >= K:
            chunks = [offsets]
        else:
            chunks = [offsets[i:i+self.chunk_k] for i in range(0, K, self.chunk_k)]

        autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

        with autocast_ctx:
            for ch in chunks:
                buf = []
                for (dy, dx) in ch:
                    ref_s = self._shift2d(ref, dy, dx)        # [B,C,H,W]
                    # 채널 합으로 상관 (Flownet/RAFT 스타일 정규화)
                    corr = (src * ref_s).sum(dim=1, keepdim=True) / (C ** 0.5)  # [B,1,H,W]
                    buf.append(corr)
                outs.append(torch.cat(buf, dim=1))  # [B, len(ch), H, W]

        return torch.cat(outs, dim=1)  # [B,K,H,W]


class UpdateBlock(nn.Module):
    def __init__(self, ch=128, k2=81):  # r=4 → 81
        super().__init__()
        self.gru = nn.GRUCell(ch + k2 + 2, ch)
        self.head = nn.Conv2d(ch, 2, 3, padding=1)

    def forward(self, h, feat, corr, flow):
        B, _, H, W = feat.shape
        # corr, feat, flow를 붙여 GRU로
        x = torch.cat([feat, corr, flow], dim=1)          # [B, ch+k2+2, H, W]
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B*H*W, -1)
        h  = self.gru(x, h.reshape(B*H*W, -1))
        h2 = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        delta = self.head(h2)
        return h2, delta


class TLAFlow(nn.Module):
    """
    Targetless flow: 각 (m,n)에 대해 F_src(1/4) -> F_can(1/4) 로 잔여 플로우 예측.
    """
    def __init__(self, ch=64, iters=6, radius=4):
        super().__init__()
        self.corr = CorrBlock(radius=radius)
        self.up   = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upd  = UpdateBlock(ch=ch, k2=(2*radius+1)**2)
        self.ctx  = nn.Conv2d(ch, ch, 3, padding=1)
        self.iters = iters

    def forward(self, Fsrc, Fcan):
        # 둘 다 [B,128,H/4,W/4]
        B,C,h,w = Fsrc.shape
        h0 = torch.tanh(self.ctx(Fsrc))                   # [B,128,h,w]
        flow = torch.zeros(B, 2, h, w, device=Fsrc.device)
        for _ in range(self.iters):
            corr = self.corr(Fsrc, Fcan)                  # [B,K,h,w]
            h0, dflow = self.upd(h0, Fsrc, corr, flow)    # [B,128,h,w], [B,2,h,w]
            flow = flow + dflow
        flow_full = self.up(flow)                         # [B,2,H,W]
        return flow_full


import torch
import torch.nn as nn
import torch.nn.functional as F


def make_identity_grid(B, H, W, device, align_corners=True):
    if align_corners:
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
    else:
        xs = (torch.arange(W, device=device) + 0.5) / W * 2.0 - 1.0
        ys = (torch.arange(H, device=device) + 0.5) / H * 2.0 - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, H, W, 2).contiguous()
    return grid


class TargetlessAlignNet(nn.Module):
    def __init__(self, base=32, ch4=128, iters=6, radius=4):
        super().__init__()
        self.enc   = FeatEncoder(in_ch=3, base=base)     # f1,f2,f3 (f3: 1/4 scale)
        #self.canon = CanonicalBuilder(ch=ch4)            # 간단 가중평균 기반
        self.flow  = TLAFlow(ch=ch4, iters=iters, radius=radius)

    @staticmethod
    def _half(x, mode="area"):
        # 이미지/float류는 area가 부드럽고, 마스크는 nearest 권장
        if mode == "nearest":
            return F.interpolate(x, scale_factor=0.5, mode="nearest")
        return F.interpolate(x, scale_factor=0.5, mode="area")

    @staticmethod
    def _flow_pix_to_norm(flow_pix, H, W, align_corners=True):
        """
        flow_pix: [B,2,H,W] (픽셀 단위: +1은 우측/하단 1픽셀 이동)
        반환: [B,H,W,2] (NDC 단위로 변환된 delta grid)
        """
        if align_corners:
            # x: [-1,1] <-> [0..W-1], y: [-1,1] <-> [0..H-1]
            sx = 2.0 / max(W - 1, 1)
            sy = 2.0 / max(H - 1, 1)
        else:
            sx = 2.0 / W
            sy = 2.0 / H
        du = flow_pix[:, 0:1] * sx
        dv = flow_pix[:, 1:2] * sy
        return torch.stack([du, dv], dim=-1).squeeze(1)  # [B,H,W,2]

    def forward(
        self,
        ldr_warp_full,   # [B,M,N,3,H,W]  (각 타겟 m에 대해 모든 src n을 기하 워핑한 LDR)
        valid_geo_up,    # [B,M,N,1,H,W]  (기하 valid/sat 등)
        base_grid_full,  # [B,M,N,H,W,2]  (기하 기반 그리드; refine해서 사용할 것)
        padding_mode="border",
        align_corners=True,
    ):
        """
        반환:
          grid_refined : [B,M,N,H,W,2]    (base + flow 보정)
          ldr_refined  : [B,M,N,3,H,W]    (풀해상도에서 warp)
          flow_res     : [B,M,N,2,H,W]    (풀해상도 flow in pixels)
          flow_conf    : [B,M,N,1,H,W]    (간단 신뢰도: |flow| 대비 유효마스크 등)
        """
        B, M, N, C, H, W = ldr_warp_full.shape
        device = ldr_warp_full.device
        dtype  = ldr_warp_full.dtype

        # 1) 절반 해상도로 다운샘플
        ldr_half   = self._half(ldr_warp_full.reshape(B*M*N, C, H, W), mode="area") \
                        .reshape(B, M, N, C, H//2, W//2)
        vmask_half = self._half(valid_geo_up.reshape(B*M*N, 1, H, W), mode="nearest") \
                        .reshape(B, M, N, 1, H//2, W//2)
        

        # 2) 특징 추출(절반 해상도 → enc 내부에서 1/4로 더 줄어듦)
        x_bmn = ldr_half.reshape(B*M*N, C, H//2, W//2).contiguous()
        feats_list = self.enc(x_bmn)          # [f1,f2,f3]; f3를 사용
        f3_bmn = feats_list[-1]               # [BMN, ch4, Hh, Wh] where Hh=(H/2)/4 = H/8
        _, ch4, Hh, Wh = f3_bmn.shape

        # 3) 타겟별 canonical feature 만들기 (N에 대한 가중평균)
        #    f3_bmn: [BMN,C,Hh,Wh] -> [B,M,N,C,Hh,Wh]
        f3_bmn = f3_bmn.reshape(B, M, N, ch4, Hh, Wh)
        v_f    = F.interpolate(
                    vmask_half.reshape(B*M*N, 1, H//2, W//2),
                    size=(Hh, Wh), mode="nearest"
                 ).reshape(B, M, N, 1, Hh, Wh)
        # CanonicalBuilder는 per-target(m) 리스트를 받지만, 메모리 절약 위해 벡터화 가중평균
        # gate 대신 간단 평균(원한다면 builder.gate를 한 번 더 호출해도 됨)
        w = v_f.clamp_min(1e-6)                       # [B,M,N,1,Hh,Wh]
        num = (w * f3_bmn).sum(dim=2)                 # [B,M,C,Hh,Wh]
        den = w.sum(dim=2)                            # [B,M,1,Hh,Wh]
        canon_bm = (num / den).contiguous()           # [B,M,C,Hh,Wh]

        # 4) 각 (m,n) 쌍에 대해 flow(절반 해상도) 추정
        #    Fsrc: [BMN,C,Hh,Wh], Fcan: [BMN,C,Hh,Wh] 로 정렬
        Fsrc = f3_bmn.reshape(B*M*N, ch4, Hh, Wh)
        Fcan = canon_bm.unsqueeze(2).expand(B, M, N, ch4, Hh, Wh) \
                               .contiguous().reshape(B*M*N, ch4, Hh, Wh)

        flow_half = self.flow(Fsrc, Fcan)
        
        flow_half = flow_half.reshape(B, M, N, 2, H//4, W//4)

        # 5) 풀 해상도로 업샘플 (×2)
        flow_full = F.interpolate(
            flow_half.reshape(B*M*N, 2, H//4, W//4),
            size=(H, W), mode="bilinear", align_corners=align_corners
        ).reshape(B, M, N, 2, H, W)

        # 6) base_grid_full + flow(NDC 변환) → refined grid
        delta_ndc = self._flow_pix_to_norm(
            flow_full.reshape(B*M*N, 2, H, W), H, W, align_corners=align_corners
        ).reshape(B, M, N, H, W, 2)
        grid_refined = base_grid_full + delta_ndc      # [B,M,N,H,W,2]

        # 7) 최종 워핑 (풀 해상도)
        ldr_refined = F.grid_sample(
            ldr_warp_full.reshape(B*M*N, C, H, W),
            grid_refined.reshape(B*M*N, H, W, 2),
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        ).reshape(B, M, N, C, H, W)

        # # 간단 confidence: 유효마스크 × exp(-|flow|/s)
        # flow_mag = torch.linalg.norm(flow_full, dim=3, keepdim=True)  # [B,M,N,1,H,W]
        # s = max(H, W) / 80.0
        # flow_conf = torch.exp(-flow_mag / max(s, 1e-6)) * valid_geo_up

        return grid_refined, ldr_refined #, flow_full, flow_conf


class WinnerTakeMostMixer(nn.Module):
    """
    Enhanced Winner-Take-Most mixer with over/dark split, tie-breaker and guardrails.

    Backwards-compatible: by default returns `lin_out` only (same as before).
    Set `return_aux=True` in forward to get (lin_out, w_final, aux).

    Inputs (kept compatible):
      lin_src   [B,N,3,H,W]
      ldr_src   [B,N,3,H,W]
      ldr_tgt   [B,1,3,H,W] or None
      valid_geo [B,N,1,H,W]
      w_trap    [B,N,1,H,W] or [B,N,2,H,W] (chan0=over_ok, chan1=dark_ok)
      snr_norm  [B,N,1,H,W]
      rerr      [B,N,1,H,W] or None

    Behavior:
      - If w_trap provides two channels (over,dark) they are used. Otherwise
        we conservatively treat both as equal to the provided w_trap channel.
      - Implements: well-exposed preference, over-vs-dark tiebreaker,
        SNR fallback, floor weights, dark suppression prior, and optional
        annealed temperature.
    """

    def __init__(
        self,
        tau=0.15,
        alpha_hard=0.7,
        ref_lambda=10.0,
        rerr_beta=4.0,
        eps=1e-8,
        tau_min_sum=1e-6,
        prefer_over_vs_dark=1.5,
        dark_margin=0.0,
        floor_weight=1e-6,
        anneal_tau=None,
        allow_empty_output=True,
    ):
        super().__init__()
        self.tau = tau
        self.alpha_hard = alpha_hard
        self.ref_lambda = ref_lambda
        self.rerr_beta = rerr_beta
        self.eps = eps
        self.tau_min_sum = tau_min_sum
        self.prefer_over_vs_dark = prefer_over_vs_dark
        self.dark_margin = dark_margin
        self.floor_weight = floor_weight
        self.anneal_tau = anneal_tau
        self.allow_empty_output = bool(allow_empty_output)

    @staticmethod
    def _ch1(x):  # [B,N,C,H,W] -> [B,N,1,H,W]
        if x.shape[2] == 1:
            return x
        return x.min(dim=2, keepdim=True).values

    def _split_trap(self, w_trap: torch.Tensor):
        """
        Interpret w_trap. If channel dim == 2, assume [over_ok, dark_ok].
        Otherwise fall back to conservative duplication.
        Returns: w_over_ok, w_dark_ok each [B,N,1,H,W]
        """
        if w_trap is None:
            return None, None
        if w_trap.shape[2] == 2:
            w_over_ok = w_trap[:, :, 0:1]
            w_dark_ok = w_trap[:, :, 1:2]
        elif w_trap.shape[2] == 6:
            w_over_ok = w_trap[:, :, 0:3]
            w_dark_ok = w_trap[:, :, 3:6]
        else:
            # conservative: use same signal for both if only one channel provided
            w_over_ok = w_trap
            w_dark_ok = w_trap
        return w_over_ok.clamp(0.0, 1.0), w_dark_ok.clamp(0.0, 1.0)

    def forward(
        self,
        lin_src,
        ldr_src,
        ldr_tgt,
        valid_geo,
        w_trap,
        snr_norm=None,
        rerr=None,
        step=None,
        return_aux=False,
    ):  # keep signature compatible
        B, N, C, H, W = lin_src.shape
        eps = self.eps

        # basic terms. Keep a raw validity/evidence tensor so pixels with no
        # reliable source can remain black instead of being filled by fallback.
        valid_raw = valid_geo.clamp(0.0, 1.0)
        vg = valid_raw.clamp_min(eps)
        sn_raw = None
        if snr_norm is not None:
            sn_raw = snr_norm.clamp(0.0, 1.0)
            sn = sn_raw.clamp_min(eps)
            vg = vg * sn

        # split trap into over/dark if possible
        w_over_ok, w_dark_ok = self._split_trap(w_trap)

        # well-exposed preference: both over_ok and dark_ok high
        if w_over_ok is None or w_dark_ok is None:
            # fallback: use valid * snr only
            w_well = (vg ).clamp_min(eps)
        else:
            w_well = (w_over_ok * w_dark_ok * vg).clamp_min(eps)
        
        

        # tie-breaker when both indicators are low: prefer over (avoid dark-noise)
        if w_over_ok is not None and w_dark_ok is not None:
            tie_cond = (w_over_ok < 0.5) & (w_dark_ok < 0.5)
            # smooth increasing function for preference (higher -> favor over)
            w_tiebreak = torch.where(
                tie_cond,
                self.prefer_over_vs_dark * torch.exp(-(1.0 - w_over_ok)).clamp_min(eps),
                torch.ones_like(w_over_ok),
            )
        else:
            w_tiebreak = torch.ones_like(vg)

        base = (w_well * w_tiebreak).clamp_min(eps)  # [B,N,1,H,W]
        if w_over_ok is not None and w_dark_ok is not None:
            evidence = (valid_raw * w_over_ok * w_dark_ok).clamp(0.0, 1.0)
        else:
            evidence = valid_raw
        if sn_raw is not None:
            evidence = evidence * sn_raw

        # reference agreement penalty/bonus
        if ldr_tgt is not None:
            ldr_tgt_exp = ldr_tgt.expand(B, N, C, H, W)
            diff = (ldr_src - ldr_tgt_exp).abs().mean(dim=2, keepdim=True)
            ref_w = torch.exp(-self.ref_lambda * diff).clamp_min(eps)
            base = base * ref_w

        # rerr penalty
        if rerr is not None:
            penal = torch.exp(-self.rerr_beta * rerr.clamp_min(0.0))
            base = base * penal

        # dark suppressor prior: if ldr_src is very low, apply a strong penalty
        if self.dark_margin is not None and self.dark_margin > 0.0 and ldr_src is not None:
            ldr_ch = self._ch1(ldr_src)  # collapse to single channel
            dark_mask = (ldr_ch < self.dark_margin).float()
            # penalize dark pixels but keep small floor (avoid zeroing-out entirely)
            dark_penalty = torch.where(dark_mask > 0.5, 0.1, 1.0)
            base = base * dark_penalty

        # fallback: old behavior filled empty pixels from geometric validity
        # alone, which preserved occlusion smears and amplified dark noise in
        # HDR_init. In allow_empty_output mode, empty evidence stays black.
        evidence_sum = evidence.sum(dim=1, keepdim=True)
        fallback_mask = evidence_sum < self.tau_min_sum
        has_evidence = (~fallback_mask).to(base.dtype)
        if (not self.allow_empty_output) and fallback_mask.any():
            base_fb = vg.clamp_min(eps)
            # broadcast fallback per-source
            base = torch.where(fallback_mask, base_fb, base)

        # floor weight to avoid zero logits where evidence exists. Empty pixels
        # are zeroed after the soft/hard winner calculation.
        base = base + self.floor_weight * has_evidence

        # logits & softmax (allow annealed tau callable)
        logits = torch.log(base + eps)
        tau = self.tau if self.anneal_tau is None else float(self.anneal_tau(step))
        soft = torch.softmax(logits / max(tau, 1e-6), dim=1)

        # hard top-1
        with torch.no_grad():
            idx = torch.argmax(soft, dim=1, keepdim=True)
            hard = torch.zeros_like(soft).scatter_(1, idx, 1.0)

        w_final = self.alpha_hard * hard + (1.0 - self.alpha_hard) * soft
        w_final = w_final / (w_final.sum(dim=1, keepdim=True) + eps)
        if self.allow_empty_output:
            w_final = w_final * has_evidence

        lin_out = (w_final * lin_src).sum(dim=1)

        aux = {
            "soft": soft,
            "hard_idx": idx.squeeze(1),
            "fallback_pix": fallback_mask.squeeze(1),
            "has_evidence": has_evidence.squeeze(1),
        }

        if return_aux:
            return lin_out, w_final, aux
        return lin_out


class ResBlock(nn.Module):
    """Residual block with dilation for multi-scale receptive field"""

    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=bias,
            ),
            nn.PReLU(channels),
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=bias,
        )
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out


class ScaledResidualSE(nn.Module):
    def __init__(self, c, r=8, gate="tanh", temp=0.3, residual=True, alpha_init=0.5):
        super().__init__()
        hid = max(1, c // r)
        self.fc1 = nn.Conv2d(c, hid, 1, bias=True)
        self.fc2 = nn.Conv2d(hid, c, 1, bias=True)
        #self.gate = gate
        #self.temp = float(temp)
        self.residual = residual
        #self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        
        self.gn_in  = nn.GroupNorm(num_groups=1, num_channels=c, affine=False)
        self.gn_out = nn.GroupNorm(num_groups=1, num_channels=c, affine=False)
        # Stable initialization: small fc1 and zero fc2 to start with g≈0 -> y≈f
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, f):
        # Perform SE computations in FP32 to avoid fp16/bf16 saturation underflow
        with torch.cuda.amp.autocast(enabled=False):
            s = f.mean((2, 3), keepdim=True).float()
            s = self.gn_in(s)
            z = self.fc2(F.relu(self.fc1(s), inplace=False))
            g = self.gn_out(z)
            # if z.device.type == "cuda" and z.device.index == 0:
            #     print("Z abs mean:", z.abs().mean().item())
            # if self.gate == "tanh":
            #     g = torch.tanh(self.temp * z)
            # else:
            #     g = torch.sigmoid(self.temp * z) * 2 - 1
            g = g.to(dtype=f.dtype)
            # if z.device.type == "cuda" and z.device.index == 0:
            #     print("G mean:", g.abs().mean().item())
            #     with torch.no_grad():
            #         g_over = g[g.abs() > 0.999]
            #         print("Ratio of saturated gates:", float(g_over.numel()) / float(g.numel()))
            

        if self.residual:
            # Residual gating ensures gradient flows back even if g saturates
            return f + (f * g)
        else:
            return f * g

class TopKHDRDenoiserUNet(nn.Module):
    """
    UNet-like HDR refiner focused on dark-region denoising.

    - UNet + skip + SE attention(잔차형 옵션) + depthwise separable conv
    - log-domain residual, gate/cap로 안전한 범위에서 보정
    - bfloat16 autocast로 인코더/디코더 가속, log/exp는 FP32

    Input per view: LDR(3) + LINEAR(3) + conf(1) + depth(1) + snr(1) + HDR_log(3) + brightness(1) = 13ch
    Total input: K × 13 channels
    Output: log-domain residual (3ch)
    """

    def __init__(self, K=3, per_view_ch=13, base=32, log_offset=1e-10,
                 use_residual_se=False, se_alpha_init=0.3, debug=False):
        super().__init__()
        self.K = K
        self.debug = debug
        in_ch = K * per_view_ch
        self.log_offset = log_offset
        self.use_residual_se = use_residual_se
        # Use mu-tonemap domain instead of natural log for HDR refinement
        # mu value chosen per user request
        self.mu = 10000.0
        # Precompute scalar denominator log(1+mu)
        self._log1p_mu = math.log1p(self.mu)

        # ----- Helper modules -----
        def dwpw(cin, cout, stride=1):
            """Depthwise separable conv: depthwise → pointwise"""
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=cin, affine=False),
                nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                nn.PReLU(cout),
            )

        
        # ----- Encoder -----
        self.e1 = dwpw(in_ch, base)             # H x W
        self.se1 = ScaledResidualSE(base, residual=True, temp=0.1, gate="tanh", alpha_init=se_alpha_init)
        self.e2 = dwpw(base, base * 2, stride=2)  # H/2 x W/2
        self.se2 = ScaledResidualSE(base * 2, residual=True, temp=0.1, gate="tanh", alpha_init=se_alpha_init)
        self.e3 = dwpw(base * 2, base * 4, stride=2)  # H/4 x W/4
        self.se3 = ScaledResidualSE(base * 4, residual=True, temp=0.1, gate="tanh", alpha_init=se_alpha_init)

        # optional: residual-style SE scaling (1 + α*se)
        if use_residual_se:
            self.se_alpha = nn.Parameter(torch.tensor(float(se_alpha_init)))
        else:
            self.register_parameter("se_alpha", None)

        # ----- Bottleneck -----
        self.bot = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=2, dilation=2),
            nn.PReLU(base * 4),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.PReLU(base * 4),
        )

        # ----- Decoder -----
        def up(cin, cout):
            """Upsample + conv"""
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(cin, cout, 3, 1, 1),
                nn.PReLU(cout),
            )

        self.d2 = up(base * 4, base * 2)                # H/2 x W/2
        self.d1 = up(base * 2, base)                    # H x W
        self.mix2 = nn.Conv2d(base * 4, base * 2, 3, 1, 1)  # concat(u2, f2)
        self.mix1 = nn.Conv2d(base * 2, base, 3, 1, 1)      # concat(u1, f1)

        # ----- Output residual in log-domain -----
        self.out = nn.Conv2d(base, 3, 3, 1, 1)
        nn.init.kaiming_normal_(self.out.weight, mode="fan_in", nonlinearity="linear")
        # 초기 보정량 매우 작게
        with torch.no_grad():
            self.out.weight.mul_(2e-3)
            nn.init.zeros_(self.out.bias)

        if not hasattr(self, "res_gate"):
            self.res_gate = nn.Parameter(torch.full((1, 3, 1, 1), -2.5))  # sigmoid≈0.018
        # # cap은 sigmoid로 (0,1) 소프트 제한
        if not hasattr(self, "res_cap_log"):
            self.res_cap_log = nn.Parameter(torch.tensor(-1.8))  # sigmoid(-1.8)≈0.141
        # # # log-scale 강도
        #self.log_alpha = nn.Parameter(torch.tensor(0.7))

    def forward(
        self,
        H_init,            # [B,N,3,H,W]
        ldr_warped,        # [B,N,N,3,H,W]
        linear_warped,     # [B,N,N,3,H,W]
        valid_mask=None,   # [B,N,N,1,H,W]
        depth_warped=None, # [B,N,N,1,H,W]
        confidence=None,   # [B,N,N,1,H,W]
        snr_norm=None,     # [B,N,N,1,H,W]
    ):
        # ===== 1) Build top-K tensor =====
        B, M, N, _, H, W = ldr_warped.shape

        # Top-K는 정밀도와 관계 적어 bfloat16 autocast로 진행해도 무방
        with torch.cuda.amp.autocast(False):
            if valid_mask is None:
                valid_mask = torch.ones(B, M, N, 1, H, W, device=H_init.device)

            # Quality: total valid pixels per pair
            quality_per_pair = valid_mask.sum(dim=[3, 4, 5])  # [B,N,N]

            # top-K
            K_actual = min(self.K, N)
            _, topk_indices = torch.topk(quality_per_pair, K_actual, dim=2)  # [B,N,K]

            # gather
            topk_idx_expanded = topk_indices.unsqueeze(3).unsqueeze(4).unsqueeze(5)  # [B,N,K,1,1,1]
            topk_idx_3ch = topk_idx_expanded.expand(B, M, K_actual, 3, H, W)
            topk_idx_1ch = topk_idx_expanded.expand(B, M, K_actual, 1, H, W)

            topk_ldr = torch.gather(ldr_warped, 2, topk_idx_3ch)       # [B,M,K,3,H,W]
            topk_lin = torch.gather(linear_warped, 2, topk_idx_3ch)    # [B,M,K,3,H,W]
            topk_conf = torch.gather(
                confidence if confidence is not None else valid_mask, 2, topk_idx_1ch
            )                                                           # [B,M,K,1,H,W]
            topk_depth = (
                torch.gather(depth_warped, 2, topk_idx_1ch)
                if depth_warped is not None
                else torch.zeros(B, M, K_actual, 1, H, W, device=H_init.device)
            )                                                           # [B,M,K,1,H,W]
            topk_snr = (
                torch.gather(snr_norm, 2, topk_idx_1ch)
                if snr_norm is not None
                else torch.ones(B, M, K_actual, 1, H, W, device=H_init.device) * 0.5
            )                                                           # [B,M,K,1,H,W]

            # HDR init guidance
            H_init_expanded = H_init.unsqueeze(2).expand(B, M, K_actual, 3, H, W)
            # Mu-tonemap the HDR init (stabilize with tiny eps)
            H_init_log = torch.log1p(self.mu * (H_init_expanded.clamp_min(0.0) + 1e-6)) / (
                self._log1p_mu
            )  # [B,M,K,3,H,W]

            H_gray = H_init.mean(dim=2, keepdim=True)                   # [B,N,1,H,W]
            H_gray_expanded = H_gray.unsqueeze(2).expand(B, M, K_actual, 1, H, W)

            # concat per-view feats (13 ch)
            feats = [
                topk_ldr,         # 3
                topk_lin,         # 3
                topk_conf,        # 1
                topk_depth,       # 1
                topk_snr,         # 1
                H_init_log,       # 3
                H_gray_expanded,  # 1
            ]
            x = torch.cat(feats, dim=3)                                 # [B,M,K,13,H,W]
            x = x.reshape(B * M, K_actual * 13, H, W)                      # [B*M, K*13, H, W]

            # ===== 2) Encoder-Decoder with SE attention (bfloat16) =====
            f1 = self.e1(x)
            f1 = self.se1(f1)

            f2 = self.e2(f1)
            f2 = self.se2(f2)

            f3 = self.e3(f2)
            f3 = self.se3(f3)

            b = self.bot(f3)

            u2 = self.d2(b)
            u2 = self.mix2(torch.cat([u2, f2], dim=1))

            u1 = self.d1(u2)
            u1 = self.mix1(torch.cat([u1, f1], dim=1))

            res_raw = self.out(u1)  # [B*M,3,H,W]

            # 안정적인 unit scale (분모 stop-grad)
            den = res_raw.detach().abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)
            res_unit = res_raw / den

            # 게이트/캡: 모두 (0,1) 범위의 소프트 제한
            gate = torch.sigmoid(self.res_gate)          # (1,3,1,1)
            cap = torch.sigmoid(self.res_cap_log)        # scalar in (0,1)

            # 소프트 포화로 안전한 보정량
            res_log = gate * cap * torch.tanh(res_unit)

            # ===== 3) Mu-tonemap domain operations in FP32 (autocast off) =====
        with torch.cuda.amp.autocast(enabled=False):
            # Forward tonemap (with small offset for numerical safety)
            H0 = torch.log1p(self.mu * (H_init.reshape(B * M, 3, H, W).clamp_min(0.0) + self.log_offset)) / (
                self._log1p_mu
            )
            # Add residual in tonemap domain
            H_ref_log = H0 + res_log.to(dtype=H0.dtype)
            # Inverse mu-tonemap: inv = (exp(y*log1p_mu)-1)/mu
            H_ref = (torch.exp(H_ref_log * self._log1p_mu) - 1.0) / self.mu - self.log_offset
            H_ref = H_ref.reshape(B, M, 3, H, W)

        # ===== 4) 디버그 출력 (옵션) =====
        if self.debug:
            used = float((res_log.abs().mean() > 0).item())
            print(f"[rank {res_log.device}] used_refiner={int(used)}, "
                  f"res|mean|={res_log.abs().mean().item():.6f}")
            #print(f"Gate: {gate.data.float().mean().item():.6f}, Cap: {cap.data.float().item():.3f}")

        return H_ref
