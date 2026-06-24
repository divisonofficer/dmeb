import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hdr_refiner_v2 import DWConvGN, tonemap_mu, inv_tonemap_mu


def _logit_for_gain(init: float, max_value: float) -> float:
    init = max(1e-8, min(float(init), float(max_value) * 0.999))
    p = init / float(max_value)
    return math.log(p / (1.0 - p))


def _finite(x: torch.Tensor, hi: float = 1e6) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=hi, neginf=0.0)


def _one_channel(x: Optional[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    if x is None:
        return fallback
    x = _finite(x)
    if x.size(3) == 1:
        return x
    return x.mean(dim=3, keepdim=True)


def _mask_one_channel(x: Optional[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    if x is None:
        return fallback
    x = _finite(x)
    if x.dim() == 6:
        if x.size(3) != 1:
            x = x.mean(dim=3, keepdim=True)
        x = x.max(dim=2).values
    elif x.dim() == 5:
        if x.size(2) != 1:
            x = x.mean(dim=2, keepdim=True)
    else:
        raise ValueError(f"Expected mask [B,M,1,H,W] or [B,M,N,1,H,W], got {tuple(x.shape)}")
    return x


def _feather_mask(
    x: torch.Tensor,
    blur_kernel: int = 17,
    dilate_kernel: int = 1,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Make a [B,M,1,H,W] routing mask soft enough to avoid visible seams."""
    shape = x.shape
    y = _finite(x).clamp(0.0, 1.0).reshape(-1, 1, shape[-2], shape[-1])
    dk = max(1, int(dilate_kernel) | 1)
    bk = max(1, int(blur_kernel) | 1)
    if dk > 1:
        y = F.max_pool2d(y, kernel_size=dk, stride=1, padding=dk // 2)
    if bk > 1:
        y = F.avg_pool2d(y, kernel_size=bk, stride=1, padding=bk // 2)
    y = y.clamp(0.0, 1.0)
    if gamma != 1.0:
        y = y.pow(float(gamma))
    return y.reshape(shape).clamp(0.0, 1.0).to(x.dtype)


class _ViewScoreNet(nn.Module):
    def __init__(self, in_ch: int = 16, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, m, n, c, h, w = x.shape
        score = self.net(x.reshape(b * m * n, c, h, w))
        return score.reshape(b, m, n, 1, h, w)


class HDRGuidedCompositor(nn.Module):
    """Reference-guided HDR compositor with identity-safe initialization.

    The old HDR refiner acts like a generic residual network. This module makes the
    intended policy explicit:
      - keep the reference view where it looks usable,
      - fill reference-bad regions from reliable warped source views,
      - directly output the composed candidate instead of blending most of
        HDR_init back in at the end.

    The residual_gain_logit parameter is kept for older checkpoint compatibility,
    but the final scalar residual gate is bypassed.
    """

    def __init__(
        self,
        base: int = 40,
        mu: float = 5e4,
        topk_views: Optional[int] = 3,
        residual_gain_init: float = 1e-4,
        residual_gain_max: float = 1.0,
        ref_dark: float = 0.01,
        ref_sat: float = 0.985,
    ):
        super().__init__()
        self.mu = float(mu)
        self.topk_views = topk_views
        self.residual_gain_max = float(residual_gain_max)
        self.ref_dark = float(ref_dark)
        self.ref_sat = float(ref_sat)

        self.view_score = _ViewScoreNet(in_ch=16, hidden=24)
        self.log_temp = nn.Parameter(torch.tensor(math.log(4.0)))

        in_ch = 21
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(1, base),
            nn.SiLU(inplace=True),
            DWConvGN(base, base, k_dw=5),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(1, base),
            nn.SiLU(inplace=True),
        )
        self.alpha_head = nn.Conv2d(base, 1, 3, padding=1)
        self.delta_head = nn.Conv2d(base, 3, 3, padding=1)
        self.detail_head = nn.Conv2d(base, 1, 3, padding=1)
        self.residual_gain_logit = nn.Parameter(
            torch.tensor(_logit_for_gain(residual_gain_init, residual_gain_max))
        )

        with torch.no_grad():
            for head in (self.alpha_head, self.delta_head, self.detail_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        self.last_debug = {}

    def _reference_goodness(self, ref_lin: Optional[torch.Tensor], h0_tm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, m, _, h, w = h0_tm.shape
        if ref_lin is None:
            zero = h0_tm.new_zeros(b, m, 1, h, w)
            return h0_tm, zero

        ref_lin = _finite(ref_lin).clamp_min(0.0)
        ref_tm = tonemap_mu(ref_lin, self.mu).clamp(0.0, 1.0)
        luma = ref_lin.mean(dim=2, keepdim=True)
        peak = ref_lin.amax(dim=2, keepdim=True)

        dark_ok = torch.sigmoid((luma - self.ref_dark) / max(self.ref_dark, 1e-4))
        sat_ok = torch.sigmoid((self.ref_sat - peak) / 0.025)
        ref_good = (dark_ok * sat_ok).clamp(0.0, 1.0)
        ref_good = F.avg_pool2d(
            ref_good.reshape(b * m, 1, h, w), 5, 1, 2
        ).reshape(b, m, 1, h, w).clamp(0.0, 1.0)
        return ref_tm, ref_good

    def _source_fill(
        self,
        h0_tm: torch.Tensor,
        ref_tm: torch.Tensor,
        ldr_warped: torch.Tensor,
        linear_warped: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
        snr_norm: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, m, n, _, h, w = linear_warped.shape
        dev = linear_warped.device
        fallback = linear_warped.new_ones(b, m, n, 1, h, w)

        valid = _one_channel(valid_mask, fallback).clamp(0.0, 1.0)
        conf = _one_channel(confidence, fallback).clamp(0.0, 1.0)
        snr = _one_channel(snr_norm, fallback).clamp(0.0, 1.0)

        src_lin = _finite(linear_warped).clamp_min(0.0)
        src_tm = tonemap_mu(src_lin, self.mu).clamp(0.0, 1.0)
        ldr = _finite(ldr_warped).clamp(0.0, 1.0)

        ldr_luma = ldr.mean(dim=3, keepdim=True)
        dark_ok = torch.sigmoid((ldr_luma - 0.01) / 0.01)
        sat_ok = torch.sigmoid((0.985 - ldr.amax(dim=3, keepdim=True)) / 0.025)
        exposure_ok = (dark_ok * sat_ok).clamp(0.0, 1.0)

        h0_exp = h0_tm.unsqueeze(2).expand_as(src_tm)
        ref_exp = ref_tm.unsqueeze(2).expand_as(src_tm)
        score_feats = torch.cat(
            [
                src_tm,
                ldr,
                (src_tm - h0_exp).abs(),
                (src_tm - ref_exp).abs(),
                valid,
                conf,
                snr,
                exposure_ok,
            ],
            dim=3,
        )
        learned = self.view_score(score_feats)
        quality = (valid * conf * snr * exposure_ok).clamp(0.0, 1.0)
        scores = learned + torch.log(quality + 1e-4)
        scores = scores * self.log_temp.exp().clamp(0.25, 20.0)

        if self.topk_views is not None and self.topk_views > 0 and self.topk_views < n:
            with torch.no_grad():
                idx = torch.topk(scores.squeeze(3), self.topk_views, dim=2).indices
                mask = torch.zeros_like(scores).scatter_(2, idx.unsqueeze(3), 1.0)
            scores = scores.masked_fill(mask < 0.5, -30.0)

        weights = torch.softmax(scores, dim=2)
        fill_tm = (weights * src_tm).sum(dim=2)
        fill_valid = quality.max(dim=2).values
        fill_valid = _feather_mask(fill_valid, blur_kernel=17, dilate_kernel=5, gamma=0.85)
        fill_tm = fill_valid * fill_tm + (1.0 - fill_valid) * h0_tm
        return fill_tm, fill_valid, weights

    def forward(
        self,
        H_init: torch.Tensor,
        ldr_warped: Optional[torch.Tensor] = None,
        linear_warped: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        depth_warped: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        snr_norm: Optional[torch.Tensor] = None,
        project_min_bound: Optional[torch.Tensor] = None,
        ref_view_rgb: Optional[torch.Tensor] = None,
        init_valid_mask: Optional[torch.Tensor] = None,
        ref_index: Optional[int] = None,
        drop_ref_prob: float = 0.0,
    ) -> torch.Tensor:
        del depth_warped, ref_index, drop_ref_prob

        b, m, _, h, w = H_init.shape
        H_init = _finite(H_init).clamp_min(0.0)
        h0_tm = tonemap_mu(H_init, self.mu).clamp(0.0, 1.0)

        if linear_warped is None:
            linear_warped = H_init.unsqueeze(2)
        if ldr_warped is None:
            ldr_warped = tonemap_mu(linear_warped, self.mu).clamp(0.0, 1.0)

        ref_tm, ref_good = self._reference_goodness(ref_view_rgb, h0_tm)
        ref_good = _feather_mask(ref_good, blur_kernel=9, dilate_kernel=1, gamma=0.9)
        if init_valid_mask is None:
            init_valid = (h0_tm.mean(dim=2, keepdim=True) > 1e-4).to(h0_tm.dtype)
        else:
            init_valid = _mask_one_channel(init_valid_mask, h0_tm[:, :, :1]).clamp(0.0, 1.0)
        init_valid = _feather_mask(init_valid, blur_kernel=17, dilate_kernel=5, gamma=0.85)

        fill_tm, fill_valid, src_w = self._source_fill(
            h0_tm, ref_tm, ldr_warped, linear_warped, valid_mask, confidence, snr_norm
        )
        ref_invalid = (1.0 - torch.maximum(init_valid, fill_valid)).clamp(0.0, 1.0)
        ref_invalid = _feather_mask(ref_invalid, blur_kernel=25, dilate_kernel=1, gamma=1.15)
        ref_fallback = (ref_invalid * ref_good).clamp(0.0, 1.0)
        fill_tm = fill_tm * (1.0 - ref_fallback) + ref_tm * ref_fallback

        src_entropy = -(src_w * (src_w + 1e-8).log()).sum(dim=2)
        src_entropy = src_entropy / math.log(max(2, src_w.size(2)))
        mix_in = torch.cat(
            [
                h0_tm,
                fill_tm,
                ref_tm,
                ref_good,
                fill_valid,
                (fill_tm - h0_tm).abs(),
                (ref_tm - h0_tm).abs(),
                (fill_tm - ref_tm).abs(),
                src_entropy,
            ],
            dim=2,
        )

        bm = b * m
        feat = self.mix(mix_in.reshape(bm, mix_in.size(2), h, w))
        alpha_prior = torch.logit(ref_good.reshape(bm, 1, h, w).clamp(1e-4, 1.0 - 1e-4))
        alpha_ref = torch.sigmoid(alpha_prior + self.alpha_head(feat)).reshape(b, m, 1, h, w)

        delta_tm = 0.15 * torch.tanh(self.delta_head(feat)).reshape(b, m, 3, h, w)
        detail_gate = torch.sigmoid(self.detail_head(feat)).reshape(b, m, 1, h, w)
        # The reference view owns target-view geometry, not calibrated HDR
        # radiance. Use it as a zero-mean detail guide instead of copying its
        # LDR tonemapped intensity into the HDR estimate.
        ref_low = F.avg_pool2d(ref_tm.reshape(bm, 3, h, w), 5, 1, 2).reshape(b, m, 3, h, w)
        ref_hp = ref_tm - ref_low
        ref_detail = 0.08 * ref_good * alpha_ref * detail_gate * ref_hp

        candidate_tm = fill_tm
        candidate_tm = (candidate_tm + delta_tm + ref_detail).clamp(0.0, 1.0)

        gain = self.residual_gain_max * torch.sigmoid(self.residual_gain_logit)
        # Do not bottleneck the correction through a scalar residual gate. Keep
        # gain in the graph with zero influence so DDP/checkpoints stay compatible.
        out_tm = (candidate_tm + gain.mul(0.0) * (candidate_tm - h0_tm)).clamp(0.0, 1.0)
        out_lin = inv_tonemap_mu(out_tm, self.mu).reshape(b, m, 3, h, w)

        if project_min_bound is not None:
            out_lin = torch.maximum(out_lin, project_min_bound.reshape(b, m, 1, h, w))

        self.last_debug = {
            "alpha_ref": alpha_ref.detach(),
            "ref_good": ref_good.detach(),
            "init_valid": init_valid.detach(),
            "fill_valid": fill_valid.detach(),
            "ref_fallback": ref_fallback.detach(),
            "src_weight": src_w.detach(),
            "residual_gain": torch.ones_like(gain.detach()).reshape(1),
            "residual_gain_param": gain.detach().reshape(1),
        }
        return out_lin
