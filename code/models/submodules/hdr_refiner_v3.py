import math
from typing import Optional, Callable, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hdr_refiner_v2 import DWConvGN, LargeKernelDWBlock, TransformerBlockWin, CrossAttentionBlock, DAAdapter, tonemap_mu, inv_tonemap_mu


def local_var(x, k=5):
    """
    Compute local variance using average pooling.
    Args:
        x: Input tensor [B, C, H, W]
        k: Kernel size for averaging (default 5)
    Returns:
        Local variance map [B, C, H, W]
    """
    pad = k // 2
    m = F.avg_pool2d(x, k, 1, pad)
    m2 = F.avg_pool2d(x * x, k, 1, pad)
    return (m2 - m * m).clamp_min(0.0)


def soft_norm(x, t=0.1):
    """
    Soft normalization: x / (x + t) in [0, 1]
    Args:
        x: Input tensor
        t: Normalization threshold (default 0.1)
    Returns:
        Normalized tensor in range [0, 1]
    """
    return x / (x + t)


@torch.no_grad()
def noise_mask_from_ref_tm(ref_tm, k_edge=0.02, k_var=1.0, ksize=5):
    """
    Generate noise mask from reference tonemap to suppress noisy regions.
    Detects high-frequency content that is NOT structural edges (likely noise).
    
    Args:
        ref_tm: Reference tonemap [B*, 3, H, W] in tonemap domain
        k_edge: Edge suppression weight (default 0.02)
        k_var: Variance threshold multiplier (default 1.0)
        ksize: Kernel size for local variance (default 5)
    
    Returns:
        Noise mask N_soft [B*, 1, H, W] in range [0, 1]
        - 0: clean regions (use reference)
        - 1: noisy regions (suppress reference)
    """
    # Convert to grayscale
    gray = ref_tm.mean(1, keepdim=True)  # [B*, 1, H, W]
    
    # Edge (structure) detection using Sobel filters
    kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = gray.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge = (gx * gx + gy * gy + 1e-12).sqrt()
    
    # Local variance (high-frequency content)
    var = local_var(gray, k=ksize)
    
    # Noise score: high variance that is NOT edge structure
    # (texture-excluding high-frequency = likely noise)
    raw = (var - k_var * edge).clamp_min(0)
    
    # Soft normalization & feather for smooth transitions
    n0 = soft_norm(raw, t=raw.mean() * 0.5 + 1e-6)
    n1 = F.avg_pool2d(n0, 9, 1, 4)  # feather with 9x9 kernel
    
    return n1.clamp(0, 1)  # N_soft: [B*, 1, H, W]


class BlurPool(nn.Module):
    """
    Anti-aliased downsampling to suppress grid artifacts.
    Uses fixed Gaussian-like blur kernel [1,2,1] as buffer (no learnable params).
    """
    def __init__(self, ch, filt=(1, 2, 1)):
        super().__init__()
        f = torch.tensor(filt, dtype=torch.float32)
        k = (f[:, None] * f[None, :])
        k = (k / k.sum()).view(1, 1, 3, 3)
        self.register_buffer('kernel', k.repeat(ch, 1, 1, 1))
        self.ch = ch
    
    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.ch)


class DetailRescueHead(nn.Module):
    """
    Band-pass residual path to recover fine details lost in reference view.
    Initialized with gamma=0 for backward compatibility with pretrained weights.
    NaN-safe implementation with bounded gradients.
    """
    def __init__(self):
        super().__init__()
        self.s1 = nn.Conv2d(3, 3, 3, padding=1)
        self.s2 = nn.Conv2d(3, 3, 3, padding=1)
        # ★ Start with 0 influence for pretrained weight compatibility
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
        with torch.no_grad():
            for m in [self.s1, self.s2]:
                nn.init.zeros_(m.bias)
                m.weight.mul_(1e-3)
    
    def forward(self, x):
        """
        Args:
            x: [B*M, 3, H, W] high-pass filtered best view
        Returns:
            Scaled residual to add to output
        
        NaN-safe: Sanitizes input and intermediate values to prevent NaN propagation
        """
        # ★ Sanitize input (may contain NaN/Inf from warping/interpolation)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = x.clamp(-1e4, 1e4)  # Prevent extreme values
        
        r = F.relu(self.s1(x))
        r = self.s2(r)
        
        # ★ Sanitize output before scaling
        r = torch.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e4, 1e4)
        
        # ★ Bounded differentiable scale using tanh (range: [-0.1, 0.1])
        scale = 0.1 * torch.tanh(self.gamma)
        
        return scale * r

def _sobel_xy(x):
    kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
    ky = x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
    kx = kx.repeat(x.size(1),1,1,1); ky = ky.repeat(x.size(1),1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=x.size(1))
    gy = F.conv2d(x, ky, padding=1, groups=x.size(1))
    return gx, gy

def ref_similarity_maps(ldr_pv, ref_ldr, eps=1e-6):
    """
    ldr_pv: [B,M,N,3,H,W] (0~1)
    ref_ldr: [B,M,3,H,W]   (0~1)  ※ reference 없으면 None 가능
    반환: sim2ch [B,M,N,2,H,W]  (grad-cos, local-ncc)
    """
    B,M,N,_,H,W = ldr_pv.shape
    if ref_ldr is None:
        return ldr_pv.new_zeros(B,M,N,2,H,W)

    # Sobel gradient cosine (grayscale)
    gray_pv = (0.299*ldr_pv[:,:,:,0]+0.587*ldr_pv[:,:,:,1]+0.114*ldr_pv[:,:,:,2]).unsqueeze(3)  # [B,M,N,1,H,W]
    gray_rf = (0.299*ref_ldr[:,:,0]+0.587*ref_ldr[:,:,1]+0.114*ref_ldr[:,:,2]).unsqueeze(2)     # [B,M,1,H,W] -> [B,M,1,1,H,W]

    gray_rf = gray_rf.unsqueeze(2).expand(B,M,N,1,H,W)

    gpvx, gpvy = _sobel_xy(gray_pv.reshape(B*M*N,1,H,W))
    grfx, grfy = _sobel_xy(gray_rf.reshape(B*M*N,1,H,W))
    gpvx = gpvx.view(B,M,N,1,H,W); gpvy = gpvy.view(B,M,N,1,H,W)
    grfx = grfx.view(B,M,N,1,H,W); grfy = grfy.view(B,M,N,1,H,W)
    num = (gpvx*grfx + gpvy*grfy).mean(3, keepdim=True)   # [B,M,N,1,H,W]
    den = (torch.sqrt(gpvx**2+gpvy**2+eps)*torch.sqrt(grfx**2+grfy**2+eps)).mean(3, keepdim=True)
    cos = (num/den).clamp(-1,1)

    # local NCC (3x3)
    k = 3
    pv  = gray_pv.reshape(B*M*N,1,H,W)
    rf  = gray_rf.reshape(B*M*N,1,H,W)
    mpv = F.avg_pool2d(pv, k, 1, k//2)
    mrf = F.avg_pool2d(rf, k, 1, k//2)
    spv = F.avg_pool2d((pv-mpv)**2, k, 1, k//2).clamp_min(eps).sqrt()
    srf = F.avg_pool2d((rf-mrf)**2, k, 1, k//2).clamp_min(eps).sqrt()
    ncc = F.avg_pool2d((pv-mpv)*(rf-mrf), k, 1, k//2)/(spv*srf+eps)
    ncc = ncc.view(B,M,N,1,H,W).clamp(-1,1)

    return torch.cat([ (cos+1)/2, (ncc+1)/2 ], dim=3)   # to 0..1
class ViewwiseLocalScorer(nn.Module):
    def __init__(self, in_ch=7, hidden=24):   # ★ 5→7
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, 1, 1)
        )
    def forward(self, q_feats):  # [B,M,N,C,H,W]
        B,M,N,C,H,W = q_feats.shape
        s = self.net(q_feats.reshape(B*M*N,C,H,W))
        return s.reshape(B,M,N,1,H,W)

class PerPixelViewAggregator(nn.Module):
    def __init__(self, temperature_init=6.0, topk: Optional[int]=3):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(temperature_init))))
        self.topk = topk
        self.scorer = ViewwiseLocalScorer(in_ch=7, hidden=24)  # ★

    def forward(self, feats_all, q_conf, q_valid, q_snr, luma_log_pv, depth_pv, sim_pv2ref=None):
        B,M,N,Cf,H,W = feats_all.shape
        if sim_pv2ref is None:
            sim_pv2ref = feats_all.new_zeros(B,M,N,2,H,W)
        q_feats = torch.cat([q_conf, q_valid, q_snr, luma_log_pv, depth_pv, sim_pv2ref], 3)  # ★ +2
        scores = self.scorer(q_feats)
        t = self.log_temp.exp().clamp(0.1, 50.0)
        w = torch.softmax(scores * t, dim=2)
        if self.topk is not None and self.topk < N:
            with torch.no_grad():
                idx = torch.topk(scores.squeeze(3), self.topk, dim=2).indices
            mask = torch.zeros_like(w); mask.scatter_(2, idx.unsqueeze(3), 1.0)
            w = w * mask; w = w / w.sum(2, keepdim=True).clamp_min(1e-8)
        x_w = (w * feats_all).sum(2)
        x_m = feats_all.max(2).values
        agg = torch.cat([x_w, x_m], 2)
        return agg, w
class RefinerCoreWithDA(nn.Module):
    def __init__(self, in_ch, base=48, n_tf=2, win=8, heads=4,
                 use_large_kernel=True, use_cross_attn=True, da_ch=128,
                 use_ref_attn=True, ref_ch=128):
        super().__init__()
        self.use_cross = use_cross_attn
        self.use_ref_attn = use_ref_attn
        self.cross_scale = nn.Parameter(torch.tensor(0.1))
        self.ref_scale   = nn.Parameter(torch.tensor(0.0))  # ★ Start at 0 for compatibility
        self.da_drop = nn.Dropout2d(p=0.1)

        # ★ Add BlurPool for anti-aliased downsampling
        self.bp1 = BlurPool(base)
        self.bp2 = BlurPool(base*2)

        self.e1 = DWConvGN(in_ch, base)
        self.e2 = DWConvGN(base, base*2, stride=2)
        self.e3 = DWConvGN(base*2, base*4, stride=2)
        bot = [DWConvGN(base*4, base*4, k_dw=3)]
        if use_large_kernel: bot += [LargeKernelDWBlock(base*4, k=9)]
        self.pre_tf = nn.Sequential(*bot)
        self.tf_blocks = nn.ModuleList([TransformerBlockWin(base*4, heads, win) for _ in range(n_tf)])

        if use_cross_attn:
            self.da_proj = nn.Conv2d(da_ch, base*4, 1)
            self.cross   = CrossAttentionBlock(base*4, heads=heads, win=win)
        if use_ref_attn:
            self.ref_proj = nn.Conv2d(ref_ch, base*4, 1)
            self.ref_cross= CrossAttentionBlock(base*4, heads=heads, win=win)
            self.null_ref = nn.Parameter(torch.zeros(1, ref_ch, 1, 1))

        self.u2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(base*4, base*2, 3, padding=1), nn.PReLU(base*2))
        self.mix2 = nn.Conv2d(base*4, base*2, 3, padding=1)
        self.u1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(base*2, base, 3, padding=1), nn.PReLU(base))
        self.mix1 = nn.Conv2d(base*2, base, 3, padding=1)

    def forward(self, x, da_feat=None, ref_feat=None):
        f1 = self.e1(x)
        # ★ Apply blur before downsampling to reduce aliasing
        f1b = self.bp1(f1)
        f2 = self.e2(f1b)
        f2b = self.bp2(f2)
        f3 = self.e3(f2b)
        b  = self.pre_tf(f3)
        for blk in self.tf_blocks: b = blk(b)

        if self.use_cross and da_feat is not None:
            d = self.da_drop(da_feat) if self.training else da_feat
            if d.shape[2:] != f3.shape[2:]:
                d = F.interpolate(d, size=f3.shape[2:], mode='bilinear', align_corners=False)
            x_cross = self.cross(b, self.da_proj(d))
            s = self.cross_scale.clamp(0,1)
            b = b + s*(x_cross - b)

        if self.use_ref_attn:
            # ★ ref_feat이 없으면 null_ref를 공간에 broadcast해서 사용
            if ref_feat is None:
                r = self.null_ref.expand(b.size(0), self.null_ref.size(1), f3.size(2), f3.size(3))
            else:
                r = ref_feat + self.null_ref.expand(b.size(0), self.null_ref.size(1), f3.size(2), f3.size(3)) * 0.0
                if r.shape[2:] != f3.shape[2:]:
                    r = F.interpolate(r, size=f3.shape[2:], mode='bilinear', align_corners=False)

            x_rc = self.ref_cross(b, self.ref_proj(r))
            s2 = self.ref_scale.clamp(0, 1)
            b = b + s2 * (x_rc - b)

        u2 = self.mix2(torch.cat([self.u2(b), f2], 1))
        u1 = self.mix1(torch.cat([self.u1(u2), f1], 1))
        return u1
class HDRRefinerLite(nn.Module):
    def __init__(self,
                 da_module: nn.Module,
                 da_extractor: Optional[Callable]=None,
                 base: int = 48, n_tf: int = 2, win: int = 8, heads: int = 4,
                 use_cross_attn: bool = True,
                 use_ref_attn: bool = True,         # ★ reference cross-attn on/off
                 mu_list: Tuple[float, ...] = (1e3, 5e4, 1e6),
                 tm_squeeze_ch: int = 6,
                 mu_for_DA: Optional[int] = 1,
                 da_out_ch: int = 128, da_in_norm: bool = True, freeze_da: bool = True, da_probed_ch: int = 768,
                 topk_views: Optional[int] = 3,
                 residual_ratio_max: float = 0.4, residual_ratio_init: float = 1e-4):
        super().__init__()
        self.mu_list = tuple(float(m) for m in mu_list)
        assert len(self.mu_list) >= 2
        self.mu_for_DA = mu_for_DA if mu_for_DA is not None else (len(self.mu_list)//2)
        self.residual_ratio_max = float(residual_ratio_max)
        self.register_buffer("_residual_ratio", torch.tensor(float(residual_ratio_init)))

        self.agg = PerPixelViewAggregator(temperature_init=6.0, topk=topk_views)
        self.da_adapter  = DAAdapter(da_module, extractor=da_extractor,
                                     in_norm=da_in_norm, out_ch=da_out_ch,
                                     freeze=freeze_da, probed_ch=da_probed_ch)
        # reference도 같은 encoder로 처리 (채널 동일)
        self.ref_adapter = self.da_adapter

        # μ-squeeze: agg 뒤 μ채널이 2배이므로 2*3*K → squeeze
        self.tm_squeeze = nn.Conv2d(2 * 3 * len(self.mu_list), tm_squeeze_ch, 1)

        prefix, suffix = 9, 1
        in_channels = 2*(prefix + suffix) + tm_squeeze_ch  # 20 + tm_squeeze_ch

        self.refine_core = RefinerCoreWithDA(in_channels, base, n_tf, win, heads,
                                             use_large_kernel=True, use_cross_attn=use_cross_attn,
                                             da_ch=da_out_ch,
                                             use_ref_attn=use_ref_attn, ref_ch=da_out_ch)

        self.head_absT = nn.Conv2d(base, 3, 3, padding=1)
        self.head_dz_list = nn.ModuleList([nn.Conv2d(base, 3, 3, padding=1)
                                           for _ in self.mu_list])
        self.head_alpha = nn.Conv2d(base, len(self.mu_list), 1)
        self.res_norm = nn.GroupNorm(1, 3, affine=True)
        
        # ★ Add DetailRescueHead for recovering fine details from other views
        self.detail_head = DetailRescueHead()

        with torch.no_grad():
            nn.init.zeros_(self.head_absT.bias); self.head_absT.weight.mul_(1e-3)
            for h in self.head_dz_list:
                nn.init.zeros_(h.bias); h.weight.mul_(1e-3)
            nn.init.zeros_(self.head_alpha.bias); self.head_alpha.weight.mul_(1e-3)
            

    def set_residual_ratio(self, r: float):
        r = max(0.0, min(float(r), self.residual_ratio_max))
        self._residual_ratio.fill_(r)

    def _build_feats_multi_mu(self, H_init, ldr_warped, linear_warped,
                              valid_mask=None, depth_warped=None, confidence=None, snr_norm=None,
                              ref_view_rgb: Optional[torch.Tensor]=None):
        B,M,N,_,H,W = ldr_warped.shape
        dev = H_init.device
        if valid_mask is None: valid_mask = torch.ones(B,M,N,1,H,W, device=dev)
        if confidence is None: confidence = valid_mask
        if depth_warped is None: depth_warped = torch.zeros(B,M,N,1,H,W, device=dev)
        if snr_norm is None: snr_norm = torch.ones(B,M,N,1,H,W, device=dev)*0.5

        # 멀티-μ
        H_init_exp = H_init.unsqueeze(2).expand(B,M,N,3,H,W)
        H_T_stack = [tonemap_mu(H_init_exp, mu) for mu in self.mu_list]
        H_T_cat   = torch.cat(H_T_stack, dim=3)
        H_T_mid   = tonemap_mu(H_init_exp, self.mu_list[self.mu_for_DA])
        H_gray_pv = H_T_mid.mean(3, keepdim=True)

        # scorer용 reference similarity (옵션)
        #sim2 = ref_similarity_maps(ldr_warped, ref_view_rgb)  # [B,M,N,2,H,W] or zeros
        
        mu_mid = self.mu_list[self.mu_for_DA]
        lin_T_mid = tonemap_mu(linear_warped, mu_mid)              # [B,M,N,3,H,W]
        ref_T_mid = None
        if ref_view_rgb is not None:
            # ref_view_rgb가 linear가 아닐 수 있으므로, 가능하면 사전 linear 정규화 후 사용
            ref_T_mid = tonemap_mu(ref_view_rgb, mu_mid)        # [B,M,3,H,W]
        sim2 = ref_similarity_maps(lin_T_mid, ref_T_mid)    
        

        feats = torch.cat([ldr_warped, linear_warped, confidence, depth_warped, snr_norm,
                           H_T_cat, H_gray_pv], 3)

        q_conf = confidence; q_valid = valid_mask.float(); q_snr = snr_norm
        luma_log_pv = H_T_mid.mean(3, keepdim=True); depth_pv = depth_warped

        agg, weights = self.agg(feats, q_conf, q_valid, q_snr, luma_log_pv, depth_pv, sim_pv2ref=sim2)

        Cv = feats.size(3); C_prefix = 3+3+1+1+1; C_mu = 3*len(self.mu_list)
        idx_mu_start = 2*C_prefix; idx_mu_end = 2*(C_prefix + C_mu)

        BM = B*M
        x_full = agg.reshape(BM, 2*Cv, H, W)
        mu_chunk = x_full[:, idx_mu_start:idx_mu_end, :, :]
        mu_squeezed = self.tm_squeeze(mu_chunk)
        x_left  = x_full[:, :idx_mu_start, :, :]
        x_right = x_full[:, idx_mu_end:, :, :]
        x_refiner = torch.cat([x_left, mu_squeezed, x_right], 1)

        # DA/Ref 입력용 tonemap (N축 없이)
        H_T_mid_target = tonemap_mu(H_init, self.mu_list[self.mu_for_DA]).reshape(BM,3,H,W)

        return x_refiner, weights, H_T_mid_target

    def forward(self, H_init, ldr_warped, linear_warped,
                valid_mask=None, depth_warped=None, confidence=None, snr_norm=None,
                project_min_bound: Optional[torch.Tensor]=None,
                ref_view_rgb: Optional[torch.Tensor]=None,   # ★ [B,M,3,H,W] 또는 None
                ref_index: Optional[int]=None,               # ★ H_init에서 뽑아 쓰고 싶으면 index 제공
                drop_ref_prob: float=0.3):                   # ★ Drop-Ref probability during training
        """
        - ref_view_rgb가 주어지면 이를 기준 좌표로 align을 유도.
        - ref_index가 주어지면 H_init[:,ref_index]을 reference로 사용(편의).
        - drop_ref_prob: Probability to drop reference features during training (default 0.3)
        둘 다 None이면 기존 동작.
        """
        

        
        
        B,M,_,H,W = H_init.shape
        if ref_view_rgb is None and ref_index is not None:
            # H_init의 ref_index를 tonemap 기준으로 사용하기 위한 RGB proxy가 필요하다면
            # 여기서는 간단히 H_init을 γ-압축해서 LDR proxy로 씀 (선택)
            ref_lin = H_init[:, ref_index]  # [B,3,H,W]
            ref_view_rgb = (ref_lin / (ref_lin.max(dim=(2,3), keepdim=True).values+1e-8)).unsqueeze(1).expand(B, M, 3, H, W)

        x_refiner, weights, H_T_mid_for_DA = self._build_feats_multi_mu(
            H_init, ldr_warped, linear_warped, valid_mask, depth_warped, confidence, snr_norm,
            ref_view_rgb=ref_view_rgb
        )

        # DA/Ref features
        da_feat  = self.da_adapter(H_T_mid_for_DA)                   # [B*M, C, H/4, W/4]
        ref_feat = None
        if ref_view_rgb is not None:
            # reference도 같은 μ(중간)로 tonemap해서 인코딩
            ref_T = tonemap_mu(ref_view_rgb.reshape(B*M,3,H,W), 5e4)
            ref_feat = self.ref_adapter(ref_T)
            
            # ★ Noise-based gating: suppress reference in noisy regions
            # Use middle μ tonemap for noise detection
            mu_mid = self.mu_list[self.mu_for_DA]
            ref_tm = tonemap_mu(ref_view_rgb.reshape(B*M, 3, H, W), mu_mid)  # [B*M, 3, H, W]
            N_soft = noise_mask_from_ref_tm(ref_tm)  # [B*M, 1, H, W]
            
            # Resize noise mask to match ref_feat spatial dimensions
            if N_soft.shape[-2:] != ref_feat.shape[-2:]:
                N_soft = F.interpolate(N_soft, size=ref_feat.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            # Gate reference features: (1 - N_soft) suppresses noisy regions
            # N_soft=0 (clean) → keep ref_feat, N_soft=1 (noisy) → zero out ref_feat
            ref_feat = ref_feat * (1.0 - N_soft)
            
            # ★ Drop-Ref: During training, randomly drop reference features
            # to prevent over-reliance on reference view
            if self.training and drop_ref_prob > 0:
                if torch.rand(1).item() < drop_ref_prob:
                    ref_feat = None
                else:
                    # ★ Enhanced stop-grad: handle both saturated AND dark regions
                    sat_mask = (ref_view_rgb.max(dim=2, keepdim=True)[0] > 0.99).float()
                    dark_mask = (ref_view_rgb.mean(dim=2, keepdim=True) < 0.01).float()
                    bad_mask = torch.maximum(sat_mask, dark_mask)
                    bad_mask = bad_mask.reshape(B*M, 1, H, W)
                    
                    # Resize bad_mask to match ref_feat spatial dimensions
                    if bad_mask.shape[-2:] != ref_feat.shape[-2:]:
                        bad_mask = F.interpolate(bad_mask, size=ref_feat.shape[-2:], 
                                                mode='nearest')
                    
                    # Apply stop-grad in bad regions: detach bad parts
                    ref_feat = ref_feat * (1.0 - bad_mask) + ref_feat.detach() * bad_mask

        feat = self.refine_core(x_refiner, da_feat=da_feat, ref_feat=ref_feat)
        BM = B*M
        eps = 1e-8
        H0_T = H_init.clamp(eps, 1-eps).reshape(BM, 3, H, W)              # [B*M,3,H,W]
        z0   = torch.log(H0_T) - torch.log1p(-H0_T)     # logit(H0_T)

        H_T_mu_refined = []
        dz_raw_list, dz_norm_list = [], []

        z_cap = 2.0  # residual cap in logit space
        for i, mu in enumerate(self.mu_list):
            dz_raw = self.head_dz_list[i](feat)         # [B*M,3,H,W]
            dz_norm = self.res_norm(dz_raw)             # GN only, no scaling by beta
            dz = dz_norm * z_cap            # cap to [-z_cap, z_cap]
            
            H_T_i = torch.sigmoid(z0 + dz)
            H_T_mu_refined.append(H_T_i)
            dz_raw_list.append(dz_raw); dz_norm_list.append(dz_norm)

        alpha_logits = self.head_alpha(feat)            # [B*M,K,H,W]
        alpha = torch.softmax(alpha_logits, dim=1)

        # mixture over μ (broadcast α to 3 channels)
        H_T_blend = 0.0
        for i, H_T_i in enumerate(H_T_mu_refined):
            H_T_blend = H_T_blend + alpha[:, i:i+1] * H_T_i

        # Absolute path
        H_T_abs = torch.sigmoid(self.head_absT(feat))

        # ---- Blend ONLY here with beta (residual_ratio) ----
        beta = float(self._residual_ratio.item())       # 0..residual_ratio_max
        H_ref_log = (1.0 - beta) * H_T_abs + beta * H_T_blend
        
        
        H_ref_lin = inv_tonemap_mu(H_ref_log, self.mu_list[self.mu_for_DA])

        # ★ Detail rescue: recover fine texture from best sharp view
        if linear_warped is not None and self.training:
            # Helper function: high-pass filter (simple unsharp) with NaN safety
            def highpass(im):
                """
                High-pass filter with NaN sanitization.
                Warped/interpolated images may contain NaN/Inf values.
                """
                lo = F.avg_pool2d(im, 3, 1, 1)
                hp = im - lo
                # ★ Sanitize against NaN/Inf from warping artifacts
                hp = torch.nan_to_num(hp, nan=0.0, posinf=1e6, neginf=-1e6)
                hp = hp.clamp(-1e4, 1e4)
                return hp
            
            # Helper function: Sobel magnitude for sharpness metric
            def sobel_mag(x):
                kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
                ky = x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
                gx = F.conv2d(x, kx, padding=1)
                gy = F.conv2d(x, ky, padding=1)
                return (gx*gx + gy*gy + 1e-12).sqrt()
            
            # Find best sharp view per pixel
            with torch.no_grad():
                sob_list = []
                for i in range(linear_warped.size(2)):  # N views
                    view_lin = linear_warped[:, :, i]  # [B,M,3,H,W]
                    # ★ Sanitize before tonemap to prevent NaN propagation
                    view_lin = torch.nan_to_num(view_lin, nan=0.0, posinf=1e6, neginf=-1e6)
                    view_lin = view_lin.clamp(0.0, 1e6)
                    
                    view_tm = tonemap_mu(view_lin.reshape(B*M,3,H,W), 5e4)
                    sxy = sobel_mag(view_tm.mean(1, True))  # [B*M,1,H,W]
                    sob_list.append(sxy)
                S = torch.stack(sob_list, dim=1)  # [B*M,N,1,H,W]
                idx = S.argmax(dim=1, keepdim=True).expand(-1,-1,3,-1,-1)  # [B*M,1,3,H,W]
                
                # Gather best view
                lin_views = linear_warped.reshape(B*M, 1, linear_warped.size(2), 3, H, W)
                lin_views = lin_views.permute(0, 2, 3, 4, 5, 1).squeeze(-1)  # [B*M,N,3,H,W]
                best_lin = lin_views.gather(1, idx).squeeze(1)  # [B*M,3,H,W]
                
                # ★ Final sanitization before high-pass
                best_lin = torch.nan_to_num(best_lin, nan=0.0, posinf=1e6, neginf=-1e6)
                best_lin = best_lin.clamp(0.0, 1e6)
            
            # Extract high-frequency component (now NaN-safe)
            residual = highpass(best_lin)
            residual = self.detail_head(residual)  # [B*M,3,H,W], gamma=0 initially
            
            # ★ Final sanitization before adding to output
            residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
            residual = residual.clamp(-1e4, 1e4)
            
            # Add detail residual to output
            H_ref_lin = H_ref_lin + residual

        if project_min_bound is not None:
            H_ref_lin = torch.maximum(H_ref_lin, project_min_bound.reshape(BM,1,H,W))

        H_ref = H_ref_lin.reshape(B, M, 3, H, W)

 
        return H_ref
