# hdr_refiner_abs_residual_cap.py
import math
from typing import Optional, Callable, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utils (Tonemap / Metrics)
# =========================
def charbonnier(x, eps=1e-6):
    return torch.sqrt(x * x + eps)

def tonemap_mu(x, mu: float = 1e4, eps: float = 1e-6):
    return torch.log1p(mu * (x.clamp_min(0.0) + eps)) / math.log1p(mu)

def inv_tonemap_mu(y, mu: float = 1e4):
    return (torch.exp(y * math.log1p(mu)) - 1.0) / mu

def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim.clamp(0, 1).mean()


# =========================
# Light blocks
# =========================
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps
    def forward(self, x):
        m = x.mean(1, keepdim=True); v = x.var(1, keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + self.eps) * self.w + self.b

class DWConvGN(nn.Module):
    def __init__(self, ci, co, stride=1, k_dw=5, gn_groups=1):
        super().__init__()
        self.dw = nn.Conv2d(ci, ci, k_dw, stride=stride, padding=k_dw//2, groups=ci, bias=False)
        self.gn = nn.GroupNorm(gn_groups, ci, affine=False)
        self.pw = nn.Conv2d(ci, co, 1, bias=False)
        self.act = nn.PReLU(co)
    def forward(self, x): return self.act(self.pw(self.gn(self.dw(x))))

class LargeKernelDWBlock(nn.Module):
    def __init__(self, c, k=9):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k//2, groups=c)
        self.pw1 = nn.Conv2d(c, c, 1)
        self.pw2 = nn.Conv2d(c, c, 1)
        self.act = nn.GELU()
    def forward(self, x):
        r = x
        x = self.dw(x); x = self.pw1(x); x = self.act(x); x = self.pw2(x)
        return x + r

def window_partition(x, win):
    B,C,H,W = x.shape; assert H%win==0 and W%win==0
    x = x.reshape(B,C,H//win,win,W//win,win).permute(0,2,4,1,3,5).reshape(B*(H//win)*(W//win),C,win,win)
    return x
def window_reverse(w, win, H, W):
    Bnw,C,_,_ = w.shape; B = Bnw // ((H//win)*(W//win))
    x = w.reshape(B,H//win,W//win,C,win,win).permute(0,3,1,4,2,5).reshape(B,C,H,W)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, heads=4, win=8, qkv_bias=True):
        super().__init__()
        self.h, self.win = heads, win
        self.scale = (dim//heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        B,C,H,W = x.shape
        xw = window_partition(x, self.win)
        qkv = self.qkv(xw); q,k,v = torch.chunk(qkv, 3, 1)
        def split(t):
            Bnw,Cc,h,w = t.shape; d = Cc//self.h
            return t.reshape(Bnw,self.h,d,h*w)
        q,k,v = map(split, (q,k,v))
        attn = torch.einsum('bhdk,bhdn->bhkn', q*self.scale, k).softmax(-1)
        y = torch.einsum('bhkn,bhdn->bhdk', attn, v).reshape(xw.size(0), -1, self.win, self.win)
        return window_reverse(self.proj(y), self.win, H, W)

class TransformerBlockWin(nn.Module):
    def __init__(self, dim, heads=4, win=8, mlp_ratio=2.0):
        super().__init__()
        self.n1, self.attn = LayerNorm2d(dim), WindowAttention(dim, heads, win)
        self.n2 = LayerNorm2d(dim)
        hidden = int(dim*mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, dim, 1)
        )
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x

class CrossAttentionBlock(nn.Module):
    """Query: refiner feat, Key/Value: DA adapted feat (detach)."""
    def __init__(self, dim_q, heads=4, win=8):
        super().__init__()
        self.nq = LayerNorm2d(dim_q); self.nkv = LayerNorm2d(dim_q)
        self.q = nn.Conv2d(dim_q, dim_q, 1)
        self.k = nn.Conv2d(dim_q, dim_q, 1)
        self.v = nn.Conv2d(dim_q, dim_q, 1)
        self.proj = nn.Conv2d(dim_q, dim_q, 1)
        self.h, self.win = heads, win
        self.scale = (dim_q//heads) ** -0.5
    def forward(self, xq, xkv):
        xq = self.nq(xq); xkv = self.nkv(xkv)
        B,C,H,W = xq.shape
        qw = window_partition(self.q(xq), self.win)
        kw = window_partition(self.k(xkv), self.win)
        vw = window_partition(self.v(xkv), self.win)
        def split(t):
            Bnw,Cc,h,w = t.shape; d=Cc//self.h
            return t.reshape(Bnw,self.h,d,h*w)
        q,k,v = map(split,(qw,kw,vw))
        attn = torch.einsum('bhdk,bhdn->bhkn', q*self.scale, k).softmax(-1)
        y = torch.einsum('bhkn,bhdn->bhdk', attn, v).reshape(qw.size(0), -1, self.win, self.win)
        return xq + window_reverse(self.proj(y), self.win, H, W)


# =========================
# Per-View Aware Aggregation
# =========================
class ViewwiseLocalScorer(nn.Module):
    def __init__(self, in_ch=5, hidden=24):
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
        self.scorer = ViewwiseLocalScorer(in_ch=5, hidden=24)
    def forward(self, feats_all, q_conf, q_valid, q_snr, luma_log_pv, depth_pv):
        B,M,N,Cf,H,W = feats_all.shape
        q_feats = torch.cat([q_conf, q_valid, q_snr, luma_log_pv, depth_pv], 3)
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
        agg = torch.cat([x_w, x_m], 2)   # [B,M,2*Cf,H,W]
        return agg, w


# =========================
# DepthAnything Adapter (no '.backbone' name)
# =========================
class DAAdapter(nn.Module):
    """
    - 생성자에서 받은 DepthAnything v2 encoder를 내부에서 사용
    - extractor: Callable(da_module, x_T)->feat@1/4 (없으면 da_module(x_T) 가정)
    - reduce: 1x1 conv로 채널 정합
    """
    def __init__(self, da_module: nn.Module, extractor: Optional[Callable]=None,
                 in_norm=True, out_ch=128, freeze=True, probed_ch=256):
        super().__init__()
        self.backbone = da_module
        self.extractor = extractor
        self.in_norm = in_norm
        self.reduce = nn.Conv2d(probed_ch, out_ch, 1)
        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        ch = out_ch
        self.depatchify = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
                nn.Conv2d(ch, ch, 1),
                nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
                nn.Conv2d(ch, ch, 1),
            )


    def _best_grid_from_tokens(self, n_tokens: int, aspect_hw: float):
        """
        n_tokens = Ph * Pw 를 만족하는 (Ph,Pw) 중에서
        Ph/Pw 가 aspect_hw (= H/W )에 가장 가까운 쌍을 고른다.
        """
        # 빠른 후보: sqrt 근처에서 시작
        root = int(math.sqrt(n_tokens))
        best = (1, n_tokens)
        best_err = float("inf")

        # 약수 탐색 (root 주변에서 양방향)
        for p in range(1, root + 1):
            if n_tokens % p == 0:
                q = n_tokens // p
                # 두 후보 (p,q)와 (q,p) 중 aspect에 더 가까운 것을 선택
                for (ph, pw) in [(p, q), (q, p)]:
                    err = abs((ph / pw) - aspect_hw)
                    if err < best_err:
                        best_err = err
                        best = (ph, pw)
        return best  # (Ph, Pw)

    @torch.no_grad()
    def _extract(self, x_T: torch.Tensor) -> torch.Tensor:

     
        B, _, H, W = x_T.shape
        feats = self.backbone.pretrained.get_intermediate_layers(
            x_T, self.backbone.intermediate_layer_idx[self.backbone.encoder], return_class_token=True
        )
        # 가장 심층 stage 사용(원하면 여러 stage 평균도 가능)
        tokens, cls_tok = feats[-1]   # tokens:[B,N,C], cls_tok:[B,C]
        # 간혹 구현체마다 tokens에 CLS가 포함되는 변종이 있어 방어적으로 제거
        # (만약 포함 안 되어 있으면 아래 조건이 False라 원본 유지)
        if tokens.shape[1] == ((H // 14) * (W // 14) + 1):
            tokens = tokens[:, 1:, :]  # 맨 앞 CLS 제거

        B_, N, C = tokens.shape
        assert B_ == B, f"Batch mismatch: tokens.B={B_}, x_T.B={B}"

        # 입력 종횡비 기반으로 N의 인수쌍 중 최적 (Ph,Pw) 선택
        aspect = (H / W) if W > 0 else 1.0
        Ph, Pw = self._best_grid_from_tokens(N, aspect_hw=aspect)
        assert Ph * Pw == N, f"Internal error: chosen grid {(Ph,Pw)} != N={N}"

        f_map = tokens.transpose(1, 2).reshape(B, C, Ph, Pw)  # [B,C,Ph,Pw]
        return f_map

    def forward(self, x_T: torch.Tensor) -> torch.Tensor:
        # x_T: tonemapped [B*,3,H,W]
        if self.in_norm:
            x_T = (x_T - 0.5) / 0.5

        B, C, H, W = x_T.shape

        # (선택) 14 배수 패딩 유지 가능; extractor는 기존 그대로
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h > 0 or pad_w > 0:
            x_T = F.pad(x_T, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            f = self._extract(x_T)                # [B, C_da(=768), Ph, Pw]

        # 1) 채널 축소를 먼저 해서 연산량↓
        f = self.reduce(f)                         # [B, out_ch, Ph, Pw]  (ex. out_ch=128)

        # 2) 목표 해상도 = H/4, W/4 로 고품질 업샘플 (bicubic + antialias)
        tgt = (H // 4, W // 4)
        f = F.interpolate(f, size=tgt, mode='bicubic', align_corners=False, antialias=True)

        # 3) De-patchify: 패치 경계 alias 제거 (depthwise + pointwise residual blocks)
 
            
        f = self.depatchify(f)                     # [B, out_ch, H/4, W/4]

        # Remove padding from output if it was added (after upsample to exact H/4,W/4)
        if pad_h > 0 or pad_w > 0:
            out_h = H // 4
            out_w = W // 4
            f = f[:, :, :out_h, :out_w]

        return f


# =========================
# Refiner core (no '.backbone' naming)
# =========================
class RefinerCoreWithDA(nn.Module):
    """UNet-lite + Windowed Transformer + (옵션) Cross-Attn with DA feats."""
    def __init__(self, in_ch, base=48, n_tf=2, win=8, heads=4,
                 use_large_kernel=True, use_cross_attn=True, da_ch=128):
        super().__init__()
        self.use_cross = use_cross_attn
        # Cross-Attn dependence scaling and stochastic DA dropout
        # cross_scale: learnable scalar starting small (e.g., 0.1)
        self.cross_scale = nn.Parameter(torch.tensor(0.1))
        # da_drop_p: during training, drop DA with this probability to avoid over-reliance
        self.da_drop_p = 0.3
        self.e1 = DWConvGN(in_ch, base)
        self.e2 = DWConvGN(base, base*2, stride=2)
        self.e3 = DWConvGN(base*2, base*4, stride=2)
        bot = [DWConvGN(base*4, base*4, k_dw=3)]
        if use_large_kernel: bot += [LargeKernelDWBlock(base*4, k=9)]
        self.pre_tf = nn.Sequential(*bot)
        self.tf_blocks = nn.ModuleList([TransformerBlockWin(base*4, heads, win) for _ in range(n_tf)])
        if use_cross_attn:
            self.da_proj = nn.Conv2d(da_ch, base*4, 1)
            self.cross = CrossAttentionBlock(base*4, heads=heads, win=win)
        self.u2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(base*4, base*2, 3, padding=1), nn.PReLU(base*2))
        self.mix2 = nn.Conv2d(base*4, base*2, 3, padding=1)
        self.u1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(base*2, base, 3, padding=1), nn.PReLU(base))
        self.mix1 = nn.Conv2d(base*2, base, 3, padding=1)
        self.da_drop = nn.Dropout2d(p=0.1)

    def forward(self, x, da_feat=None):
        f1 = self.e1(x); f2 = self.e2(f1); f3 = self.e3(f2)     # 1/4
        b  = self.pre_tf(f3)
        for blk in self.tf_blocks: b = blk(b)
        if self.use_cross and da_feat is not None:
            # 학습시 랜덤 드롭 (과의존 억제)
            da_feat = self.da_drop(da_feat) if self.training else da_feat

            # Align DA feature to f3 spatial size (1/4)
            if da_feat.shape[2:] != f3.shape[2:]:
                da_feat = F.interpolate(da_feat, size=f3.shape[2:], mode='bilinear', align_corners=False)

            # project (allow updating proj) but detach backbone features
            kv = self.da_proj(da_feat)
            x_cross = self.cross(b, kv)
            # 보간: b ← b + s * (x_cross - b)  (s is clamped 0..1)
            s = self.cross_scale.clamp(0.0, 1.0)
            b = b + s * (x_cross - b)

        u2 = self.mix2(torch.cat([self.u2(b), f2], 1))
  
        u1 = self.mix1(torch.cat([self.u1(u2), f1], 1))
        return u1   # [B*, base, H, W]



class HDRRefinerLite(nn.Module):
    """
    Multi-μ HDR Refiner
    - 입력: per-view 멀티-μ tonemap 스택을 1x1로 squeeze하여 refiner에 투입
    - 출력: μ별 logit-residual 보정 후 softmax로 가중합 → absolute와 β로 블렌딩
    - residual cap(β)은 set_residual_ratio()로 스케줄링
    - DA cross-attn/aggregator는 기존과 동일하게 사용
    """
    def __init__(self,
                 da_module: nn.Module,
                 da_extractor: Optional[Callable]=None,
                 # ==== 모델 폭/구성 ====
                 base: int = 48,
                 n_tf: int = 2, win: int = 8, heads: int = 4,
                 use_cross_attn: bool = True,
                 # ==== 멀티-μ 설정 ====
                 mu_list: Tuple[float, ...] = (1e3, 5e3, 5e4),
                 tm_squeeze_ch: int = 6,            # 멀티-μ 스택(3*|μ|) → squeeze 채널 수
                 mu_for_DA: Optional[int] = 1,      # DAAdapter 입력에 쓸 μ index (기본: 중간 μ)
                 # ==== DA 어댑터 ====
                 da_out_ch: int = 128,
                 da_in_norm: bool = True,
                 freeze_da: bool = True,
                 da_probed_ch: int = 768,
                 # ==== Aggregation ====
                 topk_views: Optional[int] = 3,
                 # ==== residual cap ====
                 residual_ratio_max: float = 0.4,
                 residual_ratio_init: float = 0.0001):
        super().__init__()
        self.mu_list = tuple(float(m) for m in mu_list)
        assert len(self.mu_list) >= 2, "μ list must have length ≥ 2"
        self.mu_for_DA = mu_for_DA if mu_for_DA is not None else (len(self.mu_list)//2)
        self.residual_ratio_max = float(residual_ratio_max)
        self.register_buffer("_residual_ratio", torch.tensor(float(residual_ratio_init)))
        # ==== Aggregator ====
        self.agg = PerPixelViewAggregator(temperature_init=6.0, topk=topk_views)

        # ==== DA Adapter ====
        self.da_adapter = DAAdapter(da_module, extractor=da_extractor,
                                    in_norm=da_in_norm, out_ch=da_out_ch,
                                    freeze=freeze_da, probed_ch=da_probed_ch)

        # ==== 멀티-μ 입력 squeeze: 3*|μ| → tm_squeeze_ch ====
        self.tm_squeeze = nn.Conv2d(6 * len(self.mu_list), tm_squeeze_ch, 1)

        # per-view 입력 채널 자동 계산:
        #   LDR(3)+LINEAR(3)+CONF(1)+DEPTH(1)+SNR(1)+TM_SQUEEZE(tm_squeeze_ch)+GRAY(1)
        self.per_view_ch = 3 + 3 + 1 + 1 + 1 + 1
        in_channels = 2 * self.per_view_ch  + tm_squeeze_ch

        # ==== Refiner core ====
        self.refine_core = RefinerCoreWithDA(in_channels, base, n_tf, win, heads,
                                             use_large_kernel=True,
                                             use_cross_attn=use_cross_attn,
                                             da_ch=da_out_ch)

        # ==== Heads ====
        # Absolute tonemap head (공통)
        self.head_absT = nn.Conv2d(base, 3, 3, padding=1)
        # μ별 residual Δz head
        self.head_dz_list = nn.ModuleList([nn.Conv2d(base, 3, 3, padding=1)
                                           for _ in self.mu_list])
        # μ mixture weight (per-pixel, per-spatial, channel 공유)
        self.head_alpha = nn.Conv2d(base, len(self.mu_list), 1)  # → softmax over μ
        self.res_norm = nn.GroupNorm(1, 3, affine=True)

        with torch.no_grad():
            nn.init.zeros_(self.head_absT.bias); self.head_absT.weight.mul_(1e-3)
            for h in self.head_dz_list:
                nn.init.zeros_(h.bias); h.weight.mul_(1e-3)
            nn.init.zeros_(self.head_alpha.bias); self.head_alpha.weight.mul_(1e-3)

    # 외부에서 residual 비율을 설정 (0..residual_ratio_max)
    def set_residual_ratio(self, r: float):
        r = max(0.0, min(float(r), self.residual_ratio_max))
        self._residual_ratio.fill_(r)

    def _build_feats_multi_mu(self, H_init, ldr_warped, linear_warped,
                              valid_mask=None, depth_warped=None, confidence=None, snr_norm=None):
        """
        멀티-μ tonemap 스택과 기존 per-view 컨텍스트를 구성.
        반환: (agg_input[B*M,C,H,W], weights, also per-view aux)
        """
        B,M,N,_,H,W = ldr_warped.shape
        dev = H_init.device
        if valid_mask is None: valid_mask = torch.ones(B,M,N,1,H,W, device=dev)
        if confidence is None: confidence = valid_mask
        if depth_warped is None: depth_warped = torch.zeros(B,M,N,1,H,W, device=dev)
        if snr_norm is None: snr_norm = torch.ones(B,M,N,1,H,W, device=dev)*0.5

        # 멀티-μ tonemap 스택
        H_init_exp = H_init.unsqueeze(2).expand(B,M,N,3,H,W)  # [B,M,N,3,H,W]
        H_T_stack = [tonemap_mu(H_init_exp, mu) for mu in self.mu_list]   # list len=K, each [B,M,N,3,H,W]
        H_T_cat = torch.cat(H_T_stack, dim=3)                               # [B,M,N,3*K,H,W]
        # per-view gray (중간 μ 기준)
        H_T_mid = tonemap_mu(H_init_exp, self.mu_list[self.mu_for_DA])
        H_gray_pv = H_T_mid.mean(3, keepdim=True)                           # [B,M,N,1,H,W]

        # per-view concat
        feats = torch.cat([ldr_warped, linear_warped, confidence, depth_warped, snr_norm,
                           H_T_cat, H_gray_pv], 3)                           # [B,M,N,Cv,H,W]

        # aggregation 점수 입력들
        q_conf = confidence; q_valid = valid_mask.float(); q_snr = snr_norm
        luma_log_pv = H_T_mid.mean(3, keepdim=True); depth_pv = depth_warped

        # per-view aggregation
        agg, weights = self.agg(feats, q_conf, q_valid, q_snr, luma_log_pv, depth_pv)  # [B,M,2*Cv,H,W]
        # ----- squeeze 멀티-μ 채널 -----
        # agg는 [B,M,2*Cv,H,W]인데, 이 안에 H_T_cat(3*K)이 포함되어 있음.
        # 여기서는 간단히 "per-view에서 concat된 모든 채널"을 refiner 입력으로 쓰되
        # 멀티-μ 부분은 1x1로 squeeze하기 위해 분리/결합 하는 경량한 방법을 쓴다.
        # 계층의 단순화를 위해 여기서는 전체 concat을 그대로 넘기고
        # refiner 진입 직전에 멀티-μ 영역만 찾아 squeeze 한다.

        # 분해: 원래 순서 = [ldr(3),lin(3),conf(1),depth(1),snr(1), H_T_cat(3*K), gray(1)]
        Cv = feats.size(3)
        C_prefix = 3+3+1+1+1
        C_mu = 3*len(self.mu_list)
        C_suffix = 1
        # 가중 합 이후 텐서를 다시 쪼개기 위해 view/permute 대신 index 슬라이스 사용
        agg_full = agg  # [B,M, 2*Cv, H, W]
        # 각 부분의 위치 인덱스
        idx_mu_start = 2*(C_prefix)
        idx_mu_end   = 2*(C_prefix + C_mu)

        # reshape to [B*M, 2*Cv, H, W]
        BM = B*M
        x_full = agg_full.reshape(BM, 2*Cv, H, W)

        # 멀티-μ 채널만 squeeze 1x1 적용
        mu_chunk = x_full[:, idx_mu_start:idx_mu_end, :, :]                          # [BM, 2*(3K), H, W]
        # 2배(agg concat)된 μ채널을 그대로 squeeze 대상으로 사용
        mu_squeezed = self.tm_squeeze(mu_chunk)                                      # [BM, tm_squeeze_ch, H, W]

        # squeeze된 멀티-μ를 원래 자리에 대체하고 refiner 입력을 구성
        x_left  = x_full[:, :idx_mu_start, :, :]
        x_right = x_full[:, idx_mu_end:, :, :]
        x_refiner = torch.cat([x_left, mu_squeezed, x_right], dim=1)                 # [BM, in_channels, H, W]
        H_T_mid_target = tonemap_mu(H_init, self.mu_list[self.mu_for_DA])   # [B,M,3,H,W]
        return x_refiner, weights, H_T_mid_target.reshape(BM,3,H,W)


    def forward(self, H_init, ldr_warped, linear_warped,
                valid_mask=None, depth_warped=None, confidence=None, snr_norm=None,
                project_min_bound: Optional[torch.Tensor]=None):
        """
        입력:
          H_init: [B,M,3,H,W]
          ldr/linear/conf/depth/snr: [B,M,N,*,H,W]
        출력:
          H_ref: [B,M,3,H,W], aux: dict(H_ref_log,res_log,weights,gate=None)
        """
        B,M,_,H,W = H_init.shape

        # ----- build features (멀티-μ squeeze 포함) -----
        x_refiner, weights, H_T_mid_for_DA = self._build_feats_multi_mu(
            H_init, ldr_warped, linear_warped, valid_mask, depth_warped, confidence, snr_norm
        )  # x_refiner: [B*M, in_channels, H, W]

        # ----- DA feats (μ 중간값 기준) -----
        da_feat = self.da_adapter(H_T_mid_for_DA)  # [B*M, da_ch, H/4, W/4]

        # ----- core -----
        feat = self.refine_core(x_refiner, da_feat)    # [B*M, base, H, W]

        # ----- Absolute path (공통 tonemap) -----
        H_T_abs = torch.sigmoid(self.head_absT(feat))  # (0..1), [B*M,3,H,W]

        # ----- Multi-μ residual path -----
        # μ별 초기 tonemap/logit
        H0_T_mu = [tonemap_mu(H_init.reshape(B*M,3,H,W), mu) for mu in self.mu_list]     # list of [B*M,3,H,W]
        eps = 1e-4
        z0_mu = [ (x.clamp(eps, 1-eps).log() - (1 - x.clamp(eps,1-eps)).log())
                  for x in H0_T_mu ]                                                     # list of [B*M,3,H,W]

        # μ별 Δz, cap
        r = float(self._residual_ratio.item())  # 0..residual_ratio_max
        H_T_mu_refined = []
        for i, z0 in enumerate(z0_mu):
            dz = self.head_dz_list[i](feat)
            dz = torch.tanh(self.res_norm(dz)) * r
            H_T_i = torch.sigmoid(z0 + dz)                                           # [B*M,3,H,W]
            H_T_mu_refined.append(H_T_i)

        # μ mixture weights α (softmax over μ)  — 채널 공유(공간/픽셀별)
        alpha_logits = self.head_alpha(feat)                                         # [B*M, K, H, W]
        alpha = torch.softmax(alpha_logits, dim=1)                                   # [B*M, K, H, W]

        # 가중합 (채널 공유 → broadcast)
        H_T_blend = 0.0
        for i, H_T_i in enumerate(H_T_mu_refined):
            H_T_blend = H_T_blend + alpha[:, i:i+1, :, :].expand_as(H_T_i) * H_T_i   # [B*M,3,H,W]

        # absolute vs residual(μ-blend) 최종 혼합
        beta = r  # 동일 스케일로 사용
        H_ref_log = (1.0 - beta) * H_T_abs + beta * H_T_blend                        # [B*M,3,H,W]

        # ----- Linear domain + min bound -----
        # 역변환 μ는 중간 μ 사용(수치 안정 + 단조성 유지)
        H_ref_lin = inv_tonemap_mu(H_ref_log, self.mu_list[self.mu_for_DA])
        if project_min_bound is not None:
            H_ref_lin = torch.maximum(H_ref_lin, project_min_bound.reshape(B*M,1,H,W))

        H_ref = H_ref_lin.reshape(B, M, 3, H, W)

        return H_ref

# =========================
# Loss (Absolute 중심 + Improvement hinge)
# =========================
# =========================
# Extra losses for sharpness & color
# =========================
def laplacian_map_tonemap(x):
    # x: [B*M, 3, H, W] or [B, M, 3, H, W] (channel-wise)
    if x.dim() == 5:
        B,M,C,H,W = x.shape
        x = x.view(B*M, C, H, W)
    k = x.new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]]).view(1,1,3,3)
    k = k.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=x.shape[1])

def laplacian_loss_tonemap(x):
    return laplacian_map_tonemap(x).abs().mean()

def sobel_grad(x):
    # x: [B*M, 3, H, W] or [B, M, 3, H, W]
    if x.dim() == 5:
        B,M,C,H,W = x.shape
        x = x.view(B*M, C, H, W)
    kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
    ky = x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
    kx = kx.repeat(x.shape[1],1,1,1); ky = ky.repeat(x.shape[1],1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return gx, gy

def mean_var_2d(x):
    # x: [B*M, 3, H, W] or [B, M, 3, H, W]
    if x.dim() == 5:
        B,M,C,H,W = x.shape
        x = x.view(B*M, C, H, W)
    m = x.mean(dim=(2,3), keepdim=True)
    v = x.var(dim=(2,3), keepdim=True, unbiased=False)
    return m, v

def local_std_2d(x, k=5, eps=1e-8):
    # x: [B*M, 3, H, W] or [B, M, 3, H, W]
    if x.dim() == 5:
        B,M,C,H,W = x.shape
        x = x.view(B*M, C, H, W)
    mean = F.avg_pool2d(x, k, stride=1, padding=k // 2)
    mean_sq = F.avg_pool2d(x * x, k, stride=1, padding=k // 2)
    return (mean_sq - mean * mean).clamp_min(0.0).add(eps).sqrt()

def highpass_2d(x, k=5):
    if x.dim() == 5:
        B,M,C,H,W = x.shape
        x = x.view(B*M, C, H, W)
    return x - F.avg_pool2d(x, k, stride=1, padding=k // 2)


# =========================
# Loss (Charbonnier + SSIM + SI-Log + Improvement + sharpness/color terms)
# =========================
class HDRRefinerLoss(nn.Module):
    """
    Tonemap Charbonnier + (1-SSIM) + SI-LogRMSE + Baseline-aware improvement hinge
    + GT Laplacian/gradient/local-contrast matching to discourage blur.
    """
    def __init__(self, mu=1e4,
                 w_charb=0.5, w_ssim=0.5, w_si=0.2, w_imp=0.3, gamma=1e-3,
                 hard_weight: float = 2.0,
                 w_lap=0.04, w_grad=0.06, w_color=0.5, w_contrast=0.08,
                 w_l1=3.0, w_init_hp=0.08):
        super().__init__()
        self.mu = mu
        self.wc, self.ws, self.wsi, self.wi, self.gamma = w_charb, w_ssim, w_si, w_imp, gamma
        self.hard_weight = hard_weight
        self.w_lap, self.w_grad, self.w_color = w_lap, w_grad, w_color
        self.w_contrast = w_contrast
        self.w_l1 = w_l1
        self.w_init_hp = w_init_hp

    def si_log_rmse(self, H_hat, H_gt, eps=1e-8):
        d = (torch.log(H_hat+eps) - torch.log(H_gt+eps))
        m = d.mean(dim=(1,2,3), keepdim=True)
        return torch.sqrt(((d-m)**2).mean(dim=(1,2,3)) + 1e-12).mean()

    def forward(self,
                H_hat, aux,
                H_init: torch.Tensor,   # [B,M,3,H,W]
                H_gt: torch.Tensor,     # [B,M,3,H,W]
                hard_mask: Optional[torch.Tensor] = None  # [B,M,1,H,W]
                ):
        B,M,_,H,W = H_hat.shape
        T = lambda x: torch.concat([x, tonemap_mu(x, self.mu), tonemap_mu(x*10, self.mu //5),tonemap_mu(x * 1000, self.mu // 10)], dim=0)  # 두 스케일 tonemap 합치기
        B_H = B * 4
        # ---- base terms (tonemap domain) ----
        H_hat_T = T(H_hat); H_gt_T = T(H_gt); H_init_T = T(H_init)
        w_mask  = torch.ones(B_H,M,1,H,W, device=H_hat.device) if hard_mask is None else \
                  (1.0 + (self.hard_weight-1.0)*hard_mask).clamp_min(1.0)
        l_l1 = (H_hat_T - H_gt_T).abs().mean()
        l_charb_map = charbonnier(H_hat_T - H_gt_T)               # [B,M,3,H,W]
        l_charb = (l_charb_map.mean(2, keepdim=True) * w_mask).mean()

        l_ssim  = 1.0 - ssim_simple(H_hat_T.reshape(B_H*M,3,H,W), H_gt_T.reshape(B_H*M,3,H,W))
        l_si    = self.si_log_rmse(H_hat.view(B*M,3,H,W), H_gt.view(B*M,3,H,W))
        if not torch.isfinite(l_si):
            l_si = torch.tensor(0.0)

        # ---- improvement hinge (init vs refined) ----
        with torch.no_grad():
            E_init = charbonnier(H_init_T-H_gt_T).mean(dim=(1,2,3))
        E_pred = charbonnier(H_hat_T - H_gt_T).mean(dim=(1,2,3))
        l_imp  = torch.clamp(E_pred - E_init + self.gamma, min=0).mean()

        # ---- sharpness & color terms (tonemap domain) ----
        # Edge-weighted Gradient/Laplacian matching. Penalizing |lap(pred)| itself
        # pushes the refiner toward blur; matching GT curvature preserves detail.
        gx1, gy1 = sobel_grad(H_hat_T); gx2, gy2 = sobel_grad(H_gt_T)
        with torch.no_grad():
            edge_mag = (gx2.pow(2) + gy2.pow(2) + 1e-12).sqrt().mean(1, keepdim=True)
            edge_norm = edge_mag.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            edge_w = (1.0 + 3.0 * edge_mag / edge_norm).clamp(1.0, 6.0)
        l_grad = (
            ((gx1 - gx2).abs().mean(1, keepdim=True) * edge_w).mean()
            + ((gy1 - gy2).abs().mean(1, keepdim=True) * edge_w).mean()
        )

        lap_pred = laplacian_map_tonemap(H_hat_T)
        with torch.no_grad():
            lap_gt = laplacian_map_tonemap(H_gt_T)
        l_lap = (charbonnier(lap_pred - lap_gt).mean(1, keepdim=True) * edge_w).mean()

        # Local contrast lower-bound: only punish lost GT texture/edge energy.
        std_pred = local_std_2d(H_hat_T, k=5).mean(1, keepdim=True)
        with torch.no_grad():
            std_gt = local_std_2d(H_gt_T, k=5).mean(1, keepdim=True)
            tex_norm = std_gt.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            tex_w = (std_gt / tex_norm).clamp(0.0, 4.0)
        l_contrast = (F.relu(std_gt - std_pred) * tex_w).mean()

        # Preserve details that both init and GT agree are real. This prevents the
        # refiner from erasing correctly warped high frequencies while avoiding
        # blind copying of init-only noise or misregistration artifacts.
        gx_init, gy_init = sobel_grad(H_init_T)
        with torch.no_grad():
            init_edge = (gx_init.pow(2) + gy_init.pow(2) + 1e-12).sqrt().mean(1, keepdim=True)
            init_norm = init_edge.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            common_detail_w = torch.minimum(edge_mag / edge_norm, init_edge / init_norm).clamp(0.0, 4.0)
        hp_pred = highpass_2d(H_hat_T, k=5)
        with torch.no_grad():
            hp_init = highpass_2d(H_init_T, k=5)
        l_init_hp = (charbonnier(hp_pred - hp_init).mean(1, keepdim=True) * common_detail_w).mean()

        # Color consistency (mean/var matching)
        m1,v1 = mean_var_2d(H_hat_T); m2,v2 = mean_var_2d(H_gt_T)
        l_color = (m1-m2).abs().mean() + (torch.sqrt(v1+1e-8)-torch.sqrt(v2+1e-8)).abs().mean()

        # ---- total ----
        loss = ( self.wc*l_charb + self.ws*l_ssim + self.wsi*l_si + self.wi*l_imp
               + self.w_lap*l_lap + self.w_grad*l_grad + self.w_color*l_color
               + self.w_contrast*l_contrast + self.w_init_hp*l_init_hp
               + self.w_l1*l_l1 )

        logs = {
            "loss": loss.detach(),
            "l_l1": l_l1.detach(),
            "l_charb": l_charb.detach(), "l_ssim": l_ssim.detach(),
            "l_si": l_si.detach(), "l_imp": l_imp.detach(),
            "l_lap": (self.w_lap*l_lap).detach(),
            "l_grad": (self.w_grad*l_grad).detach(),
            "l_color": (self.w_color*l_color).detach(),
            "l_contrast": (self.w_contrast*l_contrast).detach(),
            "l_init_hp": (self.w_init_hp*l_init_hp).detach(),
        }
        return loss, logs
