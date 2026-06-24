import math, torch
import torch.nn as nn
import torch.nn.functional as F

def tm_const_from_linear(x_lin: torch.Tensor, mu: float) -> torch.Tensor:
    """외부 입력을 tonemap(μ)으로 변환하되 grad 필요 없음."""
    with torch.no_grad():
        y = torch.log1p(mu * x_lin.clamp_min(0.0)) / math.log1p(mu)
    return y

def sat_mask_from_tm(x_tm: torch.Tensor, thr: float = 0.99) -> torch.Tensor:
    """tonemap된 참조에서 포화 마스크 생성 (no grad)."""
    with torch.no_grad():
        sat = (x_tm.max(1, keepdim=True)[0] > thr).float()
    return sat
class MuEmbed(nn.Module):
    """
    scalar μ → 작은 벡터 임베딩.
    log-공간 특징과 sin/cos positional을 섞어 분리력 확보.
    """
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        # 첫층은 identity; MLP 한 번만 사용 (경량)
        self.proj = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        mu: [B] or [B,1] (양의 실수)
        return: [B, dim]
        """
        mu = mu.view(-1)
        eps = 1e-8
        l  = torch.log(mu.clamp_min(eps))
        v  = torch.stack([
            l,                                 # ln μ
            l / math.log(2.0),                # log2 μ
            l / math.log(10.0),               # log10 μ
            1.0 / mu.clamp_min(eps),          # 1/μ
            torch.sin(l), torch.cos(l),       # sinusoid on ln μ
            torch.sin(0.5*l), torch.cos(0.5*l)
        ], dim=-1)                            # [B,8]
        return self.proj(v)                   # [B,dim]


class MuFiLM(nn.Module):
    """
    μ-조건부 Feature-wise Affine: feat_μ = feat * (1 + γ) + β
    - zero-init로 시작 → 초기에 항등.
    """
    def __init__(self, in_ch: int, mu_dim: int = 32, zero_init: bool = True):
        super().__init__()
        self.mu_embed = MuEmbed(mu_dim)
        self.to_gamma_beta = nn.Linear(mu_dim, 2*in_ch)
        if zero_init:
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, feat: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        feat: [B*, C, H, W]
        mu:   [B*] or [B*,1]
        """
        B, C, H, W = feat.shape
        e = self.mu_embed(mu.view(-1))                 # [B*, D]
        gb = self.to_gamma_beta(e).view(B, 2, C, 1, 1) # [B*,2,C,1,1]
        gamma, beta = gb[:,0], gb[:,1]                 # [B*,C,1,1] each
        return feat * (1.0 + gamma) + beta


# (A) Edge-aware Sharpen Head (tonemap 공간에서 샤픈 후 inv_tonemap)
class EdgeAwareSharpenHead(nn.Module):
    def __init__(self, in_ch: int, s_max: float = 0.8):
        super().__init__()
        # base tonemap 예측 (기존 head_absT 대체 아님: 아래 MoE와 조합됨)
        self.base = nn.Conv2d(in_ch, 3, 3, padding=1)
        # per-pixel sharpen 게인 s∈[0, s_max]
        self.smap = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
        self.s_max = float(s_max)
        k = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        #self.register_buffer('lap', k.view(1,1,3,3))  # depthwise Laplacian kernel

    def forward(self, feat, ref_rgb_tm: torch.Tensor | None = None):
        """
        feat: [B*, C, H, W]  (refine_core 출력의 feature)
        ref_rgb_tm: [B*, 3, H, W] or None (tonemap(μ_mid)된 ref 뷰; 포화 마스킹용)
        return: sharpened tonemap RGB in [0,1]
        """
        
        base = torch.sigmoid(self.base(feat))                 # tonemap domain
        s = self.smap(feat) * self.s_max                      # [B*,1,H,W]
        
        if ref_rgb_tm is not None:
            # ref 포화 영역은 샤픈 비활성화 → 복사/링잉 방지
            sat = (ref_rgb_tm.max(1, keepdim=True)[0] > 0.99).float()
            s = s * (1.0 - sat)
        sharp = base + s * (base - base.detach()*0)  # 동일하지만 grad 경로 단일화
        sharp = sharp.clamp(0.0 + 1e-10, 1.0 - 1e-10)  # 경계에서 완충
        return sharp.clamp(0.0, 1.0)

# (B) Top-K 선형소스 집계 (weights: PerPixelViewAggregator의 w 사용)
def topk_linear_stack(linear_warped: torch.Tensor,
                           weights: torch.Tensor,
                           k: int = 3,
                           reduce: str = "weighted"):
    """
    linear_warped: [B,M,N,3,H,W]
    weights:       [B,M,N,1,H,W]  (픽셀별 view score; 예: confidence*snr)
    return:
      L_fused:     [B,M,3,H,W]    (Top-K weighted or mean)
    """
    assert linear_warped.dim() == 6 and weights.dim() == 6
    B,M,N,C,H,W = linear_warped.shape
    assert C == 3, "채널 3 기준"
    device = linear_warped.device
    dtype  = linear_warped.dtype

    # score: [B,M,N,H,W] → topk over N
    score = weights.squeeze(3)                         # [B,M,N,H,W]
    # (선택) 뷰별 global score로 top-K 뽑되, 이후 픽셀별 가중합은 weights로 유지
    score_global = score.mean(dim=(-1,-2))            # [B,M,N]
    k = min(k, N)
    idx = torch.topk(score_global, k, dim=2).indices  # [B,M,k]
    idx = idx.to(torch.long).contiguous()

    # gather 준비: dim=2(N)에서 뽑는다.
    # idx_expanded: [B,M,k,1,1,1] → expand → [B,M,k,3,H,W]
    idx_expanded = idx.view(B, M, k, 1, 1, 1).expand(B, M, k, C, H, W).contiguous()

    # 입력 contiguous 보장
    lin = linear_warped.contiguous()
    wts = weights.contiguous()

    # gather: dim=2
    Lk  = torch.gather(lin, 2, idx_expanded)          # [B,M,k,3,H,W]
    wk  = torch.gather(wts, 2, idx_expanded[:, :, :, :1])  # [B,M,k,1,H,W]

    if reduce == "weighted":
        wk_sum = wk.sum(dim=2, keepdim=True)
        wk = wk / (wk_sum + (wk_sum==0).float() * 1.0 + 1e-8)  # 전부 0이면 uniform로 대체
        L_fused = (wk * Lk).sum(dim=2)                # [B,M,3,H,W]
    elif reduce == "mean":
        L_fused = Lk.mean(dim=2)
    else:
        raise ValueError("reduce ∈ {weighted, mean}")

    # 방어적 체크 (디버깅 시 활성화)
    # assert (idx >= 0).all() and (idx < N).all()
    # assert torch.isfinite(L_fused).all()

    return L_fused

# (C) MoE Combiner (tonemap 공간에서의 부드러운 혼합)
class MoECombiner(nn.Module):
    """
    H_T_abs, H_T_blend, (옵션) H_init_T, (옵션) Top-K 선형소스 L_lin → tonemap(μ_mid)
    를 per-pixel softmax 가중으로 혼합.
    """
    def __init__(self, in_ch: int, use_hinit: bool = True, use_src: bool = True):
        super().__init__()
        self.use_hinit = use_hinit
        self.use_src   = use_src
        # 최대 4개 컴포넌트: abs, blend, hinit, src
        n_comp = 2 + int(use_hinit) + int(use_src)
        self.head_logits = nn.Conv2d(in_ch, n_comp, 1)
        

    def forward(self, feat,
                H_abs_T, H_blend_T,
                H_init_T: torch.Tensor | None = None,
                L_src_T:  torch.Tensor | None = None,
                valid_mask: torch.Tensor | None = None,
                ref_sat_tm: torch.Tensor | None = None):
        """
        * 모든 입력 tonemap(μ_mid) 공간에서 [0,1] 스케일 권장
        feat:      [B*, C, H, W]
        H_abs_T:   [B*, 3, H, W]
        H_blend_T: [B*, 3, H, W]
        H_init_T:  [B*, 3, H, W] or None
        L_src_T:   [B*, 3, H, W] or None   (Top-K 선형소스 가중합 후 tonemap)
        valid_mask:[B*, 1, H, W] or None   (가시성/기하 신뢰)
        ref_sat_tm:[B*, 1, H, W] or None   (포화 마스크)
        """
        comps = [H_abs_T, H_blend_T]
        if self.use_hinit and (H_init_T is not None):
            comps.append(H_init_T)
        if self.use_src and (L_src_T is not None):
            comps.append(L_src_T)
        X = torch.stack(comps, dim=1)   # [B*, Cn, 3, H, W]
        Bm, Cn, _, H, W = X.shape
        logits = self.head_logits(feat)            # [B*, n_comp, H, W]
        # 신뢰·포화 마스크로 컴포넌트 억제
        if ref_sat_tm is not None:
            # ref 포화 → H_init, H_abs 억제 (복사/밴딩 방지), src/blend는 유지
            # 인덱스: 0=abs, 1=blend, (2=hinit), (3=src)
            sat = ref_sat_tm.clamp(0,1)                          # [B*,1,H,W]
            if Cn >= 1: logits[:, 0:1] = logits[:, 0:1] - 5.0*sat
            if Cn >= 3: logits[:, 2:3] = logits[:, 2:3] - 5.0*sat
        if valid_mask is not None and self.use_src and (Cn == 4 or Cn == 3):
            # src가 마지막이면 valid가 낮을수록 src 억제
            
            logits[:, -1:] = logits[:, -1:] + 3.0*(valid_mask-1.0)

        alpha = torch.softmax(logits, dim=1).unsqueeze(2)        # [B*, n_comp, 1, H, W]
        mix = (alpha * X).sum(1)                                 # [B*, 3, H, W]
        return mix

def tonemap_mu(x, mu=5e4, eps=1e-6):
    with torch.cuda.amp.autocast(enabled=False):
        x32 = x.to(torch.float32)
        y32 = torch.log1p(mu * (x32.clamp_min(0.) + eps)) / math.log1p(mu)
    return y32.to(x.dtype)

def inv_tonemap_mu(y, mu=5e4):
    with torch.cuda.amp.autocast(enabled=False):
        y32 = y.to(torch.float32)
        x32 = torch.expm1(y32 * math.log1p(mu)) / mu
        x32 = x32.clamp_min(0.)
    return x32.to(y.dtype)

# --- 경량 윈도우 블록 ---
class DWConvGN(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k//2, groups=c)
        self.pw = nn.Conv2d(c, c, 1)
        self.gn = nn.GroupNorm(1, c, eps=1e-5)
        self.act= nn.GELU()
    def forward(self, x):
        return self.act(self.gn(self.pw(self.dw(x))))


class LiteWindowBlock(nn.Module):
    """ Self-Attn(local) + DWConv mix """
    def __init__(self, c, heads=3, win=8):
        super().__init__()
        self.c, self.h, self.w = c, heads, win
        self.qkv = nn.Conv2d(c, c*3, 1)
        self.proj= nn.Conv2d(c, c, 1)
        self.dw  = DWConvGN(c, 5)
    def _partition(self, x):
        B,C,H,W = x.shape; w=self.w
        x = x.view(B,C,H//w,w,W//w,w).permute(0,2,4,1,3,5).contiguous() # B, nH, nW, C, w, w
        return x.view(-1,self.c,w,w)
    def _reverse(self, x, H, W):
        w=self.w; BHW = x.size(0); nH = H//w; nW = W//w; B = BHW//(nH*nW)
        x = x.view(B,nH,nW,self.c,w,w).permute(0,3,1,4,2,5).contiguous()
        return x.view(B,self.c,H,W)
    def forward(self, x):
        B,C,H,W = x.shape
        win = self._partition(x)                  # [B*nW, C, w, w]
        q,k,v = self.qkv(win).chunk(3, dim=1)
        # head split
        def to_heads(t): 
            Bn, C, w, _ = t.shape; h=self.h; d=C//h
         
            tt = t.view(Bn,h,d,w*w)
 
            return tt
        
        q,k,v = map(to_heads,(q,k,v))
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        attn = (q.transpose(2,3) @ k) / math.sqrt(q.size(2))  # [Bn,h,N,N]
        attn = attn.softmax(-1)
        out  = (attn @ v.transpose(2,3)).transpose(2,3)       # [Bn,h,d,N]
        # merge heads
        Bn,h,d,N = out.shape
        out = out.reshape(Bn,h*d,int(math.sqrt(N)),int(math.sqrt(N)))
        out = self.proj(out)
        out = self._reverse(out, H, W)
        return self.dw(x + out)

class LiteCrossWindowBlock(nn.Module):
    """ Cross-Attn(local): Q from x, KV from y """
    def __init__(self, c, heads=3, win=8):
        super().__init__()
        self.c, self.h, self.w = c, heads, win
        self.q = nn.Conv2d(c, c, 1); self.kv = nn.Conv2d(c, 2*c, 1)
        self.proj= nn.Conv2d(c, c, 1)
        self.dw  = DWConvGN(c, 3)
    def _part(self, x):
        B,C,H,W = x.shape; w=self.w
 
        x = x.view(B,C,H//w,w,W//w,w).permute(0,2,4,1,3,5).contiguous()
        return x.view(-1,self.c,w,w), H, W
    def _rev(self, x, H, W):
        w=self.w; Bn=x.size(0); nH=H//w; nW=W//w; B=Bn//(nH*nW)
        x = x.view(B,nH,nW,self.c,w,w).permute(0,3,1,4,2,5).contiguous()
        return x.view(B,self.c,H,W)
    def forward(self, x, y):
        B,C,H,W = x.shape
        q,H,W = self._part(self.q(x))
        kv = self.kv(y)
        k = kv[:, :C]
        v = kv[:, C:]
        k = self._part(k)[0]
        v = self._part(v)[0]
        
        # heads
        def to_heads(t): 
            Bn,C,w,_= t.shape; h=self.h; d=C//h
         
            mm = t.view(Bn,h,d,w*w)

            return mm
        q,k,v = map(to_heads,(q,k,v))
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        attn = (q.transpose(2,3) @ k) / math.sqrt(q.size(2))
        attn = attn.softmax(-1)
        out  = (attn @ v.transpose(2,3)).transpose(2,3)
        Bn,h,d,N = out.shape
        out = out.reshape(Bn,h*d,int(math.sqrt(N)),int(math.sqrt(N)))
        out = self.proj(out)
        out = self._rev(out, H, W)
        return self.dw(x + out)

    
def build_L_src_T_no_grad(linear_warped, confidence, snr_norm, mu_mid, k=3, reduce="weighted"):
    """
    linear_warped: [B,M,N,3,H,W]
    confidence,snr_norm: [B,M,N,1,H,W]
    return: L_src_T (tonemap, no-grad) [B*M,3,H,W]
    """
    with torch.no_grad():
        B,M,N,C,H,W = linear_warped.shape
        assert C == 3

        # (1) view score
        score = (confidence * snr_norm).squeeze(3)            # [B,M,N,H,W]
        score_global = score.mean(dim=(-1,-2))                # [B,M,N]

        # (2) top-k indices on N axis
        k = min(k, N)
        idx = torch.topk(score_global, k, dim=2).indices      # [B,M,k]
        idxL = idx.view(B, M, k, 1, 1, 1)                     # for linear_warped gather
        idxW = idx.view(B, M, k, 1, 1, 1)                     # for weight gather

        # (3) gather along dim=2 (N axis)
        # expand index to match target tensor's shape except gather dim
        Lk = torch.gather(linear_warped, 2, idxL.expand(B, M, k, C, H, W))     # [B,M,k,3,H,W]
        wk = torch.gather(score.unsqueeze(3), 2, idxW.expand(B, M, k, 1, H, W))# [B,M,k,1,H,W]

        # (4) normalize weights & fuse
        if reduce == "weighted":
            wk_sum = wk.sum(dim=2, keepdim=True)
            wk = wk / (wk_sum + (wk_sum == 0).float() + 1e-8)
            L_fused = (wk * Lk).sum(dim=2)                                     # [B,M,3,H,W]
        elif reduce == "mean":
            L_fused = Lk.mean(dim=2)
        else:
            raise ValueError("reduce ∈ {weighted, mean}")

        # (5) tonemap (no grad) → [B*M,3,H,W]
        L_src_T = tm_const_from_linear(L_fused, mu_mid).reshape(B*M, 3, H, W)
        return L_src_T

def _collapse_src(x: torch.Tensor, reduce: str = "max", color_to_gray: bool = False) -> torch.Tensor:
    """
    x: [B,M,N,C,H,W] or [B,M,N,1,H,W] -> [B,M,C,H,W]
    reduce: "max" | "mean"
    """
    if x is None:
        return None
    if x.dim() == 6:
        if reduce == "max":
            x = x.max(dim=2, keepdim=False).values
        elif reduce == "mean":
            x = x.mean(dim=2)
        else:
            raise ValueError(f"reduce must be max|mean, got {reduce}")
    if color_to_gray and x.shape[2] == 3:
        B,M,C,H,W = x.shape
        x = x.reshape(B,M,3,-1,H,W)
        x = x.mean(2)  # [B,M,1,H,W]
    # if already [B,M,C,H,W], pass through
    return x
# --- Refiner v4 Lite ---
class HDRRefinerV4Lite(nn.Module):
    def __init__(self, da_adapter, ref_adapter=None,
                 C=48, heads=3, win=8, depth_self=3, depth_cross=2,
                 topk_views=4, mu_list=(1e3,5e4,1e6), mu_mid_idx=1,
                 residual_cap=2.0, drop_ref_prob=0.3):
        super().__init__()
        self.mu_list = tuple(mu_list)
        self.mu_mid  = self.mu_list[mu_mid_idx]
        self.topk = topk_views
        self.drop_ref_prob = drop_ref_prob
        # DA / Ref adapters (재사용 가능)
        self.da_adapter  = da_adapter
        self.ref_adapter = ref_adapter

        # 입력 feature 축약: (ldr, lin, conf, depth, snr, multi-μ, gray)
        in_ch = (3+3+1+1+1) + (3*len(self.mu_list)) + 1
        self.stem = nn.Conv2d(in_ch * topk_views, C, 3, padding=1)

        # Stage-shared 경량 블록 (recurrent)
        self.self_blk  = nn.ModuleList([LiteWindowBlock(C, heads, win) for _ in range(depth_self)])
        self.cross_blk = nn.ModuleList([LiteCrossWindowBlock(C, heads, win) for _ in range(depth_cross)])

        # Ref/DA 주입을 위한 1×1
        self.da_proj  = nn.Conv2d(128, C, 1)
        self.ref_proj = nn.Conv2d(128, C, 1)
        self.gate_ref = nn.Parameter(torch.tensor(0.1))  # 게이트 초기 약함
        self.gate_da  = nn.Parameter(torch.tensor(0.1))

        # 헤드: μ-mixture + absolute
        self.head_abs = nn.Conv2d(C, 3, 3, padding=1)
        self.head_mu  = nn.ModuleList([nn.Conv2d(C, 3, 3, padding=1) for _ in self.mu_list])
        self.alpha    = nn.Conv2d(C, len(self.mu_list), 1)

        self.res_norm = nn.GroupNorm(1, 3)
        self.res_cap  = residual_cap
        
        self.moe = MoECombiner(in_ch=C, use_hinit=False, use_src=True)
        self.sharpen = EdgeAwareSharpenHead(in_ch=C, s_max=0.8)

        self.mu_film = MuFiLM(in_ch=C, mu_dim=32, zero_init=True)

    def _build_feats(self, H_init, ldr, lin, conf, depth, snr):
        # multi-μ & gray (중간 μ)
        B,M,N,_,H,W = ldr.shape
        Hexp = H_init.unsqueeze(2).expand(B,M,N,3,H,W)
        H_Ts = [tonemap_mu(Hexp, m) for m in self.mu_list]
        H_Tc = torch.cat(H_Ts, dim=3)
        H_mid= tonemap_mu(Hexp, self.mu_mid)
        Hgry = H_mid.mean(3, keepdim=True)
        feats = torch.cat([ldr, lin, conf, depth, snr, H_Tc, Hgry], dim=3)  # [B,M,N,Cf,H,W]
        # Top-K view 선택 (scores: conf*snr*valid 근사)

        with torch.no_grad():
            q = (conf * snr).squeeze(3).mean(dim=(-1,-2))        # [B,M,N]
            Hexp_mid = tonemap_mu(H_init.unsqueeze(2).expand(B,M,N,3,H,W), self.mu_mid)
            Y = (0.2126*Hexp_mid[:,:,:,0] + 0.7152*Hexp_mid[:,:,:,1] + 0.0722*Hexp_mid[:,:,:,2])  # [B,M,N,H,W]
            e = Y.median(dim=-1).values.median(dim=-1).values    # [B,M,N]
            sat_hi = (Y > 0.995).float().mean(dim=(-1,-2))       # [B,M,N]
            sat_lo = (Y < 0.005).float().mean(dim=(-1,-2))       # [B,M,N]

            sat_hi_thr, sat_lo_thr = 0.50, 0.50
            valid = (sat_hi < sat_hi_thr) & (sat_lo < sat_lo_thr)  # [B,M,N]

            topk = min(self.topk, N)
            idx = torch.empty(B, M, topk, dtype=torch.long, device=ldr.device)  # 미리 할당

            λ = 0.1
            for b in range(B):
                for m_ in range(M):
                    q_bm = q[b, m_]      # [N]
                    e_bm = e[b, m_]      # [N]
                    v_bm = valid[b, m_]  # [N]

                    # 분위수 경계 (유효 후보가 없으면 전체로 계산)
                    if v_bm.any():
                        e_lo = torch.quantile(e_bm[v_bm], 1/3)
                        e_hi = torch.quantile(e_bm[v_bm], 2/3)
                    else:
                        e_lo = torch.quantile(e_bm, 1/3)
                        e_hi = torch.quantile(e_bm, 2/3)

                    g_low  = (e_bm <= e_lo) & v_bm
                    g_mid  = (e_bm >  e_lo) & (e_bm <  e_hi) & v_bm
                    g_high = (e_bm >= e_hi) & v_bm

                    picked = []
                    for g in (g_low, g_mid, g_high):
                        if g.any():
                            i = torch.argmax(q_bm.masked_fill(~g, -1e9)).item()
                            picked.append(i)

                    remain = [i for i in range(N) if i not in picked]
                    # 보충(다양성): q - λ * Σ 1/|e - e_sel|
                    while len(picked) < topk and len(remain) > 0:
                        if len(picked) == 0:
                            i = int(torch.argmax(q_bm).item())
                            picked.append(i); remain.remove(i); continue
                        e_sel = e_bm[torch.tensor(picked, device=e_bm.device)]
                        best_i, best_val = None, -1e9
                        for i in remain:
                            dist = (e_bm[i] - e_sel).abs().clamp_min(1e-6)
                            val = q_bm[i] - λ * (1.0 / dist).sum()
                            if val > best_val:
                                best_val, best_i = val, i
                        picked.append(best_i); remain.remove(best_i)

                    # 여전히 모자라면 q 상위로 패딩
                    if len(picked) < topk:
                        top_q = torch.topk(q_bm, k=min(topk, N)).indices.tolist()
                        for i in top_q:
                            if len(picked) >= topk: break
                            if i not in picked: picked.append(i)

                    # 그래도 모자라면 마지막 것을 반복해서 길이 맞춤
                    if len(picked) < topk:
                        picked += [picked[-1]] * (topk - len(picked))

                    idx[b, m_] = torch.tensor(picked[:topk], device=idx.device, dtype=torch.long)


        # 이후 동일
        idx_e   = idx.unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1,-1,-1,feats.size(3),H,W)
        feats_k = torch.gather(feats, 2, idx_e).contiguous()                      # [B,M,K,Cf,H,W]
        x       = torch.cat([feats_k[:,:,i] for i in range(feats_k.size(2))], 2)  # [B,M,K*Cf,H,W]
        
        return x  # K-fold concat


    def forward(self, H_init, ldr_warped, linear_warped,
                valid_mask=None, depth_warped=None, confidence=None, snr_norm=None,
                project_min_bound=None,
                ref_view_rgb=None,
                ref_index=None,
                drop_ref_prob: float=0.3):
        
        B,M,_,H,W = H_init.shape
        dev = H_init.device
        if valid_mask is None:   valid_mask   = torch.ones(B,M,ldr_warped.size(2),1,H,W, device=dev)
        if confidence is None:   confidence   = valid_mask
        if depth_warped is None: depth_warped = torch.zeros_like(valid_mask)
        if snr_norm is None:     snr_norm     = torch.ones_like(valid_mask)*0.5

        # 1) 특징 구축(+Top-K 축소)
        xk = self._build_feats(H_init, ldr_warped, linear_warped, confidence, depth_warped, snr_norm)
        x  = self.stem(xk.reshape(B*M, xk.size(2), H, W))

        # 2) DA / Ref 특징 (μ_mid tonemap)
        H_T_mid = tonemap_mu(H_init, self.mu_mid).reshape(B*M,3,H,W)
        
        with torch.cuda.amp.autocast(enabled=True):
            da_feat = self.da_adapter(H_T_mid)           # [B*M,128,h/4,w/4] 가정
            da_feat = F.interpolate(da_feat, size=(H,W), mode='bilinear', align_corners=False)
            da_feat = self.da_proj(da_feat)

        ref_feat = None
        if ref_view_rgb is None and ref_index is not None:
            ref_lin = H_init[:,ref_index]                               # [B,3,H,W]
            ref_view_rgb = (ref_lin/(ref_lin.amax((-1,-2),keepdim=True)+1e-8)).unsqueeze(1).expand(B,M,3,H,W)
        if ref_view_rgb is not None:
            ref_T = tonemap_mu(ref_view_rgb.reshape(B*M,3,H,W), self.mu_mid)
        else:
            # dummy ref_T
            ref_T = H_T_mid * 0.0
        with torch.cuda.amp.autocast(enabled=True):
            ref_adapter = self.ref_adapter if self.ref_adapter is not None else self.da_adapter
            ref_feat = ref_adapter(ref_T)
            ref_feat = F.interpolate(ref_feat, size=(H,W), mode='bilinear', align_corners=False)
            ref_feat = self.ref_proj(ref_feat)

        # Drop-Ref & Sat stop-grad
        if ref_view_rgb is not None:
            rf = ref_view_rgb.reshape(B*M,3,H,W)
            sat = (rf.max(1,keepdim=True).values > 0.99).float()
            ref_feat = ref_feat*(1.0 - sat) + ref_feat.detach()*sat

        # 3) Recurrent stage
        # Self → Cross(DA) → (optional) Cross(Ref)
        with torch.cuda.amp.autocast(enabled=True):
            for blk in self.self_blk:
                x = blk(x)
            for blk in self.cross_blk:
                x = x + torch.sigmoid(self.gate_da) * (blk(x, da_feat) - x)
            if ref_feat is not None:
                g = torch.sigmoid(self.gate_ref)
                for blk in self.cross_blk:
                    x = x + g * (blk(x, ref_feat) - x)

            # ... (기존 전처리, _build_feats_multi_mu, da/ref features, refine_core 등 동일)
            feat = x
            BM, _, H, W = feat.shape
            eps = 1e-7

            with torch.cuda.amp.autocast(enabled=False):
                H0_T = H_init.reshape(BM, 3, H, W).to(torch.float32)
                # HDR 값 보존: 매우 작은 값(어두운 영역)도 유지하되, 0 근처만 안전하게 처리
                H0_T_safe = torch.where(
                    H0_T > eps,
                    H0_T,
                    torch.full_like(H0_T, eps)  # 0에 가까운 값만 eps로 대체
                )
                H0_T_safe = torch.where(
                    H0_T_safe < 1.0 - eps,
                    H0_T_safe,
                    torch.full_like(H0_T_safe, 1.0 - eps)  # 1에 가까운 값만 제한
                )
                
                # logit 변환: log(x / (1-x)) - 안전한 범위에서만 계산
                z0 = torch.log(H0_T_safe) - torch.log1p(-H0_T_safe)
                
                # 극단값 제한 (float16 안전 범위: -65504 ~ 65504이지만, 실용적으로 -10~10)
                z0 = z0.clamp(-10.0, 10.0)
                
                # NaN/Inf 방어: finite하지 않은 값을 중립값(0)으로 대체
                z0 = torch.where(torch.isfinite(z0), z0, torch.zeros_like(z0))
            
            # dtype 변환 후 다시 한번 체크 (float32 -> float16 변환 시 overflow 방지)
            z0 = z0.to(feat.dtype)
            z0 = torch.where(torch.isfinite(z0), z0, torch.zeros_like(z0))
            H_T_mu_refined = []
            z_cap = 2.0   # ← 2.0 → 0.5
            for i, mu_val in enumerate(self.mu_list):
                # (1) batch 크기에 맞춘 μ 텐서 준비
                mu_tensor = torch.full((BM,), float(mu_val), device=feat.device, dtype=feat.dtype)

                # (2) μ-조건부 feature 변조
                feat_mu = self.mu_film(feat, mu_tensor)      # [B*M, base, H, W]

                # (3) μ별 head에 통과 (기존 head_dz_list 유지)
                dz_raw = self.head_mu[i](feat_mu)
                dz = self.res_norm(dz_raw)            # (B,3,H,W)
                dz = (dz * z_cap).clamp_(-z_cap, z_cap) 
                
                # (4) logit-space residual → tonemap 출구
                H_T_i = torch.sigmoid(z0 + dz)
                H_T_mu_refined.append(H_T_i)
            alpha_logits = self.alpha(feat)
            alpha = torch.softmax(alpha_logits, dim=1)
            H_T_blend = 0.0
            for i, H_T_i in enumerate(H_T_mu_refined):
                H_T_blend = H_T_blend + alpha[:, i:i+1] * H_T_i
            H_T_abs = torch.sigmoid(self.head_abs(feat))  # absolute branch (tonemap)

            mu_mid = self.mu_list[1]
            # === NEW: Mixture-of-Experts (tonemap 공간) ===

            H_init_T = None

            # (2) Top-K 선형소스 → tonemap
            L_src_T = None
            # # weights: [B,M,N,1,H,W]  → _build_feats_multi_mu에서 받은 값
            
            # # 이미 위에서 weights 리턴받음
            # # full 해상도 linear_warped: [B,M,N,3,H,W]
            # w_pix = (confidence * snr_norm).clamp_min(0)
            # L_fused = topk_linear_stack(linear_warped, w_pix, k=3)      # [B,M,3,H,W]
            # L_fused = L_fused.reshape(BM, 3, H, W)
            # # tonemap(μ_mid)
            # L_src_T = torch.log1p(mu_mid * L_fused.clamp_min(0.0)) / math.log1p(mu_mid)
            L_src_T = build_L_src_T_no_grad(linear_warped, confidence, snr_norm, mu_mid, k=4)

            # (3) 마스크
            ref_sat_tm = None
            # ref saturation mask
            if ref_view_rgb is not None:
                ref_tm = tm_const_from_linear(ref_view_rgb.reshape(B*M,3,H,W), self.mu_mid)  # no grad
                ref_sat_tm = sat_mask_from_tm(ref_tm)  # [B*M,1,H,W], no grad
            else:
                ref_sat_tm = None

            # valid mask collapse (max/mean), no grad
            valid_tm = None
            if valid_mask is not None:
                with torch.no_grad():
                    vm = _collapse_src(valid_mask, reduce="max", color_to_gray=True)  # [B,M,1,H,W]
                    valid_tm = vm.reshape(B*M,1,H,W)
            # (4) 혼합
            H_T_moe = self.moe(
                feat, H_abs_T=H_T_abs, H_blend_T=H_T_blend,
                H_init_T=H_init_T, L_src_T=L_src_T,
                valid_mask=valid_tm, ref_sat_tm=ref_sat_tm
            )
            H_T_mix = H_T_moe


            # === NEW: Edge-aware Sharpen (tonemap 공간) ===
        ref_tm_for_sharp = None
        if ref_view_rgb is not None:
            ref_tm_for_sharp = torch.log1p(mu_mid * ref_view_rgb.reshape(BM,3,H,W).clamp_min(0.0)) / math.log1p(mu_mid)
        H_T_final = self.sharpen(feat, ref_rgb_tm=ref_tm_for_sharp)  # 예측된 base를 쓰도록 설계했지만
        # base 대신 혼합 결과를 샤픈 입력으로 쓰고 싶으면 아래로 변경:
        # H_T_final = self.sharpen(torch.cat([feat],1), ref_rgb_tm=ref_tm_for_sharp) ; 다만 파라미터 수가 늘면 in_ch 맞춰야 함
        # 간단히: 혼합 결과에 라플라시안 적용
        # Laplacian을 혼합 결과에 적용하려면 sharpen 모듈을 약간 수정 필요. 여기서는 그대로 사용:
        H_T = H_T_mix * 0.5 + H_T_final * 0.5

        

        # linear 복귀
        H_ref_lin = (torch.exp(H_T * math.log1p(mu_mid)) - 1.0) / mu_mid
        if project_min_bound is not None:
            H_ref_lin = torch.maximum(H_ref_lin, project_min_bound.reshape(BM,1,H,W))
        H_ref = H_ref_lin.reshape_as(H_init)
        def check_finite(name, t):
            if not torch.isfinite(t).all():
                raise RuntimeError(f"[NaN/Inf] at {name}")

                # 예: 핵심 텐서 직후
        check_finite("z0", z0)
        check_finite("feat", feat)
        check_finite("H_T_blend", H_T_blend)
        return H_ref


    # def forward(self, H_init, ldr_warped, linear_warped,
    #             valid_mask=None, depth_warped=None, confidence=None, snr_norm=None,
    #             ref_view_rgb=None, ref_index=None):
    #     B,M,_,H,W = H_init.shape
    #     dev = H_init.device
    #     if valid_mask is None:   valid_mask   = torch.ones(B,M,ldr_warped.size(2),1,H,W, device=dev)
    #     if confidence is None:   confidence   = valid_mask
    #     if depth_warped is None: depth_warped = torch.zeros_like(valid_mask)
    #     if snr_norm is None:     snr_norm     = torch.ones_like(valid_mask)*0.5

    #     # 1) 특징 구축(+Top-K 축소)
    #     xk = self._build_feats(H_init, ldr_warped, linear_warped, confidence, depth_warped, snr_norm)
    #     x  = self.stem(xk.reshape(B*M, xk.size(2), H, W))

    #     # 2) DA / Ref 특징 (μ_mid tonemap)
    #     H_T_mid = tonemap_mu(H_init, self.mu_mid).reshape(B*M,3,H,W)
    #     da_feat = self.da_adapter(H_T_mid)           # [B*M,128,h/4,w/4] 가정
    #     da_feat = F.interpolate(da_feat, size=(H,W), mode='bilinear', align_corners=False)
    #     da_feat = self.da_proj(da_feat)

    #     ref_feat = None
    #     if ref_view_rgb is None and ref_index is not None:
    #         ref_lin = H_init[:,ref_index]                               # [B,3,H,W]
    #         ref_view_rgb = (ref_lin/(ref_lin.amax((-1,-2),keepdim=True)+1e-8)).unsqueeze(1).expand(B,M,3,H,W)
    #     if ref_view_rgb is not None:
    #         ref_T = tonemap_mu(ref_view_rgb.reshape(B*M,3,H,W), self.mu_mid)
    #     else:
    #         # dummy ref_T
    #         ref_T = H_T_mid * 0.0
    #     ref_feat = self.ref_adapter(ref_T)
    #     ref_feat = F.interpolate(ref_feat, size=(H,W), mode='bilinear', align_corners=False)
    #     ref_feat = self.ref_proj(ref_feat)

    #     # Drop-Ref & Sat stop-grad
    #     if ref_view_rgb is not None:
    #         rf = ref_view_rgb.reshape(B*M,3,H,W)
    #         sat = (rf.max(1,keepdim=True).values > 0.99).float()
    #         ref_feat = ref_feat*(1.0 - sat) + ref_feat.detach()*sat

    #     # 3) Recurrent stage
    #     # Self → Cross(DA) → (optional) Cross(Ref)
    #     for blk in self.self_blk:
    #         x = blk(x)
    #     for blk in self.cross_blk:
    #         x = blk(x, da_feat)
    #     if ref_feat is not None:
    #         gate = torch.sigmoid(self.gate_ref).clamp(0,1)
    #         for blk in self.cross_blk:
    #             x = x + gate * (blk(x, ref_feat) - x)

    #     # 4) 헤드 (μ-mixture + absolute)
    #     BM = B*M
    #     H0 = H_init.clamp(1e-6, 1-1e-6).reshape(BM,3,H,W)
    #     z0 = torch.log(H0) - torch.log1p(-H0)

    #     H_T_list = []
    #     for i, head in enumerate(self.head_mu):
    #         dz = self.res_norm(head(x)).clamp(-self.res_cap, self.res_cap)
    #         H_T = torch.sigmoid(z0 + dz)
    #         H_T_list.append(H_T)
    #     alpha = torch.softmax(self.alpha(x), dim=1)  # [BM,K,H,W]
    #     H_mix = 0.
    #     for i,H_T in enumerate(H_T_list):
    #         H_mix = H_mix + alpha[:,i:i+1]*H_T
    #     H_abs = torch.sigmoid(self.head_abs(x))
    #     # blend
    #     beta = 0.25  # 고정 or 스케줄러
    #     H_log = (1-beta)*H_abs + beta*H_mix
    #     H_lin = inv_tonemap_mu(H_log, self.mu_mid)
    #     return H_lin.reshape(B, M, 3, H, W)
