import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .loss_function import ms_ssim


def charbonnier(x, eps=1e-6): return torch.sqrt(x*x + eps)
def tonemap_mu(x, mu=1e4, eps=1e-6): 
    return torch.log1p(mu*(x.clamp_min(0.0)+eps)) / math.log1p(mu)
def si_log_rmse(hat, gt, eps=1e-8):
    d = (hat.clamp_min(eps).log() - gt.clamp_min(eps).log())
    m = d.mean(dim=(1,2,3), keepdim=True)
    return torch.sqrt(((d - m)**2).mean(dim=(1,2,3)) + 1e-12).mean()

def sobel_mag(x):
    # x: [B,C,H,W]
    kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
    ky = x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
    gx = F.conv2d(x, kx.expand(x.size(1),1,3,3), padding=1, groups=x.size(1))
    gy = F.conv2d(x, ky.expand(x.size(1),1,3,3), padding=1, groups=x.size(1))
    return (gx*gx + gy*gy + 1e-12).sqrt()

def feather_mask(m: torch.Tensor, k: int = 9, iters: int = 2):
    """hard mask ∈{0,1} → soft ramp mask ∈[0,1]"""
    x = m
    for _ in range(iters):
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)
    return x.clamp(0, 1)

def narrow_band(m_soft: torch.Tensor, low: float = 0.1, high: float = 0.9):
    """Boundary narrow band: only the low~high region of soft mask is 1"""
    return ((m_soft > low) & (m_soft < high)).float()

def band_grad_smooth_loss(y_tm: torch.Tensor, band: torch.Tensor, w: float = 5e-4):
    """
    Gradient smoothing loss in boundary band to prevent sharp transitions.
    y_tm: [B,3,H,W] tonemap space output
    band: [B,1,H,W] boundary band mask
    """
    gx = y_tm[..., :, 1:] - y_tm[..., :, :-1]
    gy = y_tm[..., 1:, :] - y_tm[..., :-1, :]
    bx = band[..., :, 1:]
    by = band[..., 1:, :]
    return w * ((gx.abs() * bx).mean() + (gy.abs() * by).mean())

def alpha_tv_loss(alpha: torch.Tensor, band: torch.Tensor, w: float = 1e-4):
    """
    Alpha weight map TV loss in boundary band (optional for MoE models).
    alpha: [B,K,H,W] (softmax weights)
    band: [B,1,H,W] boundary band mask
    """
    ax = alpha[..., :, 1:] - alpha[..., :, :-1]
    ay = alpha[..., 1:, :] - alpha[..., :-1, :]
    bx = band[..., :, 1:]
    by = band[..., 1:, :]
    return w * ((ax.abs() * bx).mean() + (ay.abs() * by).mean())

def ref_edge_consistency(H_ref_tm, ref_tm, sat_mask, w=0.2):
    """
    Reference-edge-only consistency loss.
    Only match gradients (edges) from reference, excluding saturated regions.
    
    Args:
        H_ref_tm: Output HDR tonemapped [B,3,H,W]
        ref_tm: Reference view tonemapped [B,3,H,W]
        sat_mask: Saturation mask [B,1,H,W], 1=saturated
        w: Weight coefficient
    """
    def gradmag(x):
        kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
        ky = x.new_tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).view(1,1,3,3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return (gx*gx + gy*gy + 1e-12).sqrt()
    
    gm_o = gradmag(H_ref_tm.mean(1, True))
    gm_r = gradmag(ref_tm.mean(1, True))
    m = (1.0 - sat_mask).clamp(0, 1)  # exclude saturated ref
    return w * (m * (gm_o - gm_r).abs()).mean()

def ref_anti_copy(H_ref_tm, ref_tm, sat_mask, w=0.3):
    """
    Penalty for copying reference in saturated zones.
    Prevents the output from getting too close to ref in saturated areas.
    
    Args:
        H_ref_tm: Output HDR tonemapped [B,3,H,W]
        ref_tm: Reference view tonemapped [B,3,H,W]
        sat_mask: Saturation mask [B,1,H,W], 1=saturated
        w: Weight coefficient
    """
    # In saturated areas, penalize if output is too similar to ref
    # Use relu to only penalize when difference is very small (< 0.05)
    return w * (sat_mask * torch.relu(0.05 - (H_ref_tm - ref_tm).abs())).mean()

def tv_loss(x):
    """
    Total Variation loss to encourage smooth weight maps.
    
    Args:
        x: Weight tensor [B,M,N,1,H,W] or [B,K,H,W]
    """
    dx = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dy = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dx + dy

def entropy_loss(p, eps=1e-8):
    """
    Entropy regularization to prevent one-hot weight distributions.
    Encourages diversity in view selection.
    
    Args:
        p: Probability/weight tensor [B,M,N,1,H,W], should be normalized over N
        eps: Small constant for numerical stability
    """
    # Entropy over view dimension (dim=2)
    p_clamped = p.clamp_min(eps)
    return (-p_clamped * p_clamped.log()).sum(dim=2).mean()

def fft_ring_loss(x, r0=0.35, r1=0.5, w=0.1):
    """
    FFT-based loss to suppress checkerboard/grid artifacts.
    Penalizes high-frequency ring patterns in frequency domain.
    
    Args:
        x: Image tensor [B,3,H,W]
        r0, r1: Inner and outer radius of frequency ring to penalize
        w: Weight coefficient
    """
    X = torch.fft.rfft2(x, norm='ortho')
    mag = (X.real**2 + X.imag**2).sqrt()
    H, Wc = mag.shape[-2], mag.shape[-1]
    
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(0, 1, Wc, device=x.device),
        indexing='ij'
    )
    rr = (yy**2 + xx**2).sqrt()
    ring = ((rr > r0) & (rr < r1)).float()
    
    return w * (mag.mean(dim=1, keepdim=True) * ring).mean()

def laplacian_smoothness_loss(x, w=1e-4):
    """
    Laplacian smoothness regularization to suppress checkerboard artifacts.
    Penalizes high-frequency oscillations in residual outputs.
    
    Args:
        x: Tensor [B,C,H,W] - typically dz residuals from head_dz_list
        w: Weight coefficient (default 1e-4 to 5e-4)
    """
    # Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
    kernel = x.new_tensor([[0,1,0],[1,-4,1],[0,1,0]]).view(1,1,3,3)
    kernel = kernel.repeat(x.size(1), 1, 1, 1)
    lap = F.conv2d(x, kernel, padding=1, groups=x.size(1))
    return w * lap.abs().mean()

def ref_copy_penalty(output_tm, ref_tm, bad_mask, w=0.1):
    """
    Penalize output copying bad (saturated/dark) reference regions.
    Uses NCC (normalized cross-correlation) similarity metric.
    
    Args:
        output_tm: Output HDR tonemapped [B,3,H,W]
        ref_tm: Reference view tonemapped [B,3,H,W]  
        bad_mask: Bad region mask [B,1,H,W], 1=saturated or dark
        w: Weight coefficient
    """
    # Simple NCC proxy using 3x3 local windows
    mu = 3
    mO = F.avg_pool2d(output_tm, mu, 1, mu//2)
    mR = F.avg_pool2d(ref_tm, mu, 1, mu//2)
    sO = (F.avg_pool2d((output_tm - mO)**2, mu, 1, mu//2) + 1e-6).sqrt()
    sR = (F.avg_pool2d((ref_tm - mR)**2, mu, 1, mu//2) + 1e-6).sqrt()
    ncc = F.avg_pool2d((output_tm - mO) * (ref_tm - mR), mu, 1, mu//2) / (sO * sR + 1e-6)
    
    # Penalize positive correlation (similarity) in bad regions
    penalty = (bad_mask * ncc.clamp_min(0)).mean()
    return w * penalty

def bandpass_consistency_loss(output_tm, best_tm, w=0.1):
    """
    High-frequency consistency loss with best sharp view.
    Matches only high-frequency details, not overall brightness/color.
    
    Args:
        output_tm: Output HDR tonemapped [B,3,H,W]
        best_tm: Best sharp view tonemapped [B,3,H,W]
        w: Weight coefficient
    """
    # High-pass filter: image - lowpass
    def highpass(z):
        return z - F.avg_pool2d(z, 3, 1, 1)
    
    hp_out = highpass(output_tm)
    hp_best = highpass(best_tm)
    
    # Charbonnier L1 for robustness
    return w * charbonnier(hp_out - hp_best).mean()

def estimate_noise_tm(ref_tm, ksize=5):
    """
    Estimate noise level in tonemapped reference image.
    High local variance indicates noisy regions.
    
    Args:
        ref_tm: Reference tonemapped image [B,3,H,W]
        ksize: Kernel size for local statistics
    
    Returns:
        Noise weight map [B,1,H,W], higher = more noisy
    """
    gray = ref_tm.mean(1, keepdim=True)
    # Local variance as noise indicator
    var = local_variance(gray, kernel_size=ksize)
    # Normalize to [0,1] range
    noise_weight = var / (var.mean() + 1e-6)
    return noise_weight.clamp(0, 1)

def flat_noise_suppression(H_pred_tm, H_init, ref_tm, w=0.1):
    """
    Suppress high-frequency noise in flat regions when reference is noisy.
    Prevents noise transfer from reference to smooth areas.
    
    Args:
        H_pred_tm: Predicted HDR tonemapped [B,3,H,W]
        H_init: Initial HDR [B,3,H,W] for flat region detection
        ref_tm: Reference tonemapped [B,3,H,W] for noise estimation
        w: Weight coefficient
    
    Returns:
        Flat noise suppression loss
    """
    # Flat mask: regions with low edge magnitude in init
    edge_init = sobel_mag(tonemap_mu(H_init, 1e6).mean(1, True))
    flat = (edge_init < 0.02).float()
    
    # Estimate reference noise level
    ref_noise = estimate_noise_tm(ref_tm)
    
    # High-pass filter on prediction
    hp = H_pred_tm - F.avg_pool2d(H_pred_tm, 3, 1, 1)
    
    # Weight by flat regions and reference noise
    weight = flat * ref_noise
    
    return w * (hp.abs() * weight).mean()

def build_reliability(valid=None, conf=None, snr=None, lin=None, min_floor=None):
    # returns w_rel in [0,1], shape [B,1,H,W]
    w = None
    for t in [valid, conf, snr]:
        if t is not None:
            t1 = t
            if t1.dim()==5:  # [B,N,1,H,W] → 뷰 평균
                t1 = t1.mean(dim=1)
            if t1.size(1) != 1:  # [B,1,H,W] 로 맞춤
                t1 = t1.mean(dim=1, keepdim=True)
            w = t1 if w is None else (w * t1)
    if w is None:
        return None
    # 너무 어두운 영역 down-weight (센서 최소 무노이즈 밝기 제공 시)
    if (lin is not None) and (min_floor is not None):
        # min_floor: [B,N,1,1,1] 또는 [B,1,1,1]
        if min_floor.dim()==5: mf = min_floor.mean(dim=1)
        else: mf = min_floor
        if mf.dim()==4 and mf.size(1)==1:
            mask_dark = (lin.mean(dim=1, keepdim=True) < mf).float()
            w = w * (1.0 - 0.7*mask_dark)  # 매우 어두운 영역 0.3만 유지
    return w.clamp(0.0, 1.0)

@torch.no_grad()
def solve_gain_h(L, O, w=None, eps=1e-8):
    """
    h = argmin_h || w*(h*L - O) ||_2  → h = (Σ w L O) / (Σ w L^2)
    L,O: [B,3,H,W]; w: [B,1,H,W] or None
    returns h: [B,1,1,1]
    """
    if w is None:
        num = (L*O).sum(dim=(1,2,3), keepdim=True)
        den = (L*L).sum(dim=(1,2,3), keepdim=True) + eps
    else:
        w3 = w.expand_as(L)
        num = (w3*L*O).sum(dim=(1,2,3), keepdim=True)
        den = (w3*L*L).sum(dim=(1,2,3), keepdim=True) + eps
    h = (num / den).clamp(0.0, 1e6)
    return h

@torch.no_grad()
def sat_mask_tm(hdr, mu=1e6, thr=0.995, eps=1e-6):
    """
    Generate saturation mask in tonemap space.
    
    Args:
        hdr: HDR image [B,3,H,W]
        mu: Tonemap parameter for high-μ
        thr: Threshold for saturation (0.995 = top 0.5%)
        eps: Small constant
    
    Returns:
        Saturation mask [B,1,H,W], 1=saturated
    """
    tm = tonemap_mu(hdr, mu, eps)
    # Max across RGB channels
    tm_max = tm.max(dim=1, keepdim=True)[0]
    return (tm_max > thr).float()

@torch.no_grad()
def detail_mask_from_init(H_init, mu=1e6, k_edge=0.02, eps=1e-6):
    """
    Generate detail preservation mask from initial HDR.
    Identifies regions with fine structure that should be preserved.
    
    Args:
        H_init: Initial HDR reconstruction [B,3,H,W]
        mu: Tonemap parameter for high-μ
        k_edge: Edge threshold
        eps: Small constant
    
    Returns:
        Detail mask [B,1,H,W], 1=has detail
    """
    tm = tonemap_mu(H_init, mu, eps)
    # Compute edge magnitude
    edge_mag = sobel_mag(tm.mean(1, keepdim=True))
    return (edge_mag > k_edge).float()

def local_variance(x, kernel_size=5):
    """
    Compute local variance using average pooling.
    
    Args:
        x: Input tensor [B,C,H,W]
        kernel_size: Size of local window
    
    Returns:
        Local variance [B,C,H,W]
    """
    pad = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(x*x, kernel_size, stride=1, padding=pad)
    var = mean_sq - mean*mean
    return var.clamp_min(0.0)

def compute_gradient(x):
    """
    Compute image gradients (Sobel-style).
    
    Args:
        x: Input tensor [B,C,H,W]
    
    Returns:
        gx, gy: Gradient tensors [B,C,H,W]
    """
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return gx, gy

def camera_curve_gamma(x, gamma=2.2):
    """
    Simple gamma camera response curve.
    
    Args:
        x: Linear HDR values [B,C,H,W]
        gamma: Gamma value (default 2.2)
    
    Returns:
        LDR output [B,C,H,W]
    """
    return x.clamp(0.0, 1.0) ** (1.0 / gamma)

def reexposure_photometric_loss(
    H_pred, ldr_inputs, gamma=2.2, eps=1e-6
):
    """
    Re-exposure photometric loss.
    Renders HDR at different exposures and compares to observed LDR inputs.
    
    Args:
        H_pred: Predicted HDR [B,3,H,W]
        ldr_inputs: List of tuples (I_e, W_e, e) where
            I_e: Observed LDR image [B,3,H,W]
            W_e: Reliability weight [B,1,H,W]
            e: Exposure value (scalar or tensor [B,1,1,1])
        gamma: Camera gamma (default 2.2)
        eps: Small constant
    
    Returns:
        L_rexp: Re-exposure loss
    """
    L_rexp = 0.0
    count = 0
    
    for (I_e, W_e, e) in ldr_inputs:
        # Render at exposure e
        with torch.no_grad():
            if isinstance(e, (int, float)):
                e_tensor = torch.tensor(e, device=H_pred.device, dtype=H_pred.dtype)
            else:
                e_tensor = e
            
            # Expand e to match H_pred if needed
            if e_tensor.dim() == 0:
                e_tensor = e_tensor.view(1, 1, 1, 1)
        
        rendered = camera_curve_gamma(e_tensor * H_pred, gamma)
        
        # Weighted Charbonnier loss
        if W_e is not None:
            loss_e = charbonnier(W_e * (rendered - I_e), eps).mean()
        else:
            loss_e = charbonnier(rendered - I_e, eps).mean()
        
        L_rexp = L_rexp + loss_e
        count += 1
    
    if count > 0:
        L_rexp = L_rexp / count
    
    return L_rexp

def loss_detail_preserving(
    H_pred, H_init, GT_hdr,
    *,
    ldr_inputs=None,     # Optional: List of (I_e, W_e, e) for re-exposure
    mu_mid=5e4,
    mu_hi=1e6,
    sat_thr=0.995,
    edge_thr=0.02,
    # Loss weights
    lambda_gt=1.0,
    lambda_struct=0.7,
    lambda_edge=0.3,
    lambda_var=0.0,      # Optional, can be 0.05
    lambda_hinge=0.5,
    lambda_rexp=0.5,     # Re-exposure weight
    lambda_tv=3e-4,
    eps=1e-6
):
    """
    Detail-preserving loss for HDR reconstruction with highlight preservation.
    
    Core idea:
    - In reliable (non-saturated) regions: Standard GT supervision
    - In highlight regions: Preserve structure from H_init, prevent white-washing
    - Use inequality constraint to allow upward brightness but prevent downward
    - Re-exposure photometric loss for self-supervision from input LDR
    
    Args:
        H_pred: Predicted HDR output [B,3,H,W]
        H_init: Initial HDR reconstruction [B,3,H,W]
        GT_hdr: Ground truth HDR (may be clipped in highlights) [B,3,H,W]
        ldr_inputs: Optional list of (I_e, W_e, e) tuples for re-exposure loss
        mu_mid: Mid-μ for standard tonemap (5e4)
        mu_hi: High-μ for highlight tonemap (1e6)
        sat_thr: Saturation threshold (0.995)
        edge_thr: Edge detection threshold (0.02)
        lambda_*: Loss weights
        eps: Small constant for numerical stability
    
    Returns:
        total_loss: Combined loss value
        metrics: Dictionary of individual loss components
    """
    with autocast(enabled=False):
        H_pred = H_pred.float().clamp(eps, 20.0)
        H_init = H_init.float().clamp(eps, 20.0)
        GT_hdr = GT_hdr.float().clamp(eps, 20.0)
        
        # ---- Generate masks (no-grad) ----
        with torch.no_grad():
            # S: Saturation mask (GT is clipped)
            S = sat_mask_tm(GT_hdr, mu=mu_hi, thr=sat_thr, eps=eps)  # hard
            S_soft = feather_mask(S, k=9, iters=2)                   # soft ramp
            BAND = narrow_band(S_soft, 0.15, 0.85)                   # boundary narrow band
            
            # D: Detail mask (use soft mask for smoother boundary transition)
            D = S_soft * detail_mask_from_init(H_init, mu=mu_hi, k_edge=edge_thr, eps=eps)
        
        # ---- Tonemap for different μ values ----
        # Mid-μ for standard supervision
        Tm_pred = tonemap_mu(H_pred, mu_mid, eps)
        Tm_gt = tonemap_mu(GT_hdr, mu_mid, eps)
        
        # High-μ for highlight structure
        Th_pred = tonemap_mu(H_pred, mu_hi, eps)
        Th_init = tonemap_mu(H_init, mu_hi, eps)
        
        # ---- (A) GT loss: Standard supervision in reliable regions ----
        L_gt = charbonnier((1.0 - S_soft) * (Tm_pred - Tm_gt), eps).mean()
        
        # ---- (B) Structure preservation in highlights ----
        # Structure match (high-μ tonemap space)
        L_struct = charbonnier(D * (Th_pred - Th_init), eps).mean()
        
        # Edge preservation
        gx_pred, gy_pred = compute_gradient(Th_pred)
        gx_init, gy_init = compute_gradient(Th_init)
        
        L_edge = (
            charbonnier(D * (gx_pred - gx_init), eps).mean() +
            charbonnier(D * (gy_pred - gy_init), eps).mean()
        )
        
        # ---- (B-optional) Local variance lower bound (prevent washing-out) ----
        L_var = 0.0
        if lambda_var > 0:
            var_pred = local_variance(Th_pred, kernel_size=5)
            with torch.no_grad():
                var_init = local_variance(Th_init, kernel_size=5)
            
            # Penalize when variance drops below init variance
            L_var = (D * F.relu(var_init - var_pred)).mean()
        
        # ---- (C) Inequality-aware hinge loss ----
        # Only penalize when prediction is DARKER than clipped GT
        # Allow upward brightness to preserve highlights
        L_hinge = charbonnier(S_soft * F.relu(Tm_gt - Tm_pred), eps).mean()
        
        # ---- (C-NEW) Boundary smoothing loss ----
        # Prevent sharp "cut" artifacts at saturation boundaries
        L_band_smooth = band_grad_smooth_loss(Tm_pred, BAND, w=5e-4)
        
        # ---- (D) Re-exposure photometric loss ----
        L_rexp = 0.0
        if ldr_inputs is not None and len(ldr_inputs) > 0:
            L_rexp = reexposure_photometric_loss(H_pred, ldr_inputs, eps=eps)
        
        # ---- (E) Total Variation regularization ----
        tv_x = (H_pred[..., :, 1:] - H_pred[..., :, :-1]).abs().mean()
        tv_y = (H_pred[..., 1:, :] - H_pred[..., :-1, :]).abs().mean()
        L_tv = tv_x + tv_y
        
        # ---- Combine losses ----
        total_loss = (
            lambda_gt * L_gt +
            lambda_struct * L_struct +
            lambda_edge * L_edge +
            lambda_var * L_var +
            lambda_hinge * L_hinge +
            lambda_rexp * L_rexp +
            lambda_tv * L_tv +
            L_band_smooth
        )
        
        # Metrics for logging
        metrics = {
            'L_gt': float(L_gt),
            'L_struct': float(L_struct),
            'L_edge': float(L_edge),
            'L_var': float(L_var) if lambda_var > 0 else 0.0,
            'L_hinge': float(L_hinge),
            'L_rexp': float(L_rexp),
            'L_tv': float(L_tv),
            'L_band_smooth': float(L_band_smooth),
            'sat_ratio': float(S.mean()),
            'detail_ratio': float(D.mean()),
        }
        
        return total_loss, metrics

def hdr_criterion_evidence_lin(
    output, target,
    *,
    # stacks are in linear domain, already warped to the output view
    linear_stack=None,   # [B,N,3,H,W] or [B,M,N,3,H,W]
    valid=None,          # [B,N,1,H,W] or [B,M,N,1,H,W]
    confidence=None,     # same shape as valid
    snr=None,            # same shape as valid
    min_int_floor=None,  # [B,N,1,1,1] per-view 최소 무노이즈 밝기(있으면 좋음)
    ref_view_rgb=None,   # [B,M,3,H,W] reference view for anti-copy losses
    # Detail-preserving args
    H_init=None,         # Initial HDR reconstruction [B,M,3,H,W]
    use_detail_loss=False,  # Enable detail-preserving loss
    mu_levels=(1e3, 5e4, 1e6),
    w_asym=0.5,          # evidence-aware 비대칭 가중
    delta_hi=0.10,       # 부족 GT 판정 마진(10%)
    lambda_hi=0.15,      # 부족 GT일 때 over-penalty 축소율
    w_reproj=1.0,        # 뷰 재투영 가중
    w_silog=0.2,         # scale-invariant log
    w_tm=1.0, w_tm_ssim=2.0,
    w_hdr=0.1, w_grad=3.0,
    w_ref_edge=0.2,      # reference edge consistency weight
    w_ref_anticopy=0.3,  # reference anti-copy weight
    # Detail-preserving loss weights
    w_detail_gt=1.0,
    w_detail_struct=0.7,
    w_detail_edge=0.3,
    w_detail_var=0.0,
    w_detail_hinge=0.5,
    w_detail_tv=3e-4,
):
    with autocast(enabled=False):
        O = output.float(); G = target.float()
        if O.dim()==5:
            B,N,C,H,W = O.shape
            O = O.reshape(B*N,C,H,W); G = G.reshape(B*N,C,H,W)
        O = O.clamp(1e-12, 20.0); G = G.clamp(1e-12, 20.0)

        # ---- 기본(tonemap, multi-μ, grad, si-log) ----
        mu_mid = mu_levels[len(mu_levels)//2]
        O_tm = tonemap_mu(O, mu_mid); G_tm = tonemap_mu(G, mu_mid)
        l_tm = F.l1_loss(O_tm, G_tm)
        # 간단 SSIM 대체: 권장하던 ms-ssim을 그대로 넣어 쓰세요
        #ssim = 1.0 - (O_tm - G_tm).abs().mean()  # placeholder
        ssim = 1 - ms_ssim(O_tm, G_tm, data_range=1.0).mean()
        O_md = torch.cat([tonemap_mu(O * (50 ** i),m) for m in mu_levels for i in range(2)], dim=1)
        G_md = torch.cat([tonemap_mu(G * (50 ** i),m) for m in mu_levels for i in range(2)], dim=1)
        l_md = F.l1_loss(O_md, G_md)
        gx,gy = sobel_mag(O_md), sobel_mag(G_md)
        l_grad = (gx-gy).abs().mean()
        l_hdr = F.l1_loss(O, G)
        l_silog = si_log_rmse(O, G)
        ssim_md = 1 - ms_ssim(O_md, G_md, data_range=1.0).mean()

        loss = w_tm*l_tm + w_tm_ssim*(ssim + ssim_md) + w_hdr*l_hdr + 0.5*l_md + w_grad*l_grad + w_silog*l_silog
        metrics = dict(l_tm=float(l_tm), ssim=float(ssim), l_md=float(l_md), l_grad=float(l_grad), l_hdr=float(l_hdr), l_silog=float(l_silog), ssim_md=float(ssim_md))

        # ---- evidence-aware (비대칭) + reprojection ----
        if linear_stack is not None:
            Ls = linear_stack.float()
            if Ls.dim()==6:   # [B,M,N,3,H,W] → [B*M,N,3,H,W]
                B_,M_,N_,C_,H_,W_ = Ls.shape
                Ls = Ls.reshape(B_*M_, N_, C_, H_, W_)
                if valid is not None:      valid      = valid.reshape(B_*M_, N_, 1, H_, W_)
                if confidence is not None: confidence = confidence.reshape(B_*M_, N_, 1, H_, W_)
                if snr is not None:        snr        = snr.reshape(B_*M_, N_, 1, H_, W_)
                if min_int_floor is not None and min_int_floor.dim()==6:
                    min_int_floor = min_int_floor.reshape(B_*M_, N_, 1, 1, 1)
            Bv, Nv, _, H_, W_ = Ls.shape
            assert O.shape[-2:]==(H_,W_), "spatial mismatch"

            # per-view reliability
            wv = None
            if (valid is not None) or (confidence is not None) or (snr is not None):
                wv = build_reliability(valid, confidence, snr, lin=Ls.mean(2), min_floor=min_int_floor)  # [B*,N,1,H,W] or None

            # ---- per-view gain h_i (h L_i ≈ O)
            O_rep = O.unsqueeze(1)  # [B*,1,3,H,W]
            h_list, reproj_list = [], []
            for i in range(Nv):
                Li = Ls[:, i, :, :, :]  # [B*,3,H,W]
                wi = None if wv is None else wv[:, i, :, :, :]
                hi = solve_gain_h(Li, O, wi)  # [B*,1,1,1]
                h_list.append(hi)
                reproj = (hi * Li)  # [B*,3,H,W]
                if wi is None: reproj_list.append((reproj - O).abs().mean())
                else:
                    w3 = wi.expand_as(Li)
                    reproj_list.append((w3*(reproj - O).abs()).sum() / (w3.sum()+1e-8))
            l_reproj = torch.stack(reproj_list).mean()
            loss = loss + w_reproj * l_reproj
            metrics["l_reproj"] = float(l_reproj)

            # ---- evidence upper bound E_max = max_i (h_i * L_i)
            with torch.no_grad():
                Emax = []
                for i in range(Nv):
                    Li = Ls[:, i, :, :, :]
                    Emax.append((h_list[i] * Li))
                Emax = torch.stack(Emax, dim=1).amax(dim=1)  # [B*,3,H,W]

            # 부족 GT 마스크: Emax > (1+δ)*G  (GT가 DR 부족해 보이는 곳)
            mask_insuf = (Emax > (1.0 + delta_hi)*G).float()
            over = (O - G).clamp_min(0.0); under = (G - O).clamp_min(0.0)
            # over-penalty를 λ_hi로 약화
            l_asym = (lambda_hi*mask_insuf*over + (1.0 - mask_insuf)*over + under).mean()
            loss = loss + w_asym * l_asym
            metrics.update(l_asym=float(l_asym), hi_ratio=float(mask_insuf.mean()))

        # ---- Anti-copy losses (if reference view provided) ----
        if ref_view_rgb is not None:
            # Reshape ref_view_rgb to match output dimensions if needed
            ref_rgb = ref_view_rgb.float()
            if ref_rgb.dim() == 5:  # [B,M,3,H,W]
                B_, M_, C_, H_, W_ = ref_rgb.shape
                ref_rgb = ref_rgb.reshape(B_*M_, C_, H_, W_)
            
            # Ensure shapes match
            if ref_rgb.shape != O.shape:
                # Resize if needed
                ref_rgb = F.interpolate(ref_rgb, size=O.shape[-2:], mode='bilinear', align_corners=True)
            
            # Tonemap output and reference
            H_tm = tonemap_mu(O, mu_mid)
            ref_tm = tonemap_mu(ref_rgb.clamp(1e-12, 20.0), mu_mid)
            
            # Saturation mask: pixels where reference is saturated
            sat_hard = (ref_rgb.max(1, keepdim=True)[0] > 0.99).float()
            sat_soft = feather_mask(sat_hard, k=7, iters=1)  # smooth boundary
            
            # Add anti-copy losses with soft mask
            l_ref_edge = ref_edge_consistency(H_tm, ref_tm, sat_soft, w=w_ref_edge)
            l_ref_anticopy = ref_anti_copy(H_tm, ref_tm, sat_soft, w=w_ref_anticopy)
            
            loss = loss + l_ref_edge + l_ref_anticopy
            
            # NEW: Flat noise suppression when reference is noisy
            # Use H_init_data if available, otherwise use current output (detached)
            H_init_for_flat = None
            if use_detail_loss and H_init is not None:
                H_init_for_flat = H_init_data
            else:
                H_init_for_flat = O.detach()
            
            l_flat_noise = flat_noise_suppression(H_tm, H_init_for_flat, ref_tm, w=0.08)
            loss = loss + l_flat_noise
            
            metrics.update(
                l_ref_edge=float(l_ref_edge),
                l_ref_anticopy=float(l_ref_anticopy),
                l_flat_noise=float(l_flat_noise),
                sat_ratio=float(sat_soft.mean())
            )

        # ---- Detail-preserving loss (if enabled and H_init provided) ----
        if use_detail_loss and H_init is not None:
            # Prepare H_init
            H_init_data = H_init.float()
            if H_init_data.dim() == 5:  # [B,M,3,H,W]
                B_, M_, C_, H_, W_ = H_init_data.shape
                H_init_data = H_init_data.reshape(B_*M_, C_, H_, W_)
            
            # Ensure shapes match
            if H_init_data.shape != O.shape:
                H_init_data = F.interpolate(H_init_data, size=O.shape[-2:], mode='bilinear', align_corners=True)
            
            # Prepare LDR inputs for re-exposure (optional)
            # In practice, you would build this from linear_stack with different exposures
            # For now, we'll skip re-exposure in the integrated version
            ldr_inputs_for_detail = None
            
            # Call detail-preserving loss
            l_detail, detail_metrics = loss_detail_preserving(
                H_pred=O,
                H_init=H_init_data,
                GT_hdr=G,
                ldr_inputs=ldr_inputs_for_detail,
                mu_mid=mu_levels[len(mu_levels)//2],
                mu_hi=mu_levels[-1] if len(mu_levels) > 2 else 1e6,
                lambda_gt=w_detail_gt,
                lambda_struct=w_detail_struct,
                lambda_edge=w_detail_edge,
                lambda_var=w_detail_var,
                lambda_hinge=w_detail_hinge,
                lambda_rexp=0.0,  # Set to 0 for now, can be enabled separately
                lambda_tv=w_detail_tv,
            )
            
            loss = loss + l_detail
            # Add detail metrics with prefix
            for k, v in detail_metrics.items():
                metrics[f'detail_{k}'] = v

        return loss, metrics
