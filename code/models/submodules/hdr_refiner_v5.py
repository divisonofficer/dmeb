import math, torch
import torch.nn as nn
import torch.nn.functional as F
mu_list = [1e3, 5e4, 1e6]

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

def _build_feats(H_init, ldr, lin, conf, depth, snr):
    # multi-μ & gray (중간 μ)
    B,M,N,_,H,W = ldr.shape
    Hexp = H_init.unsqueeze(2).expand(B,M,N,3,H,W)
    H_Ts = [tonemap_mu(Hexp, m) for m in mu_list]
    H_Tc = torch.cat(H_Ts, dim=3)
    H_mid= tonemap_mu(Hexp, mu_list[1])
    Hgry = H_mid.mean(3, keepdim=True)
    feats = torch.cat([ldr, lin, conf, depth, snr, H_Tc, Hgry], dim=3)  # [B,M,N,Cf,H,W]
    # Top-K view 선택 (scores: conf*snr*valid 근사)

    with torch.no_grad():
        q = (conf * snr).squeeze(3).mean(dim=(-1,-2))        # [B,M,N]
        Hexp_mid = tonemap_mu(H_init.unsqueeze(2).expand(B,M,N,3,H,W), mu_list[1])
        Y = (0.2126*Hexp_mid[:,:,:,0] + 0.7152*Hexp_mid[:,:,:,1] + 0.0722*Hexp_mid[:,:,:,2])  # [B,M,N,H,W]
        e = Y.median(dim=-1).values.median(dim=-1).values    # [B,M,N]
        sat_hi = (Y > 0.995).float().mean(dim=(-1,-2))       # [B,M,N]
        sat_lo = (Y < 0.005).float().mean(dim=(-1,-2))       # [B,M,N]

        sat_hi_thr, sat_lo_thr = 0.50, 0.50
        valid = (sat_hi < sat_hi_thr) & (sat_lo < sat_lo_thr)  # [B,M,N]

        topk = 3
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



class HDRRefinerV5(nn.Module):

    def __init__(self, afunet):
        super().__init__()
        self.afunet = afunet
        
        # Intermediate refinement module: 57 channels -> 18 channels
        self.channel_refiner = nn.Sequential(
            nn.Conv2d(57, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 18, kernel_size=3, padding=1)
        )
        
        # Ref view RGB processing module: 3 channels -> 18 channels
        # Always executed to ensure gradient flow
        self.ref_view_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 18, kernel_size=3, padding=1)
        )
        
        
    def forward(self, H_init, ldr_warped, linear_warped,
                valid_mask=None, depth_warped=None, confidence=None, snr_norm=None,
                project_min_bound=None,
                ref_view_rgb=None):
        
        # build features
        feats = _build_feats(H_init, ldr_warped, linear_warped,
                             confidence, depth_warped, snr_norm)  # [B,M,K*Cf,H,W]
        
        B, M = feats.shape[:2]
        
        # Process ref_view_rgb to ensure gradient flow
        # Always execute the module, even if ref_view_rgb is None
        if ref_view_rgb is not None:
            ref_input = ref_view_rgb  # [B, M, 3, H, W]
            use_ref = True
        else:
            # Create dummy input with same shape to ensure gradient flow
            ref_input = torch.zeros(B, M, 3, feats.shape[-2], feats.shape[-1], 
                                   device=feats.device, dtype=feats.dtype)
            use_ref = False
        
        # Process each camera view separately
        outputs = []
        for m in range(M):
            feat_m = feats[:, m]  # [B, K*Cf, H, W] = [B, 57, H, W]
            ref_m = ref_input[:, m]  # [B, 3, H, W]
            
            # Refine 57 channels -> 18 channels
            # Use autocast for the refinement module (works with float16)
            with torch.cuda.amp.autocast(enabled=True):
                refined_feat = self.channel_refiner(feat_m)  # [B, 18, H, W]
                
                # Always process ref_view through the module (gradient flow)
                ref_feat = self.ref_view_processor(ref_m)  # [B, 18, H, W]
                
                # Add ref_feat only if ref_view_rgb was provided
                if use_ref:
                    refined_feat = refined_feat + ref_feat
                else:
                    # Multiply by 0 to keep gradient flow but ignore result
                    refined_feat = refined_feat + ref_feat * 0.0
            
            # Pass through afunet (expects 18 channels)
            # AFUNet also works with autocast
            with torch.cuda.amp.autocast(enabled=True):
                hdr_tonemapped = self.afunet(refined_feat)  # [B, 3, H, W] (tonemapped)
            
            # Inverse tonemap to get linear HDR (disable autocast for linear HDR)
            hdr_linear = inv_tonemap_mu(hdr_tonemapped, mu=mu_list[1])  # [B, 3, H, W]
            
            outputs.append(hdr_linear)
        
        # Stack results for all camera views
        result = torch.stack(outputs, dim=1)  # [B, M, 3, H, W]
        
        return result
        