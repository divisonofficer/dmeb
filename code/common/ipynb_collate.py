"""Collate helpers extracted from /jarvis/view/ipynb/hdr/comparison_hdr.ipynb cell 1.

Provides:
- hdr2ldr(img, exp_hdr, target_intensity, noise_std)
- collate_batch_input(frame, exp_adjust, with_gt, tri, hdr_adjust, near)
    -> (rgbs, prompts, sats, Ks_in, K_inv_in, Ts_in, Ks_tgt, Ts_tgt, ldr_min_gts, tgt_rgbs)
- collate_batch_tri(frame, ...) for baseline (HDRT/AFUNet/HDRFlow) inputs
- psnr / ssim / tonemap_up
"""
import torch
import cv2
def hdr2ldr(img, exp_hdr,target_intensity=[0.1, 0.25], noise_std = 0.003):
    while img.clip(0,1).mean() > target_intensity[1]:
        img = img * 0.5
        exp_hdr = exp_hdr * 0.5
    while img.clip(0,1).mean() < target_intensity[0]:
        img = img * 2.0
        exp_hdr = exp_hdr * 2.0
    img += torch.randn_like(img) * noise_std
    img = img.clamp(0,1)
    return img, exp_hdr
def collate_batch_input(frame, exp_adjust = 1000.0, with_gt=False, tri=False, hdr_adjust = 4920.762683082901, near=False):
    rgbs = []
    prompts = []
    sats = []
    Ks_in = []
    Ts_in = []
    Ks_tgt = []
    Ts_tgt = []
    ldr_min_gts = []
    tgt_rgbs = []
    input_sides = ["12_rear","12_left","12_right","12_rear_sub","12_left_sub","12_right_sub"]
    if tri:
        #find min exp side and max exp side
        exp_list = []
        for side in input_sides:
            exp_list.append(frame[f"exposure_{side}"])
        min_exp = min(exp_list)
        max_exp = max(exp_list)
        if near:
            min_exp = sorted(exp_list)[len(exp_list)//2 -1]
            max_exp = sorted(exp_list)[len(exp_list)//2]
        median_exp = sorted(exp_list)[len(exp_list)//2]
        input_sides = []
        for side in ["12_rear","12_left","12_right","12_rear_sub","12_left_sub","12_right_sub"]:
            exp = frame[f"exposure_{side}"]
            if exp == min_exp or exp == max_exp:
                input_sides.append(side)
            if exp == median_exp and not with_gt:
                input_sides.append(side)
        input_sides *= 2
    if with_gt:
        input_sides =  input_sides[:3] + ["left"]  + input_sides[3:]
    if not near:
        input_sides += ["left","right"]
    for s_i, side in enumerate(input_sides):
        rgb = (frame[f"rgb_{side}"].unsqueeze(0).cuda() * 1.0)
        
        if not "12" in side:
            with torch.no_grad():
                # ensure float tensors
                exp_key_base = f"exposure_{side}"
                base_key = f"rgb_{side}"
                
                exp_hdr = frame[exp_key_base] / exp_adjust
                hdr_input = frame[base_key].float()
                hdr_raw = hdr_input.clone()  * hdr_adjust
        
                    
      
            img1_noisy_ldr, exp_hdr = hdr2ldr(hdr_raw, exp_hdr,
                                              target_intensity=[0.001, 0.005] if s_i %2 == 0 else [0.3, 0.45], noise_std=0.001
                                              )
        
            img1_noisy = torch.concat([img1_noisy_ldr, img1_noisy_ldr / exp_hdr], dim=0).unsqueeze(0)
            rgb = img1_noisy.clamp(0,1).cuda()
            
            sat = ((img1_noisy_ldr > 0.005) & (img1_noisy_ldr < 0.99) & (hdr_input < 0.99)).float().cuda()
            
            sat = torch.max(sat, dim=0, keepdim=True)[0].unsqueeze(0)
            
        else:
            exp = frame[f"exposure_{side}"] / exp_adjust
            sat = ((rgb > 0.03) & (rgb < 0.99)).float().cuda()
            rgb = torch.concat([rgb, rgb / exp ], dim=1)
            
            sat = torch.max(sat, dim=1, keepdim=True)[0]
       
            
        
        ldr_min_gt = frame[f"min_dn_{side}"] *0.1
        if ldr_min_gt > 0.05:
            ldr_min_gt = 0.05 + (ldr_min_gt - 0.05) ** 2
        if not "12" in side:
            ldr_min_gt = 0.001
        ldr_min_gts.append(torch.tensor(ldr_min_gt).cuda())
        

        
        rgbs.append(rgb)
        prompt = frame[f"lidar_{side}"].unsqueeze(0).cuda()
        #prompt = torch.ones_like(prompt) * 1---
        prompts.append(prompt)
        
        K_in = frame[f"K_{side}"].unsqueeze(0).cuda()
        Ks_in.append(K_in)
        T_in = frame[f"E_{side}"].unsqueeze(0).cuda()
        Ts_in.append(T_in)
        
        sats.append(sat)
    for side in ["left","right"]:
        K_tgt = frame[f"K_{side}"].unsqueeze(0).cuda()
        
        Ks_tgt.append(K_tgt)
        T_tgt = frame[f"E_{side}"].unsqueeze(0).cuda()

        Ts_tgt.append(T_tgt)
        
        tgt_rgb = (frame[f"rgb_{side}"].unsqueeze(0).cuda()) * hdr_adjust
        tgt_exp = frame[f"exposure_{side}"] / exp_adjust
        tgt_rgb = torch.concat([tgt_rgb, tgt_rgb / tgt_exp ], dim=1)
        tgt_rgbs.append(tgt_rgb)
        
    rgbs = torch.cat(rgbs, dim=0).unsqueeze(0)
    adjust_ratio = 1000.0
    prompts = torch.cat(prompts, dim=0).unsqueeze(0) / adjust_ratio
    sats = torch.cat(sats, dim=0).unsqueeze(0)
    Ks_in = torch.cat(Ks_in, dim=0).unsqueeze(0)
    Ts_in = torch.cat(Ts_in, dim=0).unsqueeze(0)
    Ts_in[..., :3, 3] = Ts_in[..., :3, 3]
    Ks_tgt = torch.cat(Ks_tgt, dim=0).unsqueeze(0)
    Ts_tgt = torch.cat(Ts_tgt, dim=0).unsqueeze(0)
    Ts_tgt[..., :3, 3] = Ts_tgt[..., :3, 3]
    ldr_min_gts = torch.stack(ldr_min_gts, dim=0).unsqueeze(0)  # [B=1, N]
    tgt_rgbs = torch.cat(tgt_rgbs, dim=0).unsqueeze(0)
    
    return (rgbs, prompts, sats, Ks_in, Ks_in, Ts_in, Ks_tgt, Ts_tgt, ldr_min_gts, tgt_rgbs) # dummy for k_inv_in

import cv2
import numpy as np

def adjust_luminance(img1, img2):
    img2_max = np.percentile(img2, 99.9)
    img2_min = np.percentile(img2, 0.1)
    img1_adjust = img1.clip(img2_min, img2_max)
    img2 = img2.clip(img2_min, img2_max)
    mu = 1e5
    img1_adjust =np.log(1 + mu * img1_adjust) / np.log(1 + mu)
    img2 = np.log(1 + mu * img2) / np.log(1 + mu)
    return img1_adjust, img2

def psnr(img1, img2):
    img1, img2 = adjust_luminance(img1, img2)
    mse = ((img1 - img2) ** 2).mean()
    
    
    # max val은 img1, img2 픽셀들 중 상위 0.01% 픽셀값으로 설정
    max_val1 = np.percentile(img1, 99.9)
    max_val2 = np.percentile(img2, 99.9)
    max_val = max(max_val1, max_val2)
    
    psnr = 10 * np.log10(max_val / mse)
    return psnr


def ssim(img1, img2):
    img1, img2 = adjust_luminance(img1, img2)
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    C1 = 0.01**2
    C2 = 0.03**2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = (
        (2 * mu1_mu2 + C1)
        * (2 * sigma12 + C2)
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )
    return ssim_map.mean()
tonemap_reinhard = cv2.createTonemapReinhard(gamma=1.8, light_adapt=0)
from utils.hdr.mertens_torch import mertens_merge_differentiable
def tonemap_up(img, mu=1e6, mode="mu"):
    # img_tm = mertens_merge_differentiable(
    #     img, num_exposures=32, step=1.4, gamma=1.0
    # )[0]
    # return img_tm.permute(1,2,0).cpu().numpy().clip(0,1)
    # 
    if mode == "mertens":
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2,0,1).cuda()
        img = mertens_merge_differentiable(img.unsqueeze(0), num_exposures=10, step=2.0)[0]
        return img.permute(1,2,0).cpu().numpy().clip(0,1)
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().permute(1,2,0).numpy()
    #img_tm = tonemap_reinhard.process(img.astype(np.float32))
    img_tm = np.log(1 + mu * img) / np.log(1 + mu)
    return img_tm.clip(0,1)


def compute_dr(image: torch.Tensor):
    image_gray = 0.299 * image[:,0,:,:] + 0.587 * image[:,1,:,:] + 0.114 * image[:,2,:,:]
    dr = compute_dr_for_channel(image_gray[0].numpy(), perc=99.995)
    dr_report = {
        "DR(dB)" : f"{dr['DR_dB']:.2f}",
        "Lmin" : f"{dr['Lmin']:.10f}",
        "Lmax" : f"{dr['Lmax']:.10f}"
    }
    return dr_report


import torch



def collate_batch_tri(frame, exp_adjust = 1000.0, hdr_adjust = 815.5391462047698, frame_idx=None, single_cam=False, dataset=None, near=False):
    """
    Return:
      - tri_imgs: torch.Tensor, shape (2, 3, 3, H, W)  # (side(left/right), img0/img1/img2, C, H, W)
      - gt_hdrs:  torch.Tensor, shape (2, 3, H, W)      # clean HDR (no noise) per side
    Notes:
      - img0: LDR with shortest exposure among rgb_* candidates for the side
      - img2: LDR with longest exposure among rgb_* candidates for the side
      - img1: simulated LDR made from HDR (see below)
      - HDR (gt) is computed from a chosen base rgb (prefer f"rgb_{side}" if present) as rgb / exposure * hdr_adjust
      - img1 simulation: clip(hdr, max=hdr.max()/10) then add gaussian noise with std = hdr.max()/100
    """
    sides = ["left", "right"] 
    tri_list, gt_list = [], []





    if single_cam:
        if dataset is None or frame_idx is None:
            raise ValueError("single_cam=True requires dataset and frame_idx.")
        
        # 앞, 현재, 뒤 프레임 가져오기
        prev_frame = dataset[frame_idx - 1]
        next_frame = dataset[frame_idx + 1]
        
        for side in sides:
            base_key = f"rgb_{side}"
            exp_key = f"exposure_{side}"
            if base_key not in frame:
                raise RuntimeError(f"No base HDR found for {side}")

            # base HDR radiance
            exp_hdr = frame[exp_key] / exp_adjust
            hdr_raw = frame[base_key].float().clamp(0,1 - 1e-3) / (exp_hdr + 1e-12)
            hdr_raw = hdr_raw * hdr_adjust  # radiance scale

            hdr_prev = prev_frame[base_key].float().clamp(0,1) / (prev_frame[exp_key]/exp_adjust)
            hdr_next = next_frame[base_key].float().clamp(0,1) / (next_frame[exp_key]/exp_adjust)
            img0 = hdr_prev
            img2 = hdr_next
            # mid (current)
            img1 = hdr_raw.clone()
            # bright (next) → saturated


            # noise 추가
            img0_ldr, img0_exp = hdr2ldr(img0.clone(), prev_frame[exp_key]/exp_adjust, target_intensity=[0.002, 0.005])
            img1_ldr, img1_exp = hdr2ldr(img1.clone(), frame[exp_key]/exp_adjust)
            img2_ldr, img2_exp = hdr2ldr(img2.clone(), next_frame[exp_key]/exp_adjust, target_intensity=[0.5, 0.7])

            # exposure normalization channel 추가
            img0 = torch.concat([img0_ldr, img0_ldr / img0_exp], dim=0)
            img1 = torch.concat([img1_ldr, img1_ldr / img1_exp], dim=0)
            img2 = torch.concat([img2_ldr, img2_ldr / img2_exp], dim=0)

            tri = torch.stack([img0, img1, img2], dim=0)
            gt = hdr_raw.float()

            tri_list.append(tri)
            gt_list.append(gt)

        tri_imgs = torch.stack(tri_list, dim=0)
        gt_hdrs = torch.stack(gt_list, dim=0).clamp(0,1)
        return tri_imgs, gt_hdrs
    
    
    # collect candidate rgb keys that contain the side
    candidates = []
    for k in frame.keys():
        if k.startswith("rgb_") and "12" in k:
            suffix = k[len("rgb_"):]  # e.g. "12_left" or "left"
            exp_key = f"exposure_{suffix}"
            if exp_key in frame:
                try:
                    exp_val = float(frame[exp_key].item()) if hasattr(frame[exp_key], "item") else float(frame[exp_key])
                except Exception:
                    # fallback if exposure is tensor-like with shape ()
                    exp_val = float(torch.as_tensor(frame[exp_key]).item())
                candidates.append((exp_val, k, suffix))
    # require at least one candidate
    if len(candidates) == 0:
        raise RuntimeError(f"No rgb/exposure candidates found for side '{side}' in frame keys.")
    # pick shortest and longest exposure
    candidates.sort(key=lambda x: x[0])
    if near == False:
        exp_min, key_min, suf_min = candidates[0]
        exp_max, key_max, suf_max = candidates[-1]
    else:
        # i, i+1 = len(candidates)//2 -1, len(candidates)//2
        exp_min, key_min, suf_min = candidates[len(candidates)//2 -1]
        exp_max, key_max, suf_max = candidates[len(candidates)//2]
    for side in sides:
        
      
        img0 = frame[key_min].clone().clamp(0,1)  # LDR shortest
        img2 = frame[key_max].clone().clamp(0,1)  # LDR longest

        # choose base for HDR computation: prefer exact "rgb_{side}" if available, else pick median candidate
        base_key = f"rgb_{side}" if f"rgb_{side}" in frame else candidates[len(candidates)//2][1]
        base_suffix = base_key[len("rgb_"):]
        exp_key_base = f"exposure_{base_suffix}"
        if exp_key_base not in frame:
            # fallback to first candidate's exposure suffix
            exp_key_base = f"exposure_{candidates[0][2]}"

        # compute HDR radiance (gt)
        with torch.no_grad():
            # ensure float tensors
            exp_hdr = frame[exp_key_base] / exp_adjust
            hdr_raw = frame[base_key].float() / (exp_hdr + 1e-12)
            hdr_raw = hdr_raw * hdr_adjust  # scale to desired radiance units

            # simulate img1 from hdr_raw
            hdr_max = torch.quantile(hdr_raw.view(-1), 0.999)
            clip_val = hdr_max if hdr_max > 0 else 0.0
            if clip_val > 0:
                img1 = torch.clamp(hdr_raw, max=clip_val)
            else:
                img1 = hdr_raw.clone()
            
            img1_noisy = img1.clone()
     
                
        # ensure same device / dtype
        img0 = img0.float()
        img0 = torch.concat([img0, img0 / (exp_min / exp_adjust + 1e-12)], dim=0)  # add exposure-normalized channel
        img1_noisy = img1_noisy.float()  
        img1_noisy_ldr = (img1_noisy.clone() * exp_hdr)
        img1_noisy_ldr, exp_hdr = hdr2ldr(img1_noisy_ldr, exp_hdr)
        img1_noisy = torch.concat([img1_noisy_ldr, img1_noisy_ldr / exp_hdr], dim=0)
        img2 = img2.float()
        img2 = torch.concat([img2, img2 / (exp_max / exp_adjust + 1e-12)], dim=0)

        # stack as (3, C, H, W)
        tri = torch.stack([img0, img1_noisy, img2], dim=0)  # (3, C, H, W)
        gt = hdr_raw.float()  # (C, H, W)  -- clean HDR (no noise)

        tri_list.append(tri)
        gt_list.append(gt)

    # final tensors: (2, 3, C, H, W) and (2, C, H, W)
    tri_imgs = torch.stack(tri_list, dim=0)
    gt_hdrs = torch.stack(gt_list, dim=0)

    return tri_imgs, gt_hdrs.clamp(0,1)
