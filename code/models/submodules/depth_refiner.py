
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthResidualRefinerUNet(nn.Module):
    """
    Lightweight residual depth refiner for post-merge cleanup.

    The module is deliberately identity-initialized: the final prediction layer
    starts at zero, so an untrained refiner returns D_fused unchanged. This keeps
    older checkpoints behavior-compatible while allowing the refiner to learn
    only where merge weights are uncertain or all views are weak.
    """

    def __init__(
        self,
        base=16,
        max_delta_inv=0.10,
        internal_downsample=1,
        eps=1e-6,
    ):
        super().__init__()
        self.max_delta_inv = float(max_delta_inv)
        self.internal_downsample = max(1, int(internal_downsample))
        self.eps = eps

        in_ch = 7
        self.in_norm = nn.InstanceNorm2d(1, affine=False)

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.down1 = nn.Conv2d(base, base * 2, 3, 2, 1)
        self.enc2 = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, 2, 1)
        self.mid = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4 + base * 2, base * 2, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2 + base, base, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(base, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _resize(self, x, size, mode="bilinear"):
        if x.shape[-2:] == size:
            return x
        if mode == "nearest":
            return F.interpolate(x, size=size, mode=mode)
        return F.interpolate(x, size=size, mode=mode, align_corners=False)

    def forward(self, D_fused, D_warp, w_norm, rerr, w_strength=None):
        """
        Args:
            D_fused:   [B,M,1,H,W]
            D_warp:    [B,M,N,1,H,W]
            w_norm:    [B,M,N,1,H,W], normalized along N
            rerr:      [B,M,N,1,H,W]
            w_strength:[B,M,1,H,W] optional raw merge-weight mass

        Returns:
            D_out: [B,M,1,H,W]
            aux: dict of small diagnostic tensors
        """
        B, M, _, H, W = D_fused.shape
        N = D_warp.shape[2]

        Df = D_fused.reshape(B * M, 1, H, W)
        Dw = D_warp.reshape(B * M, N, 1, H, W)
        Wn = w_norm.reshape(B * M, N, 1, H, W).clamp_min(0.0)
        Re = rerr.reshape(B * M, N, 1, H, W)
        depth_valid = (
            torch.isfinite(Dw)
            & (Dw > 0.05)
            & (Dw < 1000.0)
        ).to(dtype=Wn.dtype)
        Wn = Wn * depth_valid
        Wn = Wn / Wn.sum(dim=1, keepdim=True).clamp_min(self.eps)
        Dw = torch.where(
            depth_valid > 0,
            Dw,
            torch.full_like(Dw, 1000.0),
        )

        inv_fused = 1.0 / Df.clamp_min(0.05)
        inv_warp = 1.0 / Dw.clamp_min(0.05)
        inv_fused_n = self.in_norm(inv_fused.float()).to(dtype=inv_fused.dtype)

        inv_mean = (Wn * inv_warp).sum(dim=1)
        inv_var = (Wn * (inv_warp - inv_mean.unsqueeze(1)).pow(2)).sum(dim=1)
        inv_var = (inv_var + self.eps).sqrt()
        min_rerr = Re.min(dim=1).values.clamp(0.0, 1.0)
        mean_rerr = (Wn * Re.clamp(0.0, 1.0)).sum(dim=1)

        entropy = -(Wn * (Wn + self.eps).log()).sum(dim=1)
        if N > 1:
            entropy = entropy / torch.log(torch.tensor(float(N), device=entropy.device, dtype=entropy.dtype))

        if w_strength is None:
            strength = torch.ones_like(Df)
        else:
            strength = w_strength.reshape(B * M, 1, H, W)
            strength = torch.log1p(strength.float()).to(dtype=Df.dtype)

        x = torch.cat(
            [
                inv_fused,
                inv_fused_n,
                inv_var,
                min_rerr,
                mean_rerr,
                entropy,
                strength,
            ],
            dim=1,
        )

        work_size = x.shape[-2:]
        if self.internal_downsample > 1:
            work_size = (
                max(1, H // self.internal_downsample),
                max(1, W // self.internal_downsample),
            )
            x_in = self._resize(x, work_size)
        else:
            x_in = x

        e1 = self.enc1(x_in)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        pred = self.out(d1)
        if pred.shape[-2:] != (H, W):
            pred = self._resize(pred, (H, W))

        delta_inv = torch.tanh(pred[:, :1]) * self.max_delta_inv
        gate = torch.sigmoid(pred[:, 1:2])
        inv_out = inv_fused + gate * delta_inv
        D_out = 1.0 / inv_out.clamp_min(self.eps)
        D_out = D_out.reshape(B, M, 1, H, W)

        aux = {
            "depth_refiner_delta_inv": delta_inv.reshape(B, M, 1, H, W),
            "depth_refiner_gate": gate.reshape(B, M, 1, H, W),
            "depth_refiner_entropy": entropy.reshape(B, M, 1, H, W),
            "depth_refiner_inv_var": inv_var.reshape(B, M, 1, H, W),
        }
        return D_out, aux


class ConvGRUCell2d(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.z = nn.Conv2d(in_ch + hid_ch, hid_ch, k, 1, p)
        self.r = nn.Conv2d(in_ch + hid_ch, hid_ch, k, 1, p)
        self.h = nn.Conv2d(in_ch + hid_ch, hid_ch, k, 1, p)
        # nn.init.zeros_(self.z.bias)
        # nn.init.zeros_(self.r.bias)
        # nn.init.zeros_(self.h.bias)
        nn.init.kaiming_normal_(self.z.weight, nonlinearity="sigmoid")
        nn.init.kaiming_normal_(self.r.weight, nonlinearity="sigmoid")
        nn.init.kaiming_normal_(self.h.weight, nonlinearity="tanh")
        

    def forward(self, x, h_prev):
        # x: [B, C_in, H, W], h_prev: [B, C_h, H, W] or None
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0),
                self.z.out_channels,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )
        cat = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.z(cat))
        r = torch.sigmoid(self.r(cat))
        cat_h = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.h(cat_h))
        h = (1 - z) * h_prev + z * h_tilde
        return h


class DepthSeqAggGRU(nn.Module):
    """
    N개 뷰(depth)를 가변 길이 시퀀스로 보고 ConvGRU로 픽셀단 은닉맵으로 축약.
    입력 스텝 x_t = [inv_d_src, inv_d_src - inv_d_fused, valid, (optional snr)]
    """

    def __init__(self, hid_ch=32, use_snr=True, eps=1e-6):
        super().__init__()
        self.use_snr = use_snr
        in_ch = 3 + (1 if use_snr else 0)  # inv, delta_inv, valid, (snr)
        self.cell = ConvGRUCell2d(in_ch=in_ch, hid_ch=hid_ch, k=3)
        self.eps = eps

        # 선택: 마지막에 살짝 정규화/압축
        self.post = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hid_ch, hid_ch, 3, 1, 1),
            nn.ReLU(True),
        )

    def forward(self, D_warp, D_fused, c_warp, valid_geo, snr=None):
        """
        D_warp:   [B,M,N,1,H,W]
        D_fused:  [B,M,1,H,W]
        valid_geo:[B,M,N,1,H,W] in {0,1}
        snr:      [B,M,N,1,H,W] in [0,1] or None
        return: F_agg [B,M,C,H,W]
        """
        B, M, N, _, H, W = D_warp.shape

        device = D_warp.device
        dtype = D_warp.dtype

        # disparity 변환
        inv_fused = 1.0 / (D_fused.clamp_min(self.eps))  # [B,M,1,H,W]
        inv_fused_rep = inv_fused.unsqueeze(2).expand(B, M, N, 1, H, W)

        inv_src = 1.0 / (D_warp.clamp_min(self.eps))  # [B,M,N,1,H,W]
        delta_inv = inv_src - inv_fused_rep  # [B,M,N,1,H,W]
        valid = (valid_geo > 0.5).float() * c_warp

        if self.use_snr:
            if snr is None:
                snr = torch.ones(B, M, N, 1, H, W, device=device, dtype=dtype) * 0.5
            snr_feat = snr
        else:
            snr_feat = None

        # N 차원 순회: (메모리/속도 절충 위해 단순 for; 필요 시 블록 처리로 개선 가능)
        h = None
        for t in range(N):
            x_list = [
                inv_src[:, :, t],
                delta_inv[:, :, t],
                valid[:, :, t],
            ]  # [B,M,1,H,W] × 3
            if snr_feat is not None:
                x_list.append(snr_feat[:, :, t])
            x_t = torch.cat(x_list, dim=2).view(B * M, -1, H, W)  # [B*M, C_in, H, W]

            # invalid는 입력을 0으로, 은닉은 keep; 게이트가 알아서 누락 반영
            # (원하면 valid로 z게이트 바이어스 조절하는 별도 기법도 가능)
            x_t = x_t * 1.0  # placeholder for potential scaling

            if h is None:
                h = None
            else:
                h = h.view(B * M, -1, H, W)

            h = self.cell(x_t, h)  # [B*M, hid, H, W]

        h = h.view(B, M, -1, H, W)
        # 후처리
        h_out = self.post(h.view(B * M, -1, H, W)).view(B, M, -1, H, W)
        return h_out  # [B,M,hid,H,W]


class TinyRefinerGRU(nn.Module):
    """
    TinyRefiner + (뷰-시퀀스 집계 feature)
    입력: [inv_Dfused, inv_Dfused_norm, edge_hint, F_agg(C)]
    """

    def __init__(self, agg_ch=32, mid=64, eps=1e-6):
        super().__init__()
        self.eps = eps

        in_ch = 3 + agg_ch

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(True),
        )
        self.out = nn.Conv2d(mid, 1, 1, 1, 0)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="tanh")
        nn.init.zeros_(self.out.bias)

        self.ins_norm = nn.InstanceNorm2d(1, affine=False)

    def forward(self, Dfused, edge_hint, F_agg):
        """
        Dfused:   [B,M,1,H,W] or [B*,1,H,W]
        edge_hint:[B,M,1,H,W] or [B*,1,H,W]
        F_agg:    [B,M,C,H,W] or [B*,C,H,W]
        """
        # 배치 차원 정규화 (B*, ..)로 맞추기

        if Dfused.dim() == 5:
            B, M, _, H, W = Dfused.shape
            Df = Dfused.view(B * M, 1, H, W)
            Eh = edge_hint.view(B * M, 1, H, W)
            Fa = F_agg.view(B * M, F_agg.shape[2], H, W)
        else:
            Df, Eh, Fa = Dfused, edge_hint, F_agg
            _, _, H, W = Df.shape

        inv_Df = 1.0 / (Df.clamp_min(self.eps))
        inv_Df_norm = self.ins_norm(inv_Df)
        x = torch.cat([inv_Df, inv_Df_norm, Eh, Fa], dim=1)  # [B*, 3+C, H, W]

        e1 = self.enc1(x)
        e2 = self.enc2(e1) + e1
        e3 = self.enc3(e2) + e2
        d2 = self.dec2(torch.cat([e3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        residual_norm = torch.tanh(self.out(d1)) * 0.1
        inv_refined = inv_Df + residual_norm
        out = 1.0 / inv_refined.clamp_min(self.eps)
        if Dfused.dim() == 5:
            out = out.view(B, M, 1, H, W)
        return out
