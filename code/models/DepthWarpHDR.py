import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import Literal

from modules.depth_densify.DepthPrompting.model_list import import_model
from modules.depth_densify.PromptDA.promptda.promptda import PromptDA
from method.depth_warp_hdr.hdr_transformer_gru import HDRReconstructionGRU
import time


def pad(img, div=14):
    h, w = img.shape[-2:]
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    return F.pad(img, (0, pad_w, 0, pad_h), mode="replicate", value=0)


class DepthPromptingModule(nn.Module):

    def freeze_sparse(self, freeze=True):
        self.sparse_model.eval()
        for param in self.sparse_model.parameters():
            param.requires_grad = not freeze

    def freeze_dense(self, freeze=True):
        self.dense_model.eval()
        for param in self.dense_model.parameters():
            param.requires_grad = not freeze

    def __init__(
        self,
        sparse_base: Literal["DepthPrompting", "Inpaint"] = "DepthPrompting",
        dense_base="PromptDA",
        sparse_ckpt="modules/depth_densify/checkpoints/Depthprompting_depthformer_kitti.tar",
        dense_ckpt="modules/depth_densify/PromptDA/pretrained/promptda_small.ckpt",
        depth_div=1000,
        downsample=2,
        inpaint_kernel=49,
    ):
        super(DepthPromptingModule, self).__init__()
        self.depth_div = depth_div
        self.sparse_base = sparse_base
        if sparse_base == "DepthPrompting":
            args = Namespace(
                data_name="NYU",  # choices=('NYU', 'KITTIDC', 'IPAD', 'NUSCENE', 'VOID', 'SUNRGBD')
                model_name="depth_prompt_main",
                gamma="1.0,0.5,0.05,0.001",
                betas=(0.9, 0.999),
                epsilon=1e-8,
                prop_time=12,
                prop_kernel=7,
                conf_prop=False,
                augment=True,
                top_crop=0,
                min_depth=1e-3,
                max_depth=10.0,
                garg_crop=False,
                eigen_crop=False,
                patch_height=None,
                patch_width=None,
                no_res_pre=True,
                init_scailing=False,
                backbone="df",
            )
            self.sparse_model = import_model(args)
            state = torch.load(sparse_ckpt)
            if "state_dict" in state:
                state = state["state_dict"]
            for key in list(state.keys()):
                if key.startswith("module."):
                    state[key[7:]] = state.pop(key)
            self.sparse_model.load_state_dict(state)
            self.sparse_model.cuda().eval()

        if dense_base == "PromptDA":
            self.dense_model = PromptDA("vits", dense_ckpt).cuda().eval()
        if dense_base == None:
            self.dense_model = lambda x, y: y  # Identity function

        self.downsample = downsample
        self.inpaint_kernel = inpaint_kernel

    def fill_sparse_depth_map(self, depth, max_iter=10, kernel_size=25):

        eps = 1e-6
        padding = kernel_size // 2
        mask = (depth > 0).float()
        depth[depth >= 50000] = 0
        for _ in range(max_iter):
            depth_sum = F.avg_pool2d(
                depth * mask, kernel_size, stride=1, padding=padding
            )
            mask_sum = F.avg_pool2d(mask, kernel_size, stride=1, padding=padding)
            avg = depth_sum / (mask_sum + eps)
            depth = torch.where(mask > 0, depth, avg)
            mask = (depth > 0).float()
        depth[depth == 0] = 50000
        return depth

    def forward(self, rgb, sparse):
        h, w = rgb.shape[-2:]
        if self.downsample > 1:
            rgb = F.interpolate(rgb, scale_factor=1 / self.downsample, mode="bilinear")
            sparse = F.interpolate(
                sparse, scale_factor=1 / self.downsample, mode="nearest"
            )

        rgb = pad(rgb)
        rgb = (rgb / rgb.max()) ** (1 / 2.2)

        sparse = pad(sparse)

        if self.sparse_base == "DepthPrompting":
            depth_rough = (
                self.sparse_model({"rgb": rgb, "dep": sparse / self.depth_div})["pred"]
                * self.depth_div
            )
        if self.sparse_base == "Inpaint":
            depth_rough = self.fill_sparse_depth_map(
                sparse, kernel_size=self.inpaint_kernel
            )
        depth_dense = self.dense_model(rgb, depth_rough)
        if self.downsample > 1:
            depth_dense = F.interpolate(depth_dense, size=(h, w), mode="bilinear")
        return depth_dense[..., :h, :w]


class StereoHDR(nn.Module):
    def __init__(self):
        super(StereoHDR, self).__init__()
        # self.tri_ths_left = [0, 0.05, 1.1, 1.1]
        # self.tri_ths_right = [0, 0.03, 0.93, 0.99]
        self.tri_ths = [[0, 0.05, 1.1, 1.1], [0, 0.03, 0.93, 0.99]]

    def tent_weight(self, img, thresholds):
        """
        Parameters:
        -----------
        img : torch.Tensor
            normalized image tensor (range: [0, 1])
        thresholds : list or tuple
            [t0, t1, t2, t3] 값

        Returns:
        --------
        weight : torch.Tensor
            동일한 shape의 weight tensor
        """
        weight = torch.zeros_like(img)
        t0, t1, t2, t3 = thresholds

        # 조건 1: img <= t0 -> weight = 0 (초기값 유지)
        # 조건 2: t0 < img <= t1 -> linear interpolation
        cond = (img > t0) & (img <= t1)
        weight[cond] = (img[cond] - t0) / (t1 - t0 + 1e-8)

        # 조건 3: t1 < img < t2 -> weight = 1
        cond = (img > t1) & (img < t2)
        weight[cond] = 1.0

        # 조건 4: t2 <= img <= t3 -> linear interpolation
        cond = (img >= t2) & (img <= t3)
        weight[cond] = (t3 - img[cond]) / (t3 - t2 + 1e-8)

        # 조건 5: img > t3 -> weight = 0 (초기값 유지)
        return weight

    def forward(self, img_pair, exp_pair):
        """
        Parameters:
        -----------
        img_pair : list of torch.Tensor
            여러 장의 이미지 (일반적으로 shape은 (C, H, W)).
            만약 입력 이미지가 uint8이면 [0, 255] 범위를 가지므로, [0, 1]로 scaling.
        exp_pair : list of float
            각 이미지의 exposure time (예: [t1, t2, ...])

        Returns:
        --------
        hdr : torch.Tensor
            HDR 복원 결과 (shape: (C, H, W), float32)
        """
        num_images = len(img_pair)  # 이미지 개수

        # 입력 이미지가 float이 아닐 경우, [0,255] -> [0,1]로 scaling
        img_pair = [
            img.float() / 255.0 if img.dtype != torch.float32 else img
            for img in img_pair
        ]

        # 각 이미지에 대해 tent weight 계산
        W = [self.tent_weight(img, self.tri_ths[i]) for i, img in enumerate(img_pair)]

        # Radiance estimation
        numerator = sum(
            W[i] * (img_pair[i] / (exp_pair[i] + 1e-8)) for i in range(num_images)
        )
        denominator = sum(W)

        # 작은 수 eps로 0 division 방지
        eps = 1e-8
        hdr = numerator / (denominator + eps)

        return hdr


class ImageWarpingModule(nn.Module):
    def __init__(self):
        super(ImageWarpingModule, self).__init__()

    def forward(self, img, disparity, direction="r2l"):
        B, C, H, W = img.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1).float().cuda()
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1).float().cuda()
        grid_x_warped = grid_x - disparity.squeeze(1)
        grid_y_warped = grid_y

        grid_x_warped = 2.0 * grid_x_warped / (W - 1) - 1.0
        grid_y_warped = 2.0 * grid_y_warped / (H - 1) - 1.0

        grid = torch.stack((grid_x_warped, grid_y_warped), dim=-1).float()

        img_right_warped = F.grid_sample(
            img,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
        return img_right_warped


class DepthWarpHDR(nn.Module):

    def __init__(
        self,
        densify_depth: bool = True,
        sparse_base: Literal["DepthPrompting", "Inpaint"] = "DepthPrompting",
        dense_base: Literal["PromptDA", None] = "PromptDA",
        debug: bool = False,
    ):
        super(DepthWarpHDR, self).__init__()
        self.debug = debug
        self.densify_depth = densify_depth
        if densify_depth:
            self.depth_densify = DepthPromptingModule(
                sparse_base=sparse_base, dense_base=dense_base
            )
        self.warping = ImageWarpingModule()
        # self.hdr_module = StereoHDR()
        self.hdr_module = HDRReconstructionGRU(
            hdr_gamma=False, unet_denoise=True, gamma_activation=True
        )
        # self.exp_cont = ExposureControl()

    def forward(self, batch):
        img_l = batch["left"]
        img_r = batch["right"]
        exp_l = batch["exp_left"]
        exp_r = batch["exp_right"]
        sparse = batch["sparse"]
        fx = batch["fx"]
        baseline = batch["baseline"]
        if self.debug:
            time_begin = time.time()
        if self.densify_depth:
            dense_depth = self.depth_densify(img_l, sparse)
            if self.debug:
                print(
                    f"Depth Densification Time: {time.time() - time_begin:.4f} seconds"
                )
                time_begin = time.time()
        else:
            dense_depth = sparse
        disparity = fx * baseline / dense_depth
        warped_img = self.warping(img_r, disparity, direction="r2l")
        if self.debug:
            print(f"Image Warping Time: {time.time() - time_begin:.4f} seconds")
            time_begin = time.time()
        # hdr = self.hdr_module((img_l, warped_img), (exp_l, exp_r))
        hdr = self.hdr_module(img_l, warped_img, 1, exp_r / exp_l) / exp_l
        if self.debug:
            print(f"HDR Reconstruction Time: {time.time() - time_begin:.4f} seconds")
            time_begin = time.time()
        # new_exp, skew, skew_2 = self.hist_exp(img_l, img_r)

        return {
            "dense_depth": dense_depth,
            "hdr": hdr,
            "warped_img": warped_img,
            # "hist": hist,
            # "skew": skew,
            # "skew_r": skew_2,
            # "new_exp": new_exp,
        }

    def freeze_depth(self):
        self.depth_densify.eval()
        for param in self.depth_densify.parameters():
            param.requires_grad = False
