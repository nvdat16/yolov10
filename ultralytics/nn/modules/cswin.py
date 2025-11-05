# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# import numpy as np
# from einops.layers.torch import Rearrange
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# def img2windows(img, H_sp, W_sp):
#     B, C, H, W = img.shape
#     img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#     img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
#     return img_perm
#
#
# def windows2img(img_splits_hw, H_sp, W_sp, H, W):
#     B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
#     img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
#     img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return img
#
#
# class LePEAttention(nn.Module):
#     def __init__(self, dim, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out or dim
#         self.split_size = split_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.idx = idx
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def im2cswin(self, x, H, W, H_sp, W_sp):
#         B, N, C = x.shape
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = img2windows(x, H_sp, W_sp)
#         x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
#         return x
#
#     def get_lepe(self, x, func, H, W, H_sp, W_sp):
#         B, N, C = x.shape
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
#         lepe = func(x)
#         lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
#         x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
#         return x, lepe
#
#     def forward(self, qkv, input_resolution):
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         B, L, C = q.shape
#         H, W = input_resolution
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         if self.idx == -1:
#             H_sp, W_sp = H, W
#         elif self.idx == 0:
#             H_sp, W_sp = H, self.split_size
#         elif self.idx == 1:
#             H_sp, W_sp = self.split_size, W
#         else:
#             raise ValueError(f"ERROR MODE {self.idx}")
#
#         q = self.im2cswin(q, H, W, H_sp, W_sp)
#         k = self.im2cswin(k, H, W, H_sp, W_sp)
#         v, lepe = self.get_lepe(v, self.get_v, H, W, H_sp, W_sp)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#         attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
#         attn = self.attn_drop(attn)
#         x = (attn @ v) + lepe
#         x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)
#         x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)
#         return x
#
#
# class CSWinBlock(nn.Module):
#     def __init__(self, dim, input_resolution, num_heads, split_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_stage=False):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.split_size = split_size
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.norm1 = norm_layer(dim)
#         self.branch_num = 1 if last_stage or min(self.input_resolution) <= self.split_size else 2
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)
#         if self.branch_num == 1:
#             self.attns = nn.ModuleList(
#                 [LePEAttention(dim, -1, split_size=split_size, num_heads=num_heads, qk_scale=qk_scale,
#                                attn_drop=attn_drop, proj_drop=drop)])
#         else:
#             self.attns = nn.ModuleList([LePEAttention(dim // 2, i, split_size=split_size, num_heads=num_heads // 2,
#                                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop) for i in
#                                         range(self.branch_num)])
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer,
#                        drop=drop)
#         self.norm2 = norm_layer(dim)
#
#     def forward(self, x, input_resolution):
#         H, W = input_resolution
#         B, L, C = x.shape
#         assert L == H * W, f"Input feature has wrong size: {L} vs {H * W}"
#
#         shortcut = x
#         img = self.norm1(x)
#         qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
#
#         if self.branch_num == 2:
#             x1 = self.attns[0](qkv[:, :, :, :C // 2], (H, W))
#             x2 = self.attns[1](qkv[:, :, :, C // 2:], (H, W))
#             attened_x = torch.cat([x1, x2], dim=2)
#         else:
#             attened_x = self.attns[0](qkv, (H, W))
#
#         attened_x = self.proj(attened_x)
#         x = shortcut + self.drop_path(attened_x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class Merge_Block(nn.Module):
#     def __init__(self, dim, dim_out):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
#         self.norm = nn.LayerNorm(dim_out)
#
#     def forward(self, x, H, W):
#         B, L, C = x.shape
#         assert L == H * W
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = self.conv(x)
#         B, C, H_new, W_new = x.shape
#         x = x.view(B, C, -1).transpose(-2, -1).contiguous()
#         x = self.norm(x)
#         return x, (H_new, W_new)
#
#
# class CSWinStem(nn.Module):
#     def __init__(self, in_chans=3, embed_dim=64):
#         super().__init__()
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3)
#         self.rearrange = Rearrange('b c h w -> b (h w) c')
#         self.norm = nn.LayerNorm(embed_dim)
#
#     def forward(self, x):
#         x = self.proj(x)
#         H_out, W_out = x.shape[2:]
#         x = self.rearrange(x)
#         x = self.norm(x)
#         return x, (H_out, W_out)
#
#
# class CSWinStage(nn.Module):
#     def __init__(self, dim, depth, num_heads, split_size, use_checkpoint=True):
#         super().__init__()
#         self.use_checkpoint = use_checkpoint
#         norm_layer = nn.LayerNorm
#         mlp_ratio = 4.0
#         qkv_bias = True
#         drop, attn_drop, drop_path_rate = 0.0, 0.0, 0.1
#
#         self.blocks = nn.ModuleList([
#             CSWinBlock(dim=dim, input_resolution=(0, 0), num_heads=num_heads,
#                        split_size=split_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                        drop=drop, attn_drop=attn_drop,
#                        drop_path=dpr, norm_layer=norm_layer,
#                        last_stage=(depth == 2 and dim == 1024))
#             for i, dpr in enumerate(torch.linspace(0, drop_path_rate, depth))])
#         self.norm = norm_layer(dim)
#
#     def forward(self, x_res):
#         x, (H, W) = x_res
#         for blk in self.blocks:
#             if self.use_checkpoint and self.training:
#                 x = checkpoint.checkpoint(blk, x, (H, W), use_reentrant=False)
#             else:
#                 x = blk(x, (H, W))
#         x = self.norm(x)
#         return x, (H, W)
#
#
# class CSWinDownsample(Merge_Block):
#     def __init__(self, dim, dim_out):
#         super().__init__(dim, dim_out)
#
#     def forward(self, x_res):
#         x, (H, W) = x_res
#         return super().forward(x, H, W)
#
#
# class Reshape(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x_res):
#         x, (H, W) = x_res
#         B, L, C = x.shape
#         return x.transpose(-2, -1).contiguous().view(B, C, H, W)

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LePEAttention(nn.Module):
    def __init__(self, dim, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, H, W, H_sp, W_sp):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func, H, W, H_sp, W_sp):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, input_resolution):
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        H, W = input_resolution
        assert L == H * W, "flatten img_tokens has wrong size"

        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            H_sp, W_sp = self.split_size, W
        else:
            raise ValueError(f"ERROR MODE {self.idx}")

        q = self.im2cswin(q, H, W, H_sp, W_sp)
        k = self.im2cswin(k, H, W, H_sp, W_sp)
        v, lepe = self.get_lepe(v, self.get_v, H, W, H_sp, W_sp)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)
        return x


class CSWinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, split_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_stage=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 1 if last_stage or min(self.input_resolution) <= self.split_size else 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        if self.branch_num == 1:
            self.attns = nn.ModuleList(
                [LePEAttention(dim, -1, split_size=split_size, num_heads=num_heads, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop)])
        else:
            self.attns = nn.ModuleList([LePEAttention(dim // 2, i, split_size=split_size, num_heads=num_heads // 2,
                                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop) for i in
                                        range(self.branch_num)])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, input_resolution):
        H, W = input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: {L} vs {H * W}"

        shortcut = x
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], (H, W))
            x2 = self.attns[1](qkv[:, :, :, C // 2:], (H, W))
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv, (H, W))

        attened_x = self.proj(attened_x)
        x = shortcut + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C, H_new, W_new = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x, (H_new, W_new)


class CSWinStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3)
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H_out, W_out = x.shape[2:]
        x = self.rearrange(x)
        x = self.norm(x)
        return x, (H_out, W_out)


class CSWinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, split_size, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        norm_layer = nn.LayerNorm
        mlp_ratio = 4.0
        qkv_bias = True
        drop, attn_drop, drop_path_rate = 0.0, 0.0, 0.1

        self.blocks = nn.ModuleList([
            CSWinBlock(dim=dim, input_resolution=(0, 0), num_heads=num_heads,
                       split_size=split_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=dpr, norm_layer=norm_layer,
                       last_stage=(depth == 2 and dim == 1024))
            for i, dpr in enumerate(torch.linspace(0, drop_path_rate, depth))])
        self.norm = norm_layer(dim)

    def forward(self, x_res):
        # SỬA LỖI Ở ĐÂY: Xử lý cả đầu vào dạng tuple và tensor
        if isinstance(x_res, tuple):
            x, (H, W) = x_res
        else:
            x = x_res
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, (H, W), use_reentrant=False)
            else:
                x = blk(x, (H, W))
        x = self.norm(x)
        return x, (H, W)


class CSWinDownsample(Merge_Block):
    def __init__(self, dim, dim_out):
        super().__init__(dim, dim_out)

    def forward(self, x_res):
        # SỬA LỖI Ở ĐÂY: Xử lý cả đầu vào dạng tuple và tensor
        if isinstance(x_res, tuple):
            x, (H, W) = x_res
        else:
            x = x_res
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

        return super().forward(x, H, W)


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_res):
        # SỬA LỖI Ở ĐÂY: Xử lý cả đầu vào dạng tuple và tensor
        if isinstance(x_res, tuple):
            x, (H, W) = x_res
        else:
            # Nếu đầu vào là tensor, nó có thể đã được flatten hoặc chưa
            if x_res.ndim == 3:  # (B, L, C)
                x = x_res
                B, L, C = x.shape
                H = W = int(L ** 0.5)
            else:  # (B, C, H, W)
                x = x_res
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)

        B, L, C = x.shape
        return x.transpose(-2, -1).contiguous().view(B, C, H, W)