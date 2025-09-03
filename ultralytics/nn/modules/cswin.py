import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Minimal offline CSWin-like backbone for YOLOv10
# Note: This implements a cross-axis (H/W) attention that is
# faithful in *structure* to CSWin (stages, splits of heads
# into horizontal & vertical groups, DWConv positional term),
# but uses a simplified stripe attention for robustness.
# It returns strides [8, 16, 32] for the neck and remaps
# channels to [256, 512, 1024].
# ------------------------------------------------------------

# -----------------
# Utility modules
# -----------------
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
    def forward(self, x):
        return self.dw(x)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.dw = DWConv(hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dw(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -----------------
# Cross-axis (H/W) attention (CSWin-style split heads)
# -----------------
class CrossAxisAttention(nn.Module):
    """
    Split heads: half attend along H stripes, half along W stripes.
    Window size sw controls local receptive field (stripe thickness).
    Includes a depthwise conv positional term (LePE-like) on V.
    """
    def __init__(self, dim, num_heads=8, sw=7, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads should be even (split H/W)."
        self.dim = dim
        self.num_heads = num_heads
        self.h_heads = num_heads // 2
        self.v_heads = num_heads // 2
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sw = sw

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.pos_v = DWConv(dim)  # positional encoding on V
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _stripe_attn_H(self, q, k, v):
        # q,k,v: (B, heads, C_h, H, W), where C_h = head_dim
        B, Hh, C, H, W = q.shape
        sw = min(self.sw, H)
        pad = (0, 0, 0, (sw - H % sw) % sw)  # pad H to multiple of sw
        q = F.pad(q, pad)  # pad along H
        k = F.pad(k, pad)
        v = F.pad(v, pad)
        H_pad = q.shape[3]
        n_stripes = H_pad // sw
        # group along H into stripes of height sw (full width)
        q = q.view(B, Hh, C, n_stripes, sw, W)
        k = k.view(B, Hh, C, n_stripes, sw, W)
        v = v.view(B, Hh, C, n_stripes, sw, W)
        # reshape to (B*Hh*n_stripes, sw*W, C)
        q = q.permute(0,1,3,4,5,2).contiguous().view(B*Hh*n_stripes, sw*W, C)
        k = k.permute(0,1,3,4,5,2).contiguous().view(B*Hh*n_stripes, sw*W, C)
        v = v.permute(0,1,3,4,5,2).contiguous().view(B*Hh*n_stripes, sw*W, C)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B*Hh*n_stripes, sw*W, C)
        out = out.view(B, Hh, n_stripes, sw, W, C).permute(0,1,5,2,3,4).contiguous()
        out = out.view(B, Hh, C, H_pad, W)
        out = out[:, :, :, :H, :]  # remove pad
        return out

    def _stripe_attn_W(self, q, k, v):
        # q,k,v: (B, heads, C_h, H, W)
        B, Vh, C, H, W = q.shape
        sw = min(self.sw, W)
        pad = (0, (sw - W % sw) % sw, 0, 0)  # pad W to multiple of sw
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
        W_pad = q.shape[4]
        n_stripes = W_pad // sw
        # group along W into stripes of width sw (full height)
        q = q.view(B, Vh, C, H, n_stripes, sw)
        k = k.view(B, Vh, C, H, n_stripes, sw)
        v = v.view(B, Vh, C, H, n_stripes, sw)
        # reshape to (B*Vh*n_stripes, H*sw, C)
        q = q.permute(0,1,4,3,5,2).contiguous().view(B*Vh*n_stripes, H*sw, C)
        k = k.permute(0,1,4,3,5,2).contiguous().view(B*Vh*n_stripes, H*sw, C)
        v = v.permute(0,1,4,3,5,2).contiguous().view(B*Vh*n_stripes, H*sw, C)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.view(B, Vh, n_stripes, H, sw, C).permute(0,1,5,3,4,2).contiguous()
        out = out.view(B, Vh, C, H, W_pad)
        out = out[:, :, :, :, :W]
        return out

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        v = v + self.pos_v(v)  # positional term on V

        # split heads
        def reshape_heads(t):
            t = t.view(B, self.num_heads, self.head_dim, H, W)
            return t
        qh, qv = torch.split(reshape_heads(q), [self.h_heads, self.v_heads], dim=1)
        kh, kv = torch.split(reshape_heads(k), [self.h_heads, self.v_heads], dim=1)
        vh, vv = torch.split(reshape_heads(v), [self.h_heads, self.v_heads], dim=1)

        out_h = self._stripe_attn_H(qh, kh, vh)
        out_v = self._stripe_attn_W(qv, kv, vv)
        out = torch.cat([out_h, out_v], dim=1)  # concat heads back
        out = out.view(B, self.num_heads * self.head_dim, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class CSWinBlock(nn.Module):
    def __init__(self, dim, num_heads, sw=7, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = CrossAxisAttention(dim, num_heads=num_heads, sw=sw, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 2, 1)  # stride-2 conv
    def forward(self, x):
        return self.conv(x)

class PatchEmbed(nn.Module):
    def __init__(self, c_in=3, c_out=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c_in, c_out, 7, 4, 3),  # H/4, W/4
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )
    def forward(self, x):
        return self.proj(x)

# -----------------
# CSWin-like backbone (Tiny config by default)
# -----------------
class CSWinTransformer(nn.Module):
    """
    Stages:
      S1: C
      S2: 2C (stride 8)  -> OUT P3
      S3: 4C (stride 16) -> OUT P4
      S4: 8C (stride 32) -> OUT P5
    """
    def __init__(self,
                 embed_dim=64, depths=(1, 1, 6, 1), heads=(2, 4, 8, 8),
                 mlp_ratio=4.0, sw=(1, 2, 7, 7), out_indices=(1, 2, 3),
                 out_channels=(256, 512, 1024)):
        super().__init__()
        self.out_indices = out_indices
        C = embed_dim
        self.patch_embed = PatchEmbed(3, C)

        # Stage 1
        self.stage1 = nn.Sequential(*[CSWinBlock(C, heads[0], sw=sw[0], mlp_ratio=mlp_ratio) for _ in range(depths[0])])
        self.down1 = Downsample(C, C * 2)
        # Stage 2
        C2 = C * 2
        self.stage2 = nn.Sequential(*[CSWinBlock(C2, heads[1], sw=sw[1], mlp_ratio=mlp_ratio) for _ in range(depths[1])])
        self.down2 = Downsample(C2, C * 4)
        # Stage 3
        C3 = C * 4
        self.stage3 = nn.Sequential(*[CSWinBlock(C3, heads[2], sw=sw[2], mlp_ratio=mlp_ratio) for _ in range(depths[2])])
        self.down3 = Downsample(C3, C * 8)
        # Stage 4
        C4 = C * 8
        self.stage4 = nn.Sequential(*[CSWinBlock(C4, heads[3], sw=sw[3], mlp_ratio=mlp_ratio) for _ in range(depths[3])])

        # Lateral 1x1 convs to match YOLO neck channels
        self.lateral = nn.ModuleList([
            nn.Conv2d(C2, out_channels[0], 1),
            nn.Conv2d(C3, out_channels[1], 1),
            nn.Conv2d(C4, out_channels[2], 1),
        ])
        self.out_channels = out_channels

    def forward(self, x):
        # Stem
        x = self.patch_embed(x)            # /4
        x = self.stage1(x)
        # S2
        x2 = self.down1(x)                 # /8
        x2 = self.stage2(x2)
        # S3
        x3 = self.down2(x2)                # /16
        x3 = self.stage3(x3)
        # S4
        x4 = self.down3(x3)                # /32
        x4 = self.stage4(x4)

        # outputs P3,P4,P5 with mapped channels
        p3 = self.lateral[0](x2)
        p4 = self.lateral[1](x3)
        p5 = self.lateral[2](x4)
        return [p3, p4, p5]
    
class Stage(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x[self.index]