# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Implementation of Swin models from :paper:`swin`.

This code is adapted from https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py with minimal modifications.  # noqa
--------------------------------------------------------
Swin Transformer
Copyright (c) 2021 Microsoft
Licensed under The MIT License [see LICENSE for details]
Written by Ze Liu, Yutong Lin, Yixuan Wei
--------------------------------------------------------
LICENSE: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/461e003166a8083d0b620beacd4662a2df306bd6/LICENSE
"""

from functools import partial
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.swin import (
    Mlp, 
    window_partition, 
    window_reverse, 
    PatchEmbed, 
    PatchMerging, 
    BasicLayer,
    SwinTransformer,
)

_to_2tuple = nn.modules.utils._ntuple(2)



def reshape_as_aspect_ratio(x, ratio, channel_last=False):
    assert x.ndim == 3
    B, N, C = x.size()
    # print('size',ratio,x.size())
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    perm = (0, 1, 2) if channel_last else (0, 2, 1)
    
    return x.permute(*perm).view(B, C, s * ratio[0], s * ratio[1])

def get_aspect_ratio(x, y):
    gcd = math.gcd(x, y)
    return x // gcd, y // gcd

class TokenClusteringBlock(nn.Module):
    def __init__(self, num_spixels=None, n_iters=5, temperture=200, window_size=7):
        super().__init__()
        if isinstance(num_spixels, tuple):
            assert len(num_spixels) == 2
        elif num_spixels is not None:
            x = int(math.sqrt(num_spixels))
            assert x * x == num_spixels
            num_spixels = (x, x)
        self.num_spixels = num_spixels
        self.n_iters = n_iters
        self.temperture = 1 / temperture
        assert window_size % 2 == 1
        self.r = window_size // 2

    def calc_init_centroid(self, images, num_spixels_width, num_spixels_height):
        """
        calculate initial superpixels

        Args:
            images: torch.Tensor
                A Tensor of shape (B, C, H, W)
            spixels_width: int
                initial superpixel width
            spixels_height: int
                initial superpixel height

        Return:
            centroids: torch.Tensor
                A Tensor of shape (B, C, H * W)
            init_label_map: torch.Tensor
                A Tensor of shape (B, H * W)
            num_spixels_width: int
                A number of superpixels in each column
            num_spixels_height: int
                A number of superpixels int each raw
        """
        batchsize, channels, height, width = images.shape
        device = images.device

        centroids = torch.nn.functional.adaptive_avg_pool2d(
            images, (num_spixels_height, num_spixels_width)
        )

        with torch.no_grad():
            num_spixels = num_spixels_width * num_spixels_height
            labels = (
                torch.arange(num_spixels, device=device)
                .reshape(1, 1, *centroids.shape[-2:])
                .type_as(centroids)
            )
            init_label_map = torch.nn.functional.interpolate(
                labels, size=(height, width), mode="nearest"
            ).type_as(centroids)
            init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

        init_label_map = init_label_map.reshape(batchsize, -1)
        centroids = centroids.reshape(batchsize, channels, -1)

        return centroids, init_label_map

    def forward(self, pixel_features, num_spixels=None):
        # import pdb; pdb.set_trace()
        if num_spixels is None:
            num_spixels = self.num_spixels
            assert num_spixels is not None
        else:
            if isinstance(num_spixels, tuple):
                assert len(num_spixels) == 2
            else:
                x = int(math.sqrt(num_spixels))
                assert x * x == num_spixels
                num_spixels = (x, x)
        num_spixels_height, num_spixels_width = num_spixels
        num_spixels = num_spixels_width * num_spixels_height
        spixel_features, init_label_map = self.calc_init_centroid(
            pixel_features, num_spixels_width, num_spixels_height
        )

        device = init_label_map.device
        spixels_number = torch.arange(num_spixels, device=device)[None, :, None]
        relative_labels_widths = init_label_map[:, None] % num_spixels_width - spixels_number % num_spixels_width
        relative_labels_heights = torch.div(init_label_map[:, None], num_spixels_width, rounding_mode='trunc') - torch.div(spixels_number, num_spixels_width, rounding_mode='trunc')
        mask = torch.logical_and(torch.abs(relative_labels_widths) <= self.r, torch.abs(relative_labels_heights) <= self.r)
        mask_dist = (~mask) * 1e16

        pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)  # (B, C, L)
        permuted_pixel_features = pixel_features.permute(0, 2, 1)       # (B, L, C)

        for _ in range(self.n_iters):
            dist_matrix = self.pairwise_dist(pixel_features, spixel_features)    # (B, L', L)
            dist_matrix += mask_dist
            affinity_matrix = (-dist_matrix * self.temperture).softmax(1)
            spixel_features = torch.bmm(affinity_matrix.detach(), permuted_pixel_features)
            spixel_features = spixel_features / affinity_matrix.detach().sum(2, keepdim=True).clamp_(min=1e-16)
            spixel_features = spixel_features.permute(0, 2, 1)
        
        dist_matrix = self.pairwise_dist(pixel_features, spixel_features)
        hard_labels = torch.argmin(dist_matrix, dim=1)

        return spixel_features.permute(0, 2, 1), hard_labels, (num_spixels_height, num_spixels_width)

    def pairwise_dist(self, f1, f2):
        return ((f1 * f1).sum(dim=1).unsqueeze(1)
                + (f2 * f2).sum(dim=1).unsqueeze(2)
                - 2 * torch.einsum("bcm, bcn -> bmn", f2, f1))

    def extra_repr(self):
        return f"num_spixels={self.num_spixels}, n_iters={self.n_iters}"


def naive_unpool(f_regions, region_indices):
    _, _, C = f_regions.shape
    N, L = region_indices.shape
    index = region_indices.view(N, L, 1).expand(N, L, C)
    result = f_regions.gather(1, index)
    return result


class State:
    def __init__(self, unpooling):
        self.unpooling = unpooling
        self.__updated = False

    @property
    def updated(self):
        return self.__updated

    def get(self, name, default=None):
        return getattr(self, name, default)

    def update_state(self, **states: dict):
        self.__updated = True
        for k, v in states.items():
            setattr(self, k, v)

    def call(self, input: torch.Tensor):
        return self.unpooling(input, self)


class UnpoolingBase(nn.Module):
    def forward(self, x, state: State):
        if not state.updated:
            return x, False
        return self._forward(x, state)

    def derive_unpooler(self):
        return State(self)


class NaiveUnpooling(UnpoolingBase):
    def _forward(self, x, state: State):
        return naive_unpool(x, state.hard_labels), False


class TokenReconstructionBlock(UnpoolingBase):
    def __init__(self, k=3, temperture=200):
        super().__init__()

        self.k = k
        self.temperture = 1 / temperture

    def _forward(self, x, state: State):
        feat = state.feat_before_pooling
        sfeat = state.feat_after_pooling
        ds = (
            (feat * feat).sum(dim=2).unsqueeze(2)
            + (sfeat * sfeat).sum(dim=2).unsqueeze(1)
            - 2 * torch.einsum("bnc, bmc -> bnm", feat, sfeat)
        )  # distance between features and super-features
        ds[ds < 0] = 0
        weight = torch.exp(-self.temperture * ds)
        if self.k >= 0:
            topk, indices = torch.topk(weight, k=self.k, dim=2)
            mink = torch.min(topk, dim=-1).values
            mink = mink.unsqueeze(-1).repeat(1, 1, weight.shape[-1])
            mask = torch.ge(weight, mink)
            zero = Variable(torch.zeros_like(weight)).cuda()
            attention = torch.where(mask, weight, zero)
        attention = F.normalize(attention, dim=2)
        ret = torch.einsum("bnm, bmc -> bnc", attention, x)

        return ret, False


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ckpt_window_size=(12, 12),
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.ckpt_window_size = ckpt_window_size

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * ckpt_window_size[0] - 1) * (2 * ckpt_window_size[1] - 1), num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def resize_rpb_table(self, relative_position_bias_table):
        table = relative_position_bias_table.permute(1, 0).reshape(
            -1, 2 * self.ckpt_window_size[0] - 1, 2 * self.ckpt_window_size[1] - 1
        )
        table = F.interpolate(
            table[None],
            (2 * self.window_size[0] - 1, 2 * self.window_size[1] - 1),
            mode="bilinear",
        )
        table = table.unsqueeze(0).view(
            -1, (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        )
        return table.permute(1, 0)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if (
            self.ckpt_window_size[0] != self.window_size[0]
            or self.ckpt_window_size[1] != self.window_size[1]
        ):
            self.relative_position_bias_table = nn.Parameter(
                self.resize_rpb_table(self.relative_position_bias_table)
            )
            self.ckpt_window_size = self.window_size
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ckpt_window_size=12,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            ckpt_window_size=_to_2tuple(ckpt_window_size),
        )

        if drop_path > 0.0:
            from timm.models.layers import DropPath

            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class HourglassBasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        token_clustering_block:TokenClusteringBlock=None,
        clustering_location=-1,
        token_reconstruction_block=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        clustered_ws = token_clustering_block.num_spixels[0]

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size if i < clustering_location else clustered_ws,
                    shift_size=0
                    if (i % 2 == 0)
                    else (window_size if i < clustering_location else clustered_ws) // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    ckpt_window_size=window_size,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.token_clustering_block = token_clustering_block
        self.clustering_location = clustering_location
        self.token_reconstruction_block = token_reconstruction_block

    def cluster(self, x, reconstructer):
        H, W = reconstructer.hw_shape
        C = x.shape[-1]
        x = x.view(-1, H, W, C)  # B, H, W, C
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        x = window_partition(x, self.window_size)  # B*H*W, WH, WW, C

        out, hard_labels, shape = self.token_clustering_block(x.permute(0, 3, 1, 2))  # B*H*W, Wh*Ww, C
        reconstructer.update_state(hard_labels=hard_labels)
        reconstructer.update_state(
            feat_before_pooling=x.view(-1, self.window_size * self.window_size, C)
        )
        reconstructer.update_state(feat_after_pooling=out)

        # merge window
        out = out.view(-1, *shape, C)
        H = Hp // self.window_size * shape[0]
        W = Wp // self.window_size * shape[1]
        out = window_reverse(out, shape[0], H, W)  # B, h, w, C
        return out.view(-1, H * W, C), shape

    def get_mask(self, H, W, device):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def reconstruct(self, x, H, W, recontructer):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        x, _ = recontructer.call(x)
        self.window_size = self.org_window_size

        # merge windows
        x = x.view(-1, self.window_size, self.window_size, C)
        H, W = recontructer.hw_shape
        H = int(np.ceil(H / self.window_size)) * self.window_size
        W = int(np.ceil(W / self.window_size)) * self.window_size
        x = window_reverse(x, self.window_size, H, W)  # B H' W' C
        if H != recontructer.hw_shape[0] or W != recontructer.hw_shape[1]:
            H, W = recontructer.hw_shape
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, -1, C)
        return x

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        attn_mask = self.get_mask(H, W, x.device)
        B, _, C = x.shape

        recontructer = self.token_reconstruction_block.derive_unpooler()
        recontructer.update_state(aspect_ratio=get_aspect_ratio(H, W))
        recontructer.update_state(hw_shape=(H, W))

        for i, blk in enumerate(self.blocks):
            if i == self.clustering_location:
                x, (Wh, Ww) = self.cluster(x, recontructer)
                H = int(np.ceil(H / self.window_size)) * Wh
                W = int(np.ceil(W / self.window_size)) * Ww
                assert Wh == Ww
                self.org_window_size = self.window_size
                self.window_size = Wh
                attn_mask = self.get_mask(H, W, x.device)

            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if x.shape[1] != np.prod(recontructer.hw_shape):
            x = self.reconstruct(x, H, W, recontructer)
            H, W = recontructer.hw_shape

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class HourglassSwinTransformer(SwinTransformer):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted
            Windows`  - https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        token_clustering_cfg=None,
        clustering_location=-1,
        token_reconstruction_cfg=None,
    ):
        super().__init__(
            pretrain_img_size,
            patch_size,
            in_chans,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            out_indices,
            frozen_stages,
            use_checkpoint,
        )

        token_clustering_block = TokenClusteringBlock(**token_clustering_cfg)
        token_reconstruction_block = TokenReconstructionBlock(**token_reconstruction_cfg)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        for i_layer in range(self.num_layers):
            if clustering_location < self.layers[i_layer].depth and clustering_location > 0:
                self.layers[i_layer] = HourglassBasicLayer(
                    dim=int(embed_dim * 2**i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    token_clustering_block=token_clustering_block,
                    clustering_location=clustering_location,
                    token_reconstruction_block=token_reconstruction_block,
                )
                break
            clustering_location -= self.layers[i_layer].depth

        self._freeze_stages()
        self.apply(self._init_weights)
