
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from torch.distributions.normal import Normal


_logger = logging.getLogger(__name__)
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from srm_conv import SRMConv2d
from spatialprior import Injector, Extractor, SpatialPriorModule_SRM
from torch.nn.init import normal_

from Conv.DepthwiseSeparableConv import *
from Conv.FDConv import *
from Conv.SFIConv import *
from Conv.WTConv import *
from torchvision.ops import DeformConv2d

'''
Here we define local difference adapter
'''

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class SFI(nn.Module):
    def __init__(self, adapter_dim):
        super().__init__()
        self.FirstMYconv = FirstSFIConv(kernel_size=(3, 3), in_channels=adapter_dim, out_channels=adapter_dim, alpha=0.5)
        self.SFIConv = SFIConv(kernel_size=(3, 3), in_channels=adapter_dim, out_channels=adapter_dim, bias=False, stride=1, alpha=0.5)
        self.SFIConvB = SFIConvB(in_channels=adapter_dim, out_channels=adapter_dim, alpha=0.5)
        self.LastSFIConv = LastSFIConv(kernel_size=(3, 3), in_channels=adapter_dim, out_channels=adapter_dim, alpha=0.5)

    def forward(self, x):
        x_out, y_out = self.FirstMYconv(x)

        i = x_out, y_out
        x_out, y_out = self.SFIConv(i)

        i = x_out, y_out
        x_out, y_out = self.SFIConvB(i)

        i = x_out, y_out
        out = self.LastSFIConv(i)

        return out


class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=False
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return out


class SRMConvLayer(nn.Module):
    """
    一个标准的SRM卷积层，使用3个固定的高通滤波器。
    """

    def __init__(self, in_channels):
        super().__init__()
        # 直接使用 torch.tensor 创建，并指定 dtype
        srm_kernel = torch.tensor([
            [[[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]],
            [[[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]],
            [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
        ], dtype=torch.float)  # 使用 torch.float (等同于 torch.float32)

        # 【核心修改】除数也用 torch.tensor 创建
        divisors = torch.tensor([4.0, 12.0, 2.0], dtype=torch.float).view(3, 1, 1, 1)

        srm_kernel = srm_kernel / divisors

        self.kernel = nn.Parameter(srm_kernel, requires_grad=False)
        self.in_channels = in_channels

    def forward(self, x):
        # 将输入通道分组，每个通道都应用3个滤波器
        # 输入: (B, C, H, W)

        # 核心修改：将 .view() 替换为 .reshape()
        x_reshaped = x.reshape(-1, 1, x.shape[2], x.shape[3])

        # 输出: (B*C, 3, H, W)
        out = F.conv2d(x_reshaped, self.kernel, padding=2)
        # (B, 3*C, H, W)
        out = out.view(x.shape[0], self.in_channels * 3, x.shape[2], x.shape[3])
        return out

class NoiseTextureExpertModule_V1(nn.Module):
    """ 
    增强版的 SRM 专家模块 (V1)。
    使用一个完整的残差块来替代简单的1x1卷积，以增强特征提取能力。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.srm_layer = SRMConvLayer(in_channels)
        srm_out_channels = in_channels * 3

        # 定义一个残差块来处理SRM特征
        self.residual_block = nn.Sequential(
            nn.Conv2d(srm_out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 用于匹配残差连接维度的 1x1 卷积（如果输入输出通道数不同）
        self.shortcut = nn.Sequential()
        if srm_out_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(srm_out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. 提取原始高频噪声特征
        srm_features = self.srm_layer(x) # 输出通道为 C * 3

        # 2. 通过残差块进行深度特征学习
        identity = self.shortcut(srm_features) # 捷径连接
        res_out = self.residual_block(srm_features)
        
        # 3. 添加残差并激活
        out = self.relu(res_out + identity)
        
        return out

class Conv2d_Adapter(nn.Module):
    def __init__(self, dim, adapter_dim, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, op_type='cv'):
        super(Conv2d_Adapter, self).__init__()

        self.adapter_down = nn.Linear(dim, adapter_dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(adapter_dim, dim)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.adapter_dim = adapter_dim
        if op_type == 'ds':
            self.adapter_conv = DepthwiseSeparableConv(in_channels=adapter_dim, out_channels=adapter_dim, kernel_size=3, stride=1, padding=1)
        elif op_type == 'fd':
            self.adapter_conv = FDConv(in_channels=adapter_dim, out_channels=adapter_dim, kernel_num=8, kernel_size=3, padding=1, bias=True)
        elif op_type == 'sfi':
            self.adapter_conv = NoiseTextureExpertModule_V1(in_channels=adapter_dim, out_channels=adapter_dim)
        elif op_type == 'wt':
            self.adapter_conv = WTConv2d(in_channels=adapter_dim, out_channels=adapter_dim)
        elif op_type == 'df':
            self.adapter_conv = DeformConv(in_channels=adapter_dim, out_channels=adapter_dim)
        self.op_type = op_type


    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.adapter_dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.adapter_dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.adapter_dim).permute(0, 3, 1, 2)
        
        if self.op_type != 'sfi' and self.op_type != 'wt':
            x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.adapter_dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class AttentionRouter(nn.Module):
    def __init__(self, dim, num_experts, num_heads=4):
        super().__init__()
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        # expert embeddings (learnable keys)
        self.expert_emb = nn.Parameter(torch.randn(num_experts, dim))

        # projection layers
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        x: (B, N, C)
        returns:
            gates: (B, N, num_experts)
        """
        B, N, C = x.shape

        # Queries from tokens
        Q = self.q_proj(x)          # (B, N, C)

        # Keys from experts
        K = self.k_proj(self.expert_emb)    # (num_experts, C)

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = K.view(self.num_experts, self.num_heads, self.head_dim).transpose(0, 1)  # (H, E, D)

        # Compute attention scores
        attn_scores = torch.einsum("b h n d, h e d -> b h n e", Q, K)  # (B, H, N, E)
        attn_scores = attn_scores * self.scale

        # Merge heads
        attn_scores = attn_scores.mean(dim=1)  # (B, N, E)

        # Softmax over experts
        gates = F.softmax(attn_scores, dim=-1)

        return gates


class Adapter_MoElayer(nn.Module):


    def __init__(self, dim=768, adapter_dim=8,adapter_type=['df', 'ds', 'fd', 'sfi', 'wt'],noisy_gating=True, k=1):
        super(Adapter_MoElayer, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = len(adapter_type)
        self.dim = dim
        self.k = k
        self.identity = nn.Identity()
        self.router = AttentionRouter(dim, self.num_experts, num_heads=4)

        adapter_experts = nn.ModuleList()
        for t in adapter_type:
            adapter_experts.append(Conv2d_Adapter(dim=dim,adapter_dim=adapter_dim,kernel_size=3,stride=1,padding=1,bias=True,op_type=t))

        # define adapter param
        self.num_experts = len(adapter_experts)
        self.adapter_experts = adapter_experts

        assert(self.k <= self.num_experts)


    def forward(self, x, loss_coef=1):
        B, N, C = x.shape

        # Get gates from attention router
        gates = self.router(x)  # (B, N, num_experts)

        # Dispatch tokens to experts
        expert_outputs = []
        for i, expert in enumerate(self.adapter_experts):
            gate = gates[:, :, i].unsqueeze(-1)  # (B, N, 1)
            expert_out = expert(x)  # (B, N, C)
            expert_outputs.append(expert_out * gate)

        # Sum outputs from all experts
        y = torch.stack(expert_outputs, dim=0).sum(dim=0)  # (B, N, C)

        # Uniform usage loss
        load = gates.sum(dim=(0, 1))  # (num_experts,)
        importance = gates.sum(dim=(0, 1))
        eps = 1e-10
        loss = ((importance.var() / (importance.mean() ** 2 + eps))
                + (load.var() / (load.mean() ** 2 + eps))) * loss_coef
        #y = y.reshape(B, 197, self.dim)

        return y, loss

class LoRA_MoElayer(nn.Module):


    def __init__(self, dim, lora_dim=[8,16,32,48,64,96,128], noisy_gating=True, k=1): #
        super(LoRA_MoElayer, self).__init__()

        self.noisy_gating = noisy_gating
        self.k = k

        # instantiate lora experts
        Lora_a_experts = nn.ModuleList()
        Lora_b_experts = nn.ModuleList()
        for i,d in enumerate(lora_dim):
            Lora_a_experts.append(nn.Linear(dim, d,bias = False))
            nn.init.kaiming_uniform_(Lora_a_experts[i].weight, a=math.sqrt(5))
            Lora_b_experts.append(nn.Linear(d, dim*3,bias = False))
            nn.init.zeros_(Lora_b_experts[i].weight)

        Lora_experts = nn.ModuleList()

        # define lora param
        self.num_experts = len(Lora_a_experts)
        self.router = AttentionRouter(dim, self.num_experts, num_heads=4)
        self.Lora_a_experts = Lora_a_experts
        self.Lora_b_experts = Lora_b_experts

        assert(self.k <= self.num_experts)


    def forward(self, x, loss_coef=1):
        B, N, C = x.shape
        gates = self.router(x)  # (B, N, E)

        gates_flat = gates.view(B * N, self.num_experts, 1)  # (B*N, E, 1)
        x_flat = x.view(B * N, C)  # (B*N, C)

        expert_outputs = []
        for i in range(self.num_experts):
            qkv_delta = self.Lora_a_experts[i](x_flat)  # (B*N, d)
            qkv_delta = self.Lora_b_experts[i](qkv_delta)  # (B*N, 3*C)
            qkv_delta = qkv_delta * gates_flat[:, i, :]  # apply gating
            expert_outputs.append(qkv_delta)

        y = torch.stack(expert_outputs, dim=0).sum(dim=0)  # (B*N, 3*C)
        y = y.view(B, N, C * 3)

        # load balancing loss
        load = gates.sum(dim=(0, 1))  # (num_experts,)
        importance = gates.sum(dim=(0, 1))
        eps = 1e-10
        loss = ((importance.var() / (importance.mean() ** 2 + eps)) +
                (load.var() / (load.mean() ** 2 + eps))) * loss_coef
        return y, loss


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    'vit_huge_patch14_224': _cfg(url=''),
    'vit_giant_patch14_224': _cfg(url=''),
    'vit_gigantic_patch14_224': _cfg(url=''),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch8_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,lora_topk=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.LoRA_k = lora_topk

        if self.LoRA_k>0:
            self.LoRA_MoE = LoRA_MoElayer(dim,k=self.LoRA_k)


    def forward(self, x):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # pass through lora_moe
        if self.LoRA_k>0:
            qkv_delta,lora_loss = self.LoRA_MoE(x)
            qkv_delta = qkv_delta.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            q_delta, k_delta, v_delta = qkv_delta.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            q,k,v = q+q_delta,k+k_delta,v+v_delta
        else:
            lora_loss =  torch.zeros(1).to(x.device)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,lora_loss#,adapter_loss


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,init_values=None,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, lora_topk=1, adapter_topk=1, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,lora_topk=lora_topk)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.adapter_k = adapter_topk
        if self.adapter_k>0:
            self.adapter_MoE = Adapter_MoElayer(dim,adapter_dim=8,k=self.adapter_k)

    def forward(self, x):
        x1, lora_loss = self.attn(self.norm1(x))
        x = x + self.drop_path(x1)
        # pass through adapter_moe
        if self.adapter_k>0:
            x_adapter,adapter_loss = self.drop_path(self.adapter_MoE(self.norm2(x)))
            x = x + x_adapter + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            adapter_loss = torch.zeros(1).to(x.device)


        return x,lora_loss,adapter_loss

class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer for ViT-like input """

    def __init__(self, in_dim, activation=None, size=14, ratio=8, ret_att=False):
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # Query Conv (using linear transformation for sequence input)
        self.key_linear1 = nn.Linear(in_dim, in_dim // ratio)
        self.key_linear2 = nn.Linear(in_dim, in_dim // ratio)
        self.key_linear_share = nn.Linear(in_dim // ratio, in_dim // ratio)

        #self.linear1 = nn.Linear(size * size, size * size)  # You may adjust this as per your need
        #self.linear2 = nn.Linear(size * size, size * size)
        self.linear1 = nn.Linear(197, 197)  # You may adjust this as per your need
        self.linear2 = nn.Linear(197, 197)

        # Separated value linear transformations
        self.value_linear1 = nn.Linear(in_dim, in_dim)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_linear2 = nn.Linear(in_dim, in_dim)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
        inputs :
            x : input feature maps ( B X L X D)    (B,197,768)
        returns :
            out : self attention value + input feature
            attention: B X L X L (L is sequence length)
        """
        B, L, D = x.size()

        def _get_att(a, b):
            # Project inputs to a common feature space
            proj_key1 = self.key_linear_share(self.key_linear1(a)).view(B, -1, L).permute(0, 2, 1)  # B, L, L
            proj_key2 = self.key_linear_share(self.key_linear2(b)).view(B, -1, L)  # B, L, L
            energy = torch.bmm(proj_key1, proj_key2)  # B, L, L

            # Calculate attention
            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(energy.permute(0, 2, 1)))  # B, L, L

            return attention1, attention2

        # Calculate attention
        att_y_on_x, att_x_on_y = _get_att(x, y)

        # Value projection
        proj_value_y_on_x = self.value_linear2(y)  # B, L, D
        out_y_on_x = torch.bmm(proj_value_y_on_x.permute(0, 2, 1), att_y_on_x)  # B, L, D
        out_y_on_x = out_y_on_x.view(B, L, D)
        out_x = self.gamma1 * out_y_on_x + x

        proj_value_x_on_y = self.value_linear1(x)  # B, L, D
        out_x_on_y = torch.bmm(proj_value_x_on_y.permute(0, 2, 1), att_x_on_y)  # B, L, D
        out_x_on_y = out_x_on_y.view(B, L, D)
        out_y = self.gamma2 * out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y


class FAL(nn.Module):
    def __init__(self, scale=24, margin=0.25):
        super(FAL, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, q, p, n):


        sim_p = F.cosine_similarity(q, p)
        sim_n = F.cosine_similarity(q, n)


        alpha_p = F.relu(-sim_p + 1 + self.margin)
        alpha_n = F.relu(sim_n + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin


        logit_p = self.scale * alpha_p * (sim_p - margin_p)
        logit_n = self.scale * alpha_n * (sim_n - margin_n)

        # with fine-grained
        label_p = torch.ones_like(logit_p)
        logit = logit_p - logit_n

        # without fine-grained
        # logit = (logit_p.unsqueeze(1) - logit_n.unsqueeze(0)).flatten()
        # label_p = torch.ones_like(logit)

        loss = F.binary_cross_entropy_with_logits(logit, label_p)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=q.device)
        return torch.mean(loss)


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



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
            # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',lora_topk=1,adapter_topk=1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                lora_topk=lora_topk, adapter_topk=adapter_topk, layer_id=i)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.norm_middle = norm_layer(embed_dim)
        self.lora_topk=lora_topk
        self.adapter_topk=adapter_topk


        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.freeze_stages()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.srm_a = SRMConv2d(inc=3)

        mlp_hidden_dim = int(embed_dim * 4 * mlp_ratio)
        self.fusion_Mlp = Mlp(in_features=embed_dim * 4, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                              drop=drop_rate)
        self.DCMA_v = DualCrossModalAttention(in_dim=768, ret_att=False)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        normal_(self.level_embed)
        self.spm_srm = SpatialPriorModule_SRM(embed_dim=embed_dim)
        self.spm_srm.apply(self._init_weights)

        self.interaction_indexes = [[0, 3], [4, 7], [8, 11]]

        self.injector = Injector(dim=embed_dim, num_heads=num_heads, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.extractor = Extractor(dim=embed_dim, num_heads=num_heads, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.creterion = FAL()

    def freeze_stages(self):

        self.pos_drop.eval()
        self.patch_embed.eval()

        for block in self.blocks:
            block.eval()
            if self.lora_topk > 0:
                block.attn.LoRA_MoE.train()
            if self.adapter_topk > 0:
                block.adapter_MoE.train()

        for name, param in self.named_parameters():
            if 'LoRA' not in name and 'adapter' not in name and 'head' not in name and 'norm1' not in name:
                param.requires_grad = False

        total_para_nums = 0
        LoRA_para_nums = 0
        adapter_para_nums = 0
        head_para_nums = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_para_nums += param.numel()
                if 'LoRA' in name:
                    LoRA_para_nums += param.numel()
                elif 'head' in name:
                    head_para_nums += param.numel()
                elif 'adapter' in name:
                    adapter_para_nums += param.numel()

        print('parameters:', total_para_nums, 'LoRA', LoRA_para_nums, 'adapter', adapter_para_nums, 'head',
              head_para_nums)


    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)


    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_features(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x, labels, is_train = False):

        x_srm = self.srm_a(x)

        # SPM forward
        c2, c3, c4, s2, s3, s4 = self.spm_srm(x, x_srm)  # c2[B, 784, 768]    c3[B, 196, 768]    c4[B, 49, 768]
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        s2, s3, s4 = self._add_level_embed(s2, s3, s4)
        c = torch.cat([c2, c3, c4], dim=1)  # [B, 1029, 768]
        c_srm = torch.cat([s2, s3, s4], dim=1)  # [B, 1029, 768]

        x_srm = self.forward_features(x)
        x = self.forward_features(x)


        lora_loss_list = []
        adapter_loss_list = []
        lora_loss_list_srm = []
        adapter_loss_list_srm = []
        for i in range(3):
            indexes = self.interaction_indexes[i]
            layer = self.blocks[indexes[0]: indexes[-1] + 1]
            x = self.injector(query=x, feat=c)
            x_srm = self.injector(query=x_srm, feat=c_srm)
            for i, blk in enumerate(layer):
                x ,cur_lora_loss,cur_adapter_loss= blk(x)
                x_srm ,cur_lora_loss_srm,cur_adapter_loss_srm= blk(x_srm)
                lora_loss_list.append(cur_lora_loss)
                adapter_loss_list.append(cur_adapter_loss)
                lora_loss_list_srm.append(cur_lora_loss_srm)
                adapter_loss_list_srm.append(cur_adapter_loss_srm)
                if blk.layer_id % 4 == 0:
                    x, x_srm = self.DCMA_v(x, x_srm)
            c = self.extractor(query=c, feat=x)
            c_srm = self.extractor(query=c_srm, feat=x_srm)
        lora_loss = torch.mean(torch.stack(lora_loss_list))
        adapter_loss = torch.mean(torch.stack(adapter_loss_list))
        moe_loss = lora_loss * 200 + adapter_loss * 1


        lora_loss_srm = torch.mean(torch.stack(lora_loss_list_srm))
        adapter_loss_srm = torch.mean(torch.stack(adapter_loss_list_srm))
        moe_loss_srm = lora_loss_srm * 200 + adapter_loss_srm * 1

        x = self.norm(c)
        x = x[:, 0]
        x_srm = self.norm(c_srm)
        x_srm = x_srm[:, 0]

        out = []
        out_rgb = self.head(x)
        out.append(out_rgb)
        out_srm = self.head(x_srm)
        out.append(out_srm)

        out = torch.mean(torch.stack(out, dim=0), dim=0)
        

        if is_train:
            B, L = x.shape
            fake_feat, real_feat = self.get_p_and_n(x, labels)
            fake_feat_srm, real_feat_srm = self.get_p_and_n(x_srm, labels)
            #print(fake_feat.shape)
            #print(real_feat.shape)
            #fake_feat, real_feat = torch.split(x, B // 2, dim=0)
            real_prototype = self.head.weight[:1].expand(fake_feat.shape[0],-1)
            fal_loss = self.creterion(real_prototype,real_feat,fake_feat)
            fal_loss_srm = self.creterion(real_prototype, real_feat_srm, fake_feat_srm)

            return out, moe_loss, moe_loss_srm, fal_loss, fal_loss_srm

        return out, moe_loss, moe_loss_srm

    def get_p_and_n(self, x, x_label):
        # 获取真实图像和伪造图像的索引
        real_indices = (x_label == 0).nonzero(as_tuple=True)[0]  # 真实图像索引
        fake_indices = (x_label == 1).nonzero(as_tuple=True)[0]  # 伪造图像索引

        # 获取数量
        real_count = len(real_indices)
        fake_count = len(fake_indices)

        # 确保 p 和 n 的数量相同
        if real_count >= fake_count:
            # 真实图像数量大于等于伪造图像数量，取 fake_count 张真实图像
            selected_real_indices = real_indices[:fake_count]
            p = x[fake_indices]  # 伪造图像
            n = x[selected_real_indices]  # 真实图像
        else:
            # 伪造图像数量大于真实图像数量，取 real_count 张伪造图像
            selected_fake_indices = fake_indices[:real_count]
            p = x[selected_fake_indices]  # 伪造图像
            n = x[real_indices]  # 真实图像

        return p, n



def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_sam_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=0, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_sam_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_sam_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=0, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_sam_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    """ ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """ ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        'deit_small_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'deit_base_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model


if __name__ == '__main__':
    model = vit_base_patch16_224_in21k(pretrained=True,num_classes=2)
    x = torch.rand(32,3,224,224)
    y,_ = model(x)
    print(y.shape)

