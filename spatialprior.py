# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

class LearnableHFS(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(3.0))  # 可学习的标准差参数

    def forward(self, x):
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=[-2, -1])
        b, c, h, w = x_fft.shape
        
        # 生成高斯掩码
        center_h, center_w = h//2, w//2
        y_coords = torch.arange(h, device=x.device).float() - center_h
        x_coords = torch.arange(w, device=x.device).float() - center_w
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        
        sigma = torch.clamp(self.sigma, min=1.0)  # 防止sigma过小
        mask = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        mask = mask.view(1, 1, h, w).expand_as(x_fft)
        
        x_fft = x_fft * (1 - mask)  # 抑制低频区域
        x_fft = torch.fft.ifftshift(x_fft, dim=[-2, -1])
        x = torch.fft.ifft2(x_fft, norm="ortho")
        return F.relu(torch.real(x), inplace=True)

class LearnableHFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        x_fft = torch.fft.fft(x, dim=1, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=1)
        b, c, h, w = x_fft.shape
        
        # 通道维度高斯掩码
        center_c = c // 2
        c_coords = torch.arange(c, device=x.device).float() - center_c
        sigma = torch.clamp(self.sigma, min=1.0)
        mask = torch.exp(-c_coords**2 / (2 * sigma**2))
        mask = mask.view(1, c, 1, 1).expand_as(x_fft)
        
        x_fft = x_fft * (1 - mask)
        x_fft = torch.fft.ifftshift(x_fft, dim=1)
        x = torch.fft.ifft(x_fft, dim=1, norm="ortho")
        return F.relu(torch.real(x), inplace=True)


class SpatialPriorModule_SRM(nn.Module):
    def __init__(self, inplanes=64, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        
        # 可学习的频域滤波
        self.hfs = LearnableHFS()
        self.hfc = LearnableHFC()
        
        # 注意力模块
        self.se1 = self._make_se_layer(64)
        self.se2 = self._make_se_layer(128)
        self.se3 = self._make_se_layer(256)
        self.se4 = self._make_se_layer(512)
        
        # 增强的特征映射
        self.fc2 = nn.Sequential(
            nn.Conv2d(128, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.fc3 = nn.Sequential(
            nn.Conv2d(256, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.fc4 = nn.Sequential(
            nn.Conv2d(512, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    
    def _make_se_layer(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels//16, 4), 1),
            nn.ReLU(),
            nn.Conv2d(max(channels//16, 4), channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        bs = x.shape[0]
        
        # 处理原始图像
        c1 = self.stem(x)
        
        # 增强的特征融合
        hfs_c1 = self.hfs(c1)
        hfc_c1 = self.hfc(c1)
        s1 = self.stem(y) + self.se1(hfs_c1 + hfc_c1) * (hfs_c1 + hfc_c1)
        
        # 后续处理
        c2 = self.conv2(c1)
        s2 = self.conv2(s1) + self.se2(self.hfs(c2) + self.hfc(c2)) * (self.hfs(c2) + self.hfc(c2))
        
        c3 = self.conv3(c2)
        s3 = self.conv3(s2) + self.se3(self.hfs(c3) + self.hfc(c3)) * (self.hfs(c3) + self.hfc(c3))
        
        c4 = self.conv4(c3)
        s4 = self.conv4(s3) + self.se4(self.hfs(c4) + self.hfc(c4)) * (self.hfs(c4) + self.hfc(c4))
        
        # 调整维度
        c2 = self.fc2(c2).view(bs, self.embed_dim, -1).transpose(1, 2)
        c3 = self.fc3(c3).view(bs, self.embed_dim, -1).transpose(1, 2)
        c4 = self.fc4(c4).view(bs, self.embed_dim, -1).transpose(1, 2)
        
        s2 = self.fc2(s2).view(bs, self.embed_dim, -1).transpose(1, 2)
        s3 = self.fc3(s3).view(bs, self.embed_dim, -1).transpose(1, 2)
        s4 = self.fc4(s4).view(bs, self.embed_dim, -1).transpose(1, 2)
        
        return c2, c3, c4, s2, s3, s4




class Injector(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.0):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)

        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):
        attn = self.self_attn(self.query_norm(query), self.feat_norm(feat), value=self.feat_norm(feat))[0]
        return query + self.gamma * attn


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)

    def forward(self, query, feat):
        attn = self.self_attn(self.query_norm(query), self.feat_norm(feat), value=self.feat_norm(feat))[0]
        query = query + attn

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.injector = Injector(dim=dim, num_heads=num_heads, norm_layer=norm_layer)
        self.extractor = Extractor(dim=dim, num_heads=num_heads, norm_layer=norm_layer)
        self.attn_blk = [7, 8, 9, 10, 11]
        self.feet_blk = 6

    def forward(self, x, c, blocks):
        x = self.injector(query=x, feat=c)
        for idx, blk in enumerate(blocks):
            x, atten = blk(x)
        c = self.extractor(query=c, feat=x)

        return x, c
