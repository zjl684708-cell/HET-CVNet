
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    

class SRMConv2d(nn.Module):
    
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        # 2. 对每个通道计算梯度并获取高频部分
        # 初始化梯度幅度列表
        high_freq_list = []

        for c in range(out.shape[1]):  # 遍历每个通道
            # 提取单通道的特征图
            single_channel_feature = out[:, c:c + 1, :, :]  # shape: (Batch, 1, Height, Width)

            # 计算水平梯度（Sobel算子）
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float().to(x.device)
            dx = F.conv2d(single_channel_feature, sobel_x, padding=1)

            # 计算垂直梯度（Sobel算子）
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float().to(x.device)
            dy = F.conv2d(single_channel_feature, sobel_y, padding=1)

            # 计算梯度幅度（即高频噪声的强度）
            magnitude = torch.sqrt(dx ** 2 + dy ** 2)

            # 将高频噪声的单通道幅度保存到列表
            high_freq_list.append(magnitude)

        # 3. 合并所有通道的高频噪声
        high_freq_noise = torch.cat(high_freq_list, dim=1)  # 合并为 (Batch, 3, Height, Width)

        # 4. 可选择：对幅度进行进一步处理（如归一化）
        high_freq_noise = torch.clamp(high_freq_noise, 0, 1)  # 限制范围为 [0, 1]

        # 5. 返回高频噪声部分（可以选择进行可视化或后续处理）
        return high_freq_noise

    def _build_kernel(self, inc):
        # filter1: KB
        filter2 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter1 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 12.
        filter2 = np.asarray(filter2, dtype=float) / 4.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],#, filter1, filter1],
                   [filter2],#, filter2, filter2],
                   [filter3]]#, filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        return filters




