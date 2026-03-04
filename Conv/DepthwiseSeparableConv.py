import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积（Depthwise Convolution）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        # 点卷积（Pointwise Convolution）
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 示例用法
if __name__ == "__main__":
    # 输入张量：batch_size=1, channels=3, height=32, width=32
    x = torch.randn(1, 3, 32, 32)
    model = DepthwiseSeparableConv(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    output = model(x)
    print("输出形状:", output.shape)
