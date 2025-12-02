import builtins

import torch
import torch.nn
from torch.nn import Module,Conv2d,BatchNorm2d,SiLU,Identity,Sequential,MaxPool2d

# =====================================================================================
# Part 1: 基础模块
#
# 这里定义了构成整个YOLOv5网络的所有基础“零件”或“积木”。
# =====================================================================================

class Conv(Module):
    # 'c1' 是输入通道数 (input channels)
    # 'c2' 是输出通道数 (output channels)
    # 'k' 是卷积核大小 (kernel size)
    # 's' 是步长 (stride)
    # 'p' 是填充 (padding)，可以自动计算
    # 'g' 是分组卷积 (groups)
    # 'act' 控制是否使用激活函数
    def __init__(self, in_channel, out_channel, kernel_size = 1, stride = 1, padding = None, groups = 1, act = True):
        super().__init__()
        # 1. 定义卷积层
        #    bias=False 是因为后面的BatchNorm层包含了偏置参数，可以起到同样的效果
        self.conv = Conv2d(in_channel, out_channel, kernel_size, stride, self.auto_padding(kernel_size, padding), groups = groups, bias = False)
        self.bn = BatchNorm2d(out_channel)
        self.act = SiLU() if act is True else (act if isinstance(act, Module) else Identity())


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


    def auto_padding(self, kernel_size ,padding = None):
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else  [x // 2 for x in kernel_size]
        return padding


class Bottleneck(Module):
    def __init__(self, in_channel, out_channel, shortcut = True, groups = 1, e = 0.5):
        super().__init__()
        channel_ = int(out_channel * e)  # 隐藏层通道数
        self.cv1 = Conv(in_channel, channel_, 1, 1)
        self.cv2 = Conv(channel_, out_channel, 3, 1, groups = groups)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(Module):
    """
        CSP Bottleneck with 3 convolutions (跨阶段局部网络)
        这是YOLOv5的Backbone和Neck中的核心模块。
    """
    def __init__(self, in_channel, out_channel, n = 1, shortcut = True, groups = 1, e = 0.5):
        super().__init__()
        channel_ = int(out_channel * e)
        self.cv1 = Conv(in_channel, channel_, 1, 1)
        self.cv2 = Conv(in_channel, channel_, 1, 1)
        self.cv3 = Conv(2 * channel_, out_channel, 1, 1)
        self.m = Sequential( *(Bottleneck(channel_, channel_, shortcut=shortcut, e = 1) for _ in range(n)))
        # 降维由c3的入口进行，此处不需要降维

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(Module):
    """
    Spatial Pyramid Pooling - Fast version (空间金字塔池化-快速版)
    用于增大感受野，提取多尺度特征。
    """
    def __init__(self, in_channel, out_channel, kernel_size = 5):
        super().__init__()
        channel_ = in_channel // 2
        self.cv1 = Conv(in_channel, channel_, 1, 1)
        self.cv2 = Conv(channel_ * 4, out_channel, 1, 1)
        self.pool = MaxPool2d(kernel_size, 1, kernel_size // 2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.pool(x1)
        x3 = self.pool(x2)

        return self.cv2(torch.cat((x1, x2, x3, self.pool(x3)), dim=1))

class Concat(Module):
    """
    传入一个tuple[Tensor, ...] | list[Tensor] | None
    """
    def __init__(self, dimension = 1):
        super().__init__()
        self.dim = dimension

    def forward(self, x):
        return torch.cat(tensors = x, dim = self.dim)



