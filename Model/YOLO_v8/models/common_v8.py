import builtins

import torch
import torch.nn
from torch.nn import Module, Conv2d, BatchNorm2d, SiLU, Identity, Sequential, MaxPool2d, ModuleList, Parameter


# =====================================================================================
# Part 1: 基础模块
#
# 这里定义了构成整个YOLOv8网络的所有基础“零件”或“积木”。
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


# --------------------------------------------------
# 模块 2: Bottleneck 模块
# --------------------------------------------------

class Bottleneck(Module):
    def __init__(self, in_channel, out_channel, shortcut = True, groups = 1, e = 0.5):
        super().__init__()
        channel_ = int(out_channel * e)  # 隐藏层通道数
        self.cv1 = Conv(in_channel, channel_, 1, 1)
        self.cv2 = Conv(channel_, out_channel, 3, 1, groups = groups)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# --------------------------------------------------
# 模块 3: C2f 模块 (v8 核心)
# --------------------------------------------------
class C2f(Module):
    def __init__(self, in_channel, out_channel, n = 1, shortcut = False, g = 1, e = 0.5):
        super().__init__()
        # 计算隐藏层的通道数
        self.c = int(out_channel * e)

        # 入口 1x1 卷积，将通道数 c1 变为 2 * c
        # (e.g., c1=64 -> 2 * 64 = 128)
        self.cv1 = Conv(in_channel, 2 * self.c, 1, 1)

        # 定义 n 个 Bottleneck 模块
        self.moduleList = ModuleList(Bottleneck(self.c, self.c, shortcut, g, e = 1) for _ in range(n))

        # 出口 1x1 卷积
        # 它的输入通道数是 (2 + n) * c
        self.cv2 = Conv((n + 2) * self.c, out_channel, 1)

    def forward(self, x):
        # 1. 拆分:
        #    输入 x (bs, in_channel, h, w) 通过 cv1 变为 (bs, 2*self.c, h, w)
        #    .split() 将其在通道维度上拆分为两个 (bs, self.c, h, w) 的张量
        #    y 现在是一个结果列表,现在里面有 [张量A, 张量B]
        y = list(self.cv1(x).split((self.c, self.c), 1))

        # 2. 扩展:
        #    我们遍历 self.m 中的 n 个 Bottleneck 模块,每一个模块是一层,我们输入 输入 来获得 输出
        #    y[-1] 指的是列表 y 的最后一个元素
        #    y.extend(...) 会把每个模块 m 的输出都追加到列表 y 的末尾
        #    (e.g., n=3, 循环后 y = [A, B, C, D, E])
        for module in self.moduleList:
            y.append(module(y[-1]))

        # 3. 拼接: (bs, (n+2)*self.c, h, w)
        # 4. 融合: (bs, out_channel, h, w)
        return self.cv2(torch.cat(y, dim = 1))


# --------------------------------------------------
# 模块 4: SPPF 模块 (v8 升级)
# --------------------------------------------------
class SPPF(Module):
    # Spatial Pyramid Pooling - Fast (SPPF)
    def __init__(self, c1, c2, k=5):  # ch_in, ch_out, kernel_size
        super().__init__()
        c_ = c1 // 2  # 隐藏层通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # 核心: 连续（串行）的 MaxPool
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x) # 先降维
        with torch.cuda.amp.autocast(enabled=False): # v8 建议在 SPPF 中关闭 amp
            y1 = self.m(x)
            y2 = self.m(y1)
            # 将 4 个不同感受野的特征图拼接
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# --------------------------------------------------
# 模块 5: DFL 模块 (v8 核心)解码器
# --------------------------------------------------

class DFL(Module):
    # DFL (Distribution Focal Loss)
    # c1 就是 reg_max，即模型用来表示一个坐标的通道数，默认为 16
    def __init__(self, c1=16):
        super().__init__()
        # 1. 定义一个 1x1 卷积层
        #    输入通道 c1 (16)，输出通道 1
        #    bias=False 因为我们不需要偏置
        self.conv = Conv2d(c1, 1, 1, bias=False)

        # 2. [关键] 关闭这个卷积层的梯度
        #    我们不希望这个卷积核在训练中被改变，它是一个固定的工具
        for param in self.conv.parameters():
            param.requires_grad = False

        # 3. 创建我们想要的权重
        #    torch.arange(c1) 会创建 [0., 1., 2., ..., 15.]
        x = torch.arange(c1, dtype = torch.float)

        # 4. [关键] 将权重赋值给卷积核
        #    卷积核权重的标准形状是 (out_ch, in_ch, kH, kW)
        #    所以我们把 x (shape [16]) 变为 (1, 16, 1, 1)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)

        # 5. 把 c1 存起来，forward 的时候会用到
        self.c1 = c1

    def forward(self, x):
        # [!!! 更改 !!!] 使用源码中的"压扁" (flatten) 版本
        # x shape 必须是 3D: (bs, 4 * c1, a)
        # a = h * w (所有网格点的总数)
        b, c, a = x.shape

        # 1. 拆分 & 计算 Softmax (获取概率)
        #    x.view(b, 4, self.c1, a)
        #    -> (bs, 4, 16, 6400) 把 64 拆成 4 组，每组 16
        #
        #    .transpose(2, 1)
        #    -> (bs, 16, 4, 6400) 把 16 换到前面，准备计算 softmax
        #
        #    .softmax(1)
        #    -> [核心] 在 dim=1 (16个通道) 上计算 softmax
        #       现在这 16 个通道变成了总和为 1 的“概率”
        x = x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)

        # 2. 计算“数学期望” (解码)
        #    x (bs, 16, 4, 6400)
        #    self.conv (in_ch=16, out_ch=1)
        #
        #    用我们固定的 [0...15] 卷积核去“卷”这 16 个概率通道
        #    输出 x 变为 (bs, 1, 4, 6400)
        #    这一步就是在计算 (p0*0 + p1*1 + ... + p15*15)
        #    最后合并
        x = self.conv(x).view(b, 4, a)

        return x

class Concat(Module):
    """
    传入一个tuple[Tensor, ...] | list[Tensor] | None
    """
    def __init__(self, dimension = 1):
        super().__init__()
        self.dim = dimension

    def forward(self, x):
        return torch.cat(tensors = x, dim = self.dim)