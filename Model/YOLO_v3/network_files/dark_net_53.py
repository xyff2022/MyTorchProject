import torch
from torch.nn import Module, Conv2d,BatchNorm2d,LeakyReLU,Sequential

class BasicConv(Module):
    def __init__(self, in_channel, out_channel, kernel_size,stride =1):
        super().__init__()
        self.conv = Conv2d(in_channel, out_channel, kernel_size, stride, padding=kernel_size // 2, bias=False)
        self.bn = BatchNorm2d(out_channel)
        self.activation = LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x



class ResidualBlock(Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = Conv2d(in_channel, in_channel//2, 1)
        self.conv2 = Conv2d(in_channel//2, in_channel, 3)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class DarknetBody(Module):
    def __init__(self, in_channel = 3):
        super().__init__()
        self.conv1 = Conv2d(in_channel, 32, 3, 1)
        # 定义五个大的层块，包含下采样和残差块
        # (通道数, 残差块数量)
        self.layer1 = self._make_layer(32, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 8)
        self.layer4 = self._make_layer(256, 512, 8)
        self.layer5 = self._make_layer(512, 1024, 4)


    def _make_layer(self, in_channel, out_channel, num_block):
        layers = []
        layers.append(BasicConv(in_channel, out_channel, 3, 2))

        for _ in range(num_block):
            layers.append(out_channel)
        return Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)

        # 通过五个层块
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)  # 第一个有效特征图
        out4 = self.layer4(out3)  # 第二个有效特征图
        out5 = self.layer5(out4)  # 第三个有效特征图

        return out3, out4, out5


def darknet53(pretrained_path=None):
    """
    创建 Darknet-53 模型并加载预训练权重。

    Args:
        pretrained_path (str): 预训练权重文件的路径。如果为 None，则不加载权重。

    Returns:
        nn.Module: Darknet-53 模型实例。
    """
    model = DarknetBody()
    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        # 加载预训练权重文件
        # 注意：这里假设权重文件是已经转换为 PyTorch state_dict 格式的 .pth 文件
        # 如果是原始的 .weights 文件，需要一个更复杂的加载脚本。
        # 为简化教学，我们先使用 .pth 文件。
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
    return model
