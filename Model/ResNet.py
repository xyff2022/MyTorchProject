from typing import List, Optional

import torch
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Identity, MaxPool2d, Sequential


class ShortCut(Module):
    # 快捷链接
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, 1, stride, 0, bias= False)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn( self.conv(x) )


class BottleneckResidualBlock(Module):
    # 瓶颈残差块
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride):
        super().__init__()
        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, 1)
        self.bn1 = BatchNorm2d(bottleneck_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, 3, stride, 1)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, 1)
        self.bn3 = BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            # 修正输入的x的规格！
            self.shortcut = ShortCut(in_channels, out_channels, stride)
        else:
            self.shortcut = Identity()
        # Sequential()为容器，无法进行判断，所以先创造一个空容器，然后进行判断添加卷积操作
        # 也可选择用nn.identity来“恒等”

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out + shortcut)
        return out


class ResidualBlock(Module):
    # 常规残差块
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = Conv2d(in_channels ,out_channels, 3, stride, 1)
        # 第一个3*3卷积进行特征提取(由卷积核完成)和尺寸缩减(下采样)
        # 下采样只让第一个卷积核选择参与(由传入不为1的stride控制(通常为2))
        # stride=2时,(hin-1)/2+1=hout,小数全舍去即可实现输出减半
        # 如果卷积后归一化，偏执bias可以设置False节省计算资源
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1,1)
        # Conv2d(in_channels, out_channels, 3, 1, 1)不改变图像大小
        self.bn2 = BatchNorm2d(out_channels)
        self.relu = ReLU()

        if stride != 1 or in_channels != out_channels :
            self.shortcut = ShortCut(in_channels, out_channels, stride)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + shortcut)
        return out


class ResNet(Module):
    def __init__(self,
                 n_blocks: List[int],
                 # 网络中进行残差处理的阶段和各阶段有几个残差块，如ResNet-18 中，n_blocks 就是 [2, 2, 2, 2]
                 n_channels: List[int],
                 # 对应阶段的输出通道数，如标准的 ResNet-18 中，n_channels 对应的是 [64, 128, 256, 512]
                 # 提示传入参数为int型List
                 img_channels: int = 3,
                 # 第一次卷积输入图片的通道数
                 first_kernel_size: int = 7,
                 # 第一次卷积时卷积核大小
                 bottlenecks: Optional[List[int]] = None
                 # 作用: Optional 用来表示一个变量既可以是某种指定的类型，也可以是 None。
                 ):
        super().__init__()


        # assert 代码是灵活的 “安全卫士”。在正式开始构建网络之前，严格地检查了用户传入的配置参数是否匹配。
        # n_block的长度义了ResNet每个阶段有多少个残差块，n_channels定义了ResNet每个阶段的输出通道数
        # 如 n_blocks=[2, 2, 2, 2] (长度为4) 和 n_channels=[64, 128, 256] (长度为3)，程序就会在这里立刻停止
        assert len(n_channels) == len(n_blocks)
        # 如 bottlenecks=[64, 128] (长度为2) 和 n_channels=[64, 128, 256] (长度为3)，都不满足，程序就会在这里停止
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        # 初始处理 - 快速降维与浅层特征提取
        self.conv = Conv2d(img_channels, n_channels[0], first_kernel_size, 2, padding= first_kernel_size // 2)
        self.bn = BatchNorm2d(n_channels[0])
        self.maxpool = MaxPool2d(3,2,1)
        # 对于ResNet第一步，先进行一次卷积下采样，随后进行非线性激活和最大池化
        # 目的将原来x*x的输入变为 1/4*x * 1/4x


        blocks = []
        # 初始化一个残差操作列表
        prev_channel = n_channels[0]

        for i, channels in enumerate(n_channels):
            if len(blocks) ==0:
                stride = 2
            else:
                stride = 1

            if bottlenecks :
                blocks.append(BottleneckResidualBlock(prev_channel, bottlenecks[i], channels, stride))
            else:
                blocks.append(ResidualBlock(prev_channel, channels, stride))
            # 需要传入输入通道，即上次的输出通道，之后更新


            prev_channel = channels

            for _ in range(n_blocks[i] - 1):

                if bottlenecks :
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, 1))
                else:
                    blocks.append(ResidualBlock(channels, channels, 1))

        self.blocks = Sequential(*blocks)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.maxpool(out)
        out = self.blocks(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        return out.mean(dim=-1)


if __name__ == '__main__':
    print("--- 正在测试 ResNet-34 ---")
    resnet34_blocks = [3, 4, 6, 3]
    resnet34_channels = [64, 128, 256, 512]

    try:
        model = ResNet(
            n_blocks=resnet34_blocks,
            n_channels=resnet34_channels
            # img_channels 和 first_kernel_size 使用默认值 3 和 7
        )
        print("模型已成功创建！")

        # 创建一个虚拟的输入张量
        # 形状: (批次大小, 通道数, 高, 宽)
        dummy_input = torch.randn(4, 3, 224, 224)

        # 4. 将输入送入模型进行一次前向传播
        output = model(dummy_input)

        # 5. 打印输出的形状以进行验证
        print(f"\n输入尺寸: {dummy_input.shape}")
        print(f"输出尺寸: {output.shape}")

        # 根据ResNet-34的结构，最终输出的特征向量应该是512维
        # 所以期望的输出尺寸是 (批次大小, 512)
        assert output.shape == (4, 512), "输出尺寸不匹配！"
        print("\n模型尺寸验证通过！")

    except Exception as e:
        print(f"\n模型测试失败，错误信息: {e}")
        print("请检查您的 ResNet34 类实现是否有误。")

    print("--- 正在测试 ResNet-50 ---")
    resnet50_blocks = [3, 4, 6, 3]
    resnet50_channels = [256, 512, 1024, 2048]
    resnet50_bottlenecks = [64, 128, 256, 512]

    try:
        model = ResNet(
            n_blocks=resnet50_blocks,
            bottlenecks=resnet50_bottlenecks,
            n_channels=resnet50_channels,

            # img_channels 和 first_kernel_size 使用默认值 3 和 7
        )
        print("模型已成功创建！")

        # 创建一个虚拟的输入张量
        # 形状: (批次大小, 通道数, 高, 宽)
        dummy_input = torch.randn(4, 3, 224, 224)

        # 4. 将输入送入模型进行一次前向传播
        output = model(dummy_input)

        # 5. 打印输出的形状以进行验证
        print(f"\n输入尺寸: {dummy_input.shape}")
        print(f"输出尺寸: {output.shape}")

        # 根据ResNet-50的结构，最终输出的特征向量应该是2048维
        # 所以期望的输出尺寸是 (批次大小, 2048)
        assert output.shape == (4, 2048), "输出尺寸不匹配！"
        print("\n模型尺寸验证通过！")

    except Exception as e:
        print(f"\n模型测试失败，错误信息: {e}")
        print("请检查您的 ResNet34 类实现是否有误。")









