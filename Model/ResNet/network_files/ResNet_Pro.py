# 复现论文
# ResNet strikes back: An improved training procedure in timm


# ResNet v2 论文 "Identity Mappings in Deep Residual Networks"
from typing import Optional, Type, Union, Tuple, Dict, Any
import torch
from torch import nn
from torch.nn import Module, Conv2d, Sequential, AvgPool2d, MaxPool2d

from Model.ResNet.utils.utils import normalize_to_tuple


# def get_padding(kernel_size: int,stride: int, dilation: int = 1):
#     padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
#     return padding


def downsample_conv(
        in_channels : int,
        out_channels : int,
        stride : int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    return Sequential(*[
        Conv2d(in_channels, out_channels, 1, stride, 0, bias = False),
        norm_layer(out_channels)
    ]
    )


def downsample_avg(
        # from <Bag of Tricks for Image Classification with Convolutional Neural Networks>
        in_channels : int,
        out_channels : int,
        stride : int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> Module:
    norm_layer = norm_layer or nn.BatchNorm2d

    return Sequential(*[
        AvgPool2d(2, 2),
        Conv2d(in_channels, out_channels, 1, 1, bias = False),
        norm_layer(out_channels)
    ])


class BottleneckResidualBlock(Module):

    expansion = 4

    def __init__(self,
                 in_channels,
                 planes,
                 stride,
                 downsample: Optional[nn.Module] = None,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 attn_layer: Optional[Type[nn.Module]] = None,
                 aa_layer: Optional[Type[nn.Module]] = None,
                 drop_block: Optional[Type[nn.Module]] = None,
                 drop_path: Optional[nn.Module] = None,
                 reduce_first: int = 1,
    ) -> None :

        """
        1. Type[X]: 代表 X 这个“类”本身，而不是它的实例。
           常用于工厂模式，传入一个“蓝图”，在函数内部用它来创建对象。
           示例: act_layer: Type[nn.Module]  # 传入 nn.ReLU 这个类

        2. Union[X, Y]: 代表一个值“要么是 X 类型，要么是 Y 类型”。
           用于函数能处理多种不同类型的输入。
           示例: block: Union[BasicBlock, Bottleneck] # 既能接收BasicBlock，也能接收Bottleneck

        3. Optional[X]: 代表一个值“要么是 X 类型，要么是 None”。
           它是 Union[X, None] 的简写，专门用于标记可选参数。
           示例: downsample: Optional[nn.Module] # 传入一个nn.Module实例，或者不传(为None)
        """

        """
        Args:
        downsample: Optional[nn.Module] = None,
        下采样
        cardinality: int = 1,
        base_width: int = 64,
        对应ResNeXt
        reduce_first: int = 1,
        对应SENet
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        空洞处理，此处不考虑
        act_layer: Type[nn.Module] = nn.ReLU,
        激活函数
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        归一化
        attn_layer: Optional[Type[nn.Module]] = None,
        注意力层
        aa_layer: Optional[Type[nn.Module]] = None,
        抗锯齿层
        drop_block: Optional[Type[nn.Module]] = None,
        正则化技术，它会随机地将特征图的一个连续矩形区域置零
        drop_path: Optional[nn.Module] = None,
        用于实现随机深度 (Stochastic Depth)，非常有效的正则化技术，训练时以一定概率随机地“跳过”整个残差块
        """

        super().__init__()
        out_channels = planes * self.expansion
        # use_aa = aa_layer is not None and stride == 2

        self.conv1 = Conv2d(in_channels, planes, 1, 1, 0, bias = False)
        self.bn1= norm_layer(planes)
        self.act1 = act_layer(inplace = True)
        self.conv2 = Conv2d(planes, planes, 3, stride ,1, bias = False)
        self.bn2 = norm_layer(planes)
        # self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace = True)
        # self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)
        self.conv3 = Conv2d(planes, out_channels, 1, 1, 0,bias = False)

        self.bn3 = norm_layer(out_channels)

        # self.action = create_attention(attn_layer, out_channels)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = drop_path


    def zero_init_last(self) -> None:
        """Initialize the last batch norm layer weights to zero for better convergence."""
        if getattr(self.bn3, 'weight', None) is not None:
            # getattr(object, name, default)
            # 如果object有'name'属性,getattr就会返回这个属性的值.但没有,程序不会报错崩溃,而是会返回default
            nn.init.zeros_(self.bn3.weight)


    def forward(self, x) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        return self.act3(shortcut + x)



def make_blocks(
        n_channels : Tuple[int, ...],
        block_fns : Tuple[Union[Type[BottleneckResidualBlock]], ...],
        in_channel : int,
        n_blocks: Tuple[int, ...],
        avg_down: bool = False,
        **kwargs


) :
    stage = []
    for stage_idx,(block_fn, planes, num_blocks) in enumerate(zip(block_fns, n_channels, n_blocks)):
        # 使用 zip 将 channels 和 layers 两个列表并行打包。
        # 每次循环，zip会取出一对 (block_fns, planes, num_blocks)，例如 (ResidualBlock,64, 3)。
        # 使用 enumerate 为 zip 的每一对结果添加一个从0开始的索引 stage_idx。如(0,(ResidualBlock,64,3))
        # 最终，每次循环都会自动解包得到 stage_idx, block_fn, planes, num_blocks 四个值。
        stage_name = f'layer{stage_idx + 1}'
        if stage_idx == 0:
            stride = 1
        else:
            stride = 2
        downsample = None
        if stride != 1 or in_channel != planes *block_fn.expansion:
            down_kwargs = dict(
                in_channels = in_channel,
                out_channels = planes * block_fn.expansion,
                stride = stride,
                norm_layer=kwargs.get('norm_layer')
            )
            if avg_down :
                downsample = downsample_avg(**down_kwargs)
            else:
                downsample = downsample_conv(**down_kwargs)

        block_kwargs = dict(**kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            if block_idx == 0:
                downsample = downsample
            else:
                downsample = None
            if block_idx == 0:
                stride = stride
            else:
                stride = 1
            blocks.append(block_fn(
                in_channel,
                planes,
                stride,
                downsample,
                **block_kwargs
            ))
            in_channel = planes * block_fn.expansion
        stage.append((stage_name, Sequential(*blocks)))
    return stage



class ResNet50(Module):
    def __init__(self,
                 block: Union[BottleneckResidualBlock],
                 n_blocks : Tuple[int, ...],
                 in_channels: int = 3,
                 n_channels: Optional[Tuple[int, ...]] = (64, 128, 256, 512),
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 attn_layer: Optional[Type[nn.Module]] = None,
                 aa_layer : Optional[Type[nn.Module]] = None,
                 avg_down: bool = False,
                 block_args: Optional[Dict[str, Any]] = None,

                 ):
        super().__init__()
        block_args = block_args or dict()
        self.conv1 = Conv2d(in_channels, n_channels[0], 7, 2, 3, bias = False)
        self.bn1 = norm_layer(n_channels[0])
        # self.attn = attn_layer()
        self.act1 = act_layer(inplace=True)
        if aa_layer is not None :
            pass
        else :
            self.maxpool = MaxPool2d(3, 2, 1)

        block_fns = normalize_to_tuple(block, len(n_channels))
        stage_modules = make_blocks(
            n_channels,
            block_fns,
            n_channels[0],
            n_blocks,
            avg_down,
            **block_args
        )
        for stage in stage_modules:
            self.add_module(*stage)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x




if __name__ == '__main__':
    print("--- 正在测试 ResNet-50_pro ---")

    resnet50_blocks = (3, 4, 6, 3)
    resnet50_channels = (64, 128, 256, 512)
    in_channels = 3

    try:
        model = ResNet50(
            BottleneckResidualBlock,
            resnet50_blocks,
            in_channels,
            resnet50_channels
        )

        print("模型已成功创建！")
        dummy_input = torch.randn(4, 3, 224, 224)
        out = model(dummy_input)
        print(f"\n输入尺寸: {dummy_input.shape}")
        print(f"输出尺寸: {out.shape}")
        assert out.shape == (4, 2048, 7, 7), "输出尺寸不匹配！"
        print("\n模型尺寸验证通过！")

    except Exception as e:
        print(f"\n模型测试失败，错误信息: {e}")
        print("请检查您的 ResNet34 类实现是否有误。")





