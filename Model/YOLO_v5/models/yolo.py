import math

import torch
import torch.nn
from torch.nn import Module,Conv2d,BatchNorm2d,SiLU,Identity,Sequential,MaxPool2d,Upsample,ModuleList

from .common import Conv, Bottleneck, C3, SPP, Concat

# =====================================================================================
# Part 2: 主模型
#
# 这里负责将第一部分中定义好的所有“零件”组装成一个完整、可以运行的YOLOv5模型。
# =====================================================================================
class Detect(Module):
    """
    YOLOv5的检测头，用于对三个不同尺度的特征图进行最终预测。
    """
    stride = None  # 步长列表，会在YOLOv5类中动态设置
    def __init__(self, number_classes = 20, anchors = (), channels = ()):
        # number_classes: 类别数
        # anchors: 锚框列表
        # channels: 三个检测头输入的通道数列表
        super().__init__()
        self.number_classes = number_classes
        self.number_out = number_classes + 5
        self.number_detect = len(anchors) // 3

        self.number_anchor = len(anchors)


        self.anchors = anchors


        self.m = ModuleList(Conv2d(channel, self.number_out * self.number_detect, 1) for channel in channels)

    def forward(self, x):
        # x 是一个列表，包含三个检测头的输入特征图 [P3, P4, P5]
        for i in range (self.number_detect):
            x[i] = self.m[i](x[i])
            batch_size, _, height, width = x[i].shape
            print(x[i].shape)
            x[i] = x[i].view(batch_size, self.number_detect, self.number_out, height, width).permute(0, 1, 3, 4, 2).contiguous()

        # return x
        return x[0], x[1], x[2]




class YOLOv5(Module):
    def __init__(self, number_classes =20, depth_multiple = 0.33, width_multiple = 0.50,
                 anchors_config=None):
        super().__init__()

        # --- 1. 定义模型缩放系数和辅助函数 ---
        # 这些系数控制了模型的深度（C3模块重复次数）和宽度（卷积通道数）
        if anchors_config is None:
            anchors_config = torch.tensor([
                [12, 16], [19, 36], [40, 28],
                [36, 75], [76, 55], [72, 146],
                [142, 110], [192, 243], [459, 401]
            ], dtype = torch.float32, device = "cuda" if torch.cuda.is_available() else "cpu")
        self.anchor_config = anchors_config
        self.number_classes = number_classes
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        # 根据缩放系数计算实际的层数
        def get_depth(n):
            return max(round(n * self.depth_multiple), 1) if n > 1 else n
            # n应该被传入一个大于等于1的数

        def get_width(out_channel):
            # 通过数学技巧，强制保证输出结果永远是8的倍数。
            return math.ceil(out_channel * width_multiple / 8) * 8

        # --- 3. Backbone (骨干网络) ---
        # Backbone负责从输入图像中提取从低级到高级的特征。
        # 初始输入通道为3 (RGB)
        in_channel = 3

        self.p1 = Conv(in_channel, get_width(64), 6, 2, 2)

        self.p2 = Conv(get_width(64), get_width(128), 3, 2, 1)

        self.c3_1_1 = C3(get_width(128), get_width(128), n = get_depth(3))

        self.p3 = Conv(get_width(128), get_width(256), 3, 2, 1)

        self.c3_1_2 = C3(get_width(256), get_width(256), n = get_depth(6))

        self.p4 = Conv(get_width(256), get_width(512), 3, 2, 1)

        self.c3_1_3 = C3(get_width(512), get_width(512), n = get_depth(9))

        self.p5 = Conv(get_width(512), get_width(1024), 3, 2, 1)

        self.c3_1_4 = C3(get_width(1024), get_width(1024), n = get_depth(3))

        self.spp = SPP(get_width(1024), get_width(1024), kernel_size = 5)

        # --- 4. Neck & Head (颈部和头部) ---
        # Neck部分采用FPN+PAN的结构，融合Backbone不同层级的特征，以增强对不同尺寸物体的检测能力。
        self.p6 = Conv(get_width(1024), get_width(512), 1, 1, 0)

        self.up1 = Upsample(scale_factor = 2, mode = "nearest")

        self.concat_1 = Concat(dimension = 1)

        self.c3_2_1 = C3(get_width(512) + get_width(512), get_width(512), n = get_depth(3), shortcut = False)

        self.p7 = Conv(get_width(512), get_width(256), 1, 1, 0)

        self.up2 = Upsample(scale_factor = 2, mode = "nearest")

        self.concat_2 = Concat(dimension = 1)

        self.c3_2_2 = C3(get_width(256) + get_width(256), get_width(256), n = get_depth(3), shortcut = False)

        self.p8 = Conv(get_width(256), get_width(256), 3, 2, 1)

        self.concat_3 = Concat(dimension = 1)

        self.c3_2_3 = C3(get_width(256) + get_width(256), get_width(512), n = get_depth(3), shortcut = False)

        self.p9 = Conv(get_width(512), get_width(512), 3, 2, 1)

        self.concat_4 = Concat(dimension = 1)

        self.c3_2_4 = C3(get_width(512) + get_width(512), get_width(1024), n = get_depth(3), shortcut = False)

        # --- 5. Detect Head (检测头) ---
        detect_channels = [get_width(256), get_width(512), get_width(1024)]
        self.detect = Detect(self.number_classes, self.anchor_config, detect_channels)

        self.detect.stride = torch.tensor([8., 16., 32.])

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.c3_1_1(x)
        x = self.p3(x)
        x = self.c3_1_2(x)
        concat_x_1 = x

        x = self.p4(x)
        x = self.c3_1_3(x)
        concat_x_2 = x

        x = self.p5(x)
        x = self.c3_1_4(x)
        x = self.spp(x)
        x = self.p6(x)
        concat_x_3 = x

        x = self.up1(x)
        x = self.concat_1((x, concat_x_2))
        x = self.c3_2_1(x)
        x = self.p7(x)
        concat_x_4 = x
        x = self.up2(x)
        x = self.concat_2((x, concat_x_1))
        x = self.c3_2_2(x)
        detect_1 = x
        x = self.p8(x)
        x = self.concat_3((x, concat_x_4))
        x = self.c3_2_3(x)
        detect_2 = x
        x = self.p9(x)
        x = self.concat_4((x, concat_x_3))
        detect_3 = self.c3_2_4(x)

        return self.detect([detect_1, detect_2, detect_3])
        # return detect_1, detect_2, detect_3


# --- 测试代码 ---
# 当你直接运行这个文件时，以下代码会被执行，用于快速验证模型是否能成功构建和运行。
if __name__ == '__main__':
    # 创建一个随机的输入张量，模拟一张640x640的RGB图片
    # (batch_size=1, channels=3, height=640, width=640)
    dummy_input = torch.randn(1, 3, 640, 640)

    # 实例化模型，假设有80个类别 (同COCO数据集)
    model = YOLOv5()

    # 将模型设置为评估模式，这会关闭BatchNorm和Dropout等层的训练行为
    model.eval()

    # 执行前向传播
    outputs = model(dummy_input)

    # 打印每个检测头的输出形状，以验证模型结构和数据流是否正确
    print("YOLOv5s model created successfully!")
    print("-" * 40)
    # P3/8 head, grid size 80x80
    print(f"Output from P3/8 head shape: {outputs[0].shape}")
    # P4/16 head, grid size 40x40
    print(f"Output from P4/16 head shape: {outputs[1].shape}")
    # P5/32 head, grid size 20x20
    print(f"Output from P5/32 head shape: {outputs[2].shape}")
    print("-" * 40)
    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f} M")