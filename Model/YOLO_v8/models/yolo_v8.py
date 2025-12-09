import math
from typing import List

import torch
import torch.nn
from torch import meshgrid
from torch.nn import Module, Conv2d, BatchNorm2d, SiLU, Identity, Sequential, MaxPool2d, Upsample, ModuleList

from .common_v8 import Conv, Bottleneck, C2f, SPPF, Concat, DFL


class Detect(Module):
    # YOLOv8 解耦检测头 (Decoupled-Head)

    # 我们把 DFL 的 reg_max=16 定义为这个类的属性
    reg_max = 16

    # n_classes = 类别数 (e.g., 20)
    # in_channels = Neck 输出的 P3, P4, P5 的通道列表, e.g., (256, 512, 1024)
    def __init__(self, n_classes:int=20, in_channels:List[int]=[]):
        super().__init__()
        self.nc = n_classes  # 类别数 (e.g., 20)
        self.nl = len(in_channels)  # 检测层数 (nl=3, 对应 P3, P4, P5)
        self.no = self.nc + self.reg_max * 4  # 总输出通道数 = 分类 + 回归

        # 定义一个空的 stride 张量，它会在 YOLOv8 模型组装时被填入
        self.stride = torch.zeros(self.nl)

        # 定义两个 ModuleList, 分别存放 nl (3) 个回归分支和分类分支
        self.cv2 = ModuleList()  # 回归(Box)分支
        self.cv3 = ModuleList()  # 分类(Cls)分支

        # 计算两个分支的隐藏层通道数
        # (这是一个经验值，让通道数保持合理)
        c2 = max((16, in_channels[0] // 4, self.reg_max * 4))  # 回归分支的隐藏通道
        c3 = max(in_channels[0], self.nc)  # 分类分支的隐藏通道

        # 遍历 P3, P4, P5 (i = 0, 1, 2)
        for i in range(self.nl):
            # --- 构建第 i 层的 回归分支 (Box) ---
            # 它由 2 个 3x3 Conv 和 1 个 1x1 Conv 组成
            self.cv2.append(Sequential(
                Conv(in_channels[i], c2, 3),
                Conv(c2, c2, 3),
                # 最后的 1x1 卷积, 输出 4 * reg_max (64) 个通道
                Conv2d(c2, 4 * self.reg_max, 1)
            ))

            # --- 构建第 i 层的 分类分支 (Cls) ---
            # 它也由 2 个 3x3 Conv 和 1 个 1x1 Conv 组成
            self.cv3.append(Sequential(
                Conv(in_channels[i], c3, 3),
                Conv(c3, c3, 3),
                # 最后的 1x1 卷积, 输出 nc (20) 个通道
                Conv2d(c3, self.nc, 1)
            ))

        # 实例化 DFL 模块 (c1 = reg_max = 16)
        self.dfl = DFL(self.reg_max)

        # 用于存储推理时的锚点和步长，避免重复计算
        self.all_anchors = torch.empty(0)
        self.all_strides = torch.empty(0)

    def forward(self, x):
        # x 是一个列表 [P3, P4, P5]

        # 遍历 P3, P4, P5，分别通过解耦头
        for i in range(self.nl):
            # 1. 回归分支: (bs, ch[i], h, w) -> (bs, 64, h, w)
            box_out = self.cv2[i](x[i])
            # 2. 分类分支: (bs, ch[i], h, w) -> (bs, 20, h, w)
            cls_out = self.cv3[i](x[i])

            # 3. 拼接: (bs, 84, h, w)
            x[i] = torch.cat((box_out, cls_out), 1)

        if self.training:
            # 训练时：直接返回 [P3, P4, P5] 的原始输出
            return x
        else:
            # 推理时：调用 _inference 进行后处理
            inference_out = self._inference(x)
            return inference_out, x

    def _inference(self, x):
        """
            推理逻辑: 解码、生成锚点、坐标转换
        """
        # x 是列表 [ (bs, 84, 80, 80), (bs, 84, 40, 40), ... ]
        shape = x[0].shape

        # 1. 压扁并拼接所有层的输出
        #    xi.view(bs, 84, -1) 把 (bs, 84, 80, 80) 变成 (bs, 84, 6400)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)
        # x_cat 的形状 (bs, 84, 8400)

        # 2. 生成或获取锚点 (Anchor Points) 和 步长 (Strides)
        if self.all_anchors.shape[0] == 0:
            all_anchors, all_strides = self.make_anchors(x, self.stride)

            # [核心优化 1] 立即转置: (N, 2) -> (2, 8400)
            self.all_anchors = all_anchors.transpose(0, 1)

            # [核心优化 2] 立即转置: (N, 1) -> (1, 8400)
            self.all_strides = all_strides.transpose(0, 1)
            # 为了配合特征图 x_cat 的形状 (bs, 84, 8400)
            self.shape = shape


        # 3. 分离 回归分支(box) 和 分类分支(cls)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        # 4. 解码 DFL
        dbox = self.dfl(box)

        # 5. 坐标转换 (distance2bbox)
        #    self.anchors.unsqueeze(0) 形状变为 (1, 2, 8400)，可以直接与 (bs, 2, 8400) 运算
        #    dist2bbox 返回的是 (bs, 4, 8400) 的 xywh
        #    最后乘以 strides (1, 1, 8400)
        bbox = distance2bbox(dbox, self.all_anchors.unsqueeze(0), xywh = True, dim=1) * self.all_strides

        # 7. 拼接并返回 (bs, 8400, 24)
        return torch.cat((bbox, cls.sigmoid()), dim=1).transpose(1, 2)


    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """
        生成锚点 (Anchor Points) 和 步长张量 (Stride Tensor)

        Args:
            feats: 列表，包含三个特征图 [P3, P4, P5]
                   例如 P3 shape: (bs, 84, 80, 80)
            strides: 列表，包含三个步长 [8, 16, 32]
                     例如 P3 步长为 8，意味着特征图上 1 个单位 = 原图 8 个像素
            grid_cell_offset: 网格中心偏移量，0.5 表示中心

        Returns:
            anchor_points: (Total_Anchors, 2) 所有层的网格中心点 (cx, cy)
            stride_tensor: (Total_Anchors, 1) 所有层的步长
        """
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device

        for i, stride in enumerate(strides):
            # 获取当前特征图的宽 (w) 和 高 (h)
            # 例如 P3 层: _, _, 80, 80 = feats[0].shape
            _, _, h, w = feats[i].shape

            # 生成 x 和 y 的坐标序列
            # sx (x 坐标): 生成从 0 到 w-1 的序列，并加上偏移量 0.5。这是每个格子中心的 x 坐标。
            # sy (y 坐标): 生成从 0 到 h-1 的序列，并加上偏移量 0.5。
            sx = torch.arange(0, w, 1, dtype=dtype, device=device) + grid_cell_offset
            sy = torch.arange(0, h, 1, dtype=dtype, device=device) + grid_cell_offset

            # 生成网格矩阵
            # meshgrid (网格化): 将 sx 和 sy 结合，生成一个完整的网格。
            # indexing='ij' 参数很重要，它决定了返回的顺序是 (y, x) 还是 (x, y)。
            # 这里 sy 在前，sx 在后，使用 'ij' 索引，生成的 sy 形状是 (h, w)，sx 形状也是 (h, w)。
            # 此时 sy 矩阵里的值是每一行的 y 坐标，sx 矩阵里的值是每一列的 x 坐标。
            sy, sx = meshgrid(sy, sx, indexing='ij')

            # 堆叠并压扁，得到 (Total_Cells_In_Layer, 2)
            # torch.stack((sx, sy), -1): 在最后一个维度堆叠。形状变为 (h, w, 2)。每个元素都是 [x, y]。
            # .view(-1, 2): 压扁前两个维度。形状变为 (h*w, 2)。
            # 例如 80*80 = 6400，所以这一层会生成 6400 个 (cx, cy) 坐标点。
            anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))

            # 对于这一层的所有 h * w 个点，它们的缩放倍数都是相同的（即当前的 stride）。我们需要创建一个同样长度的张量来存储这个值。
            # torch.full(...): 创建一个形状为 (h*w, 1) 的张量，里面所有的值都填充为 stride（例如 8）。
            stride_tensor.append(torch.full((h * w, 1), stride, device=device, dtype=dtype))

        # 拼接列表中的张量
        return torch.cat(anchor_points), torch.cat(stride_tensor)


def distance2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


# =====================================================================================
# Part 2: YOLOv8 完整模型
# =====================================================================================

class YOLOv8(Module):
    def __init__(self, number_classes=20, depth_multiple=0.33, width_multiple=0.50):
        super().__init__()
        self.number_classes = number_classes

        def get_depth(n): return max(round(n * depth_multiple), 1) if n > 1 else n

        def get_width(out_channel): return int(out_channel * width_multiple)

        self.p1 = Conv(3, get_width(64), 3, 2, 1)
        self.p2 = Conv(get_width(64), get_width(128), 3, 2, 1)
        self.c2f_1 = C2f(get_width(128), get_width(128), n=get_depth(3), shortcut=True)
        self.p3 = Conv(get_width(128), get_width(256), 3, 2, 1)
        self.c2f_2 = C2f(get_width(256), get_width(256), n=get_depth(6), shortcut=True)
        self.p4 = Conv(get_width(256), get_width(512), 3, 2, 1)
        self.c2f_3 = C2f(get_width(512), get_width(512), n=get_depth(6), shortcut=True)
        self.p5 = Conv(get_width(512), get_width(1024), 3, 2, 1)
        self.c2f_4 = C2f(get_width(1024), get_width(1024), n=get_depth(3), shortcut=True)
        self.sppf = SPPF(get_width(1024), get_width(1024), k=5)

        self.up1 = Upsample(scale_factor=2, mode="nearest")
        self.concat_1 = Concat(1)
        self.c2f_neck_1 = C2f(get_width(512) + get_width(1024), get_width(512), n=get_depth(3), shortcut=False)

        self.up2 = Upsample(scale_factor=2, mode="nearest")
        self.concat_2 = Concat(1)
        self.c2f_neck_2 = C2f(get_width(256) + get_width(512), get_width(256), n=get_depth(3), shortcut=False)

        self.p_neck_1 = Conv(get_width(256), get_width(256), 3, 2, 1)
        self.concat_3 = Concat(1)
        self.c2f_neck_3 = C2f(get_width(256) + get_width(512), get_width(512), n=get_depth(3), shortcut=False)

        self.p_neck_2 = Conv(get_width(512), get_width(512), 3, 2, 1)
        self.concat_4 = Concat(1)
        self.c2f_neck_4 = C2f(get_width(512) + get_width(1024), get_width(1024), n=get_depth(3), shortcut=False)

        detect_channels = [get_width(256), get_width(512), get_width(1024)]
        self.detect = Detect(self.number_classes, detect_channels)
        self.detect.stride = torch.tensor([8., 16., 32.])

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p2 = self.c2f_1(p2)
        p3 = self.p3(p2)
        p3 = self.c2f_2(p3)
        p4 = self.p4(p3)
        p4 = self.c2f_3(p4)
        p5 = self.p5(p4)
        p5 = self.c2f_4(p5)
        p5 = self.sppf(p5)

        neck_p4 = self.c2f_neck_1(self.concat_1((self.up1(p5), p4)))
        neck_p3 = self.c2f_neck_2(self.concat_2((self.up2(neck_p4), p3)))
        neck_p4_pan = self.c2f_neck_3(self.concat_3((self.p_neck_1(neck_p3), neck_p4)))
        neck_p5_pan = self.c2f_neck_4(self.concat_4((self.p_neck_2(neck_p4_pan), p5)))

        return self.detect([neck_p3, neck_p4_pan, neck_p5_pan])

# --- 测试代码 ---
if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 640, 640)
    model = YOLOv8()
    model.eval()
    outputs = model(dummy_input)
    print("YOLOv8s model created successfully!")
    print(f"Output shape: {outputs.shape}")