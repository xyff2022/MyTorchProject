from typing import Union

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class AnchorGenerator:
    """
        根据特征图生成锚点。
    """
    def __init__(self, base_sizes= (128, 256, 512), ratios=(0.5, 1, 2)):
        """
             初始化锚点生成器。

             Args:
                 base_sizes (list of int): 锚点的基础尺寸（像素单位）。
                 ratios (list of float): 锚点的长宽比 (height / width)。
        """
        self.base_sizes = torch.tensor(base_sizes, dtype = torch.float32)
        self.rations = torch.tensor(ratios, dtype = torch.float32)
        self.base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self):
        """
            以(0, 0)为中心，生成9个基础锚点。
            返回一个形状为 [9, 4] 的张量。
        """
        w_ratios = self.base_sizes / torch.sqrt(self.rations)
        h_ratios = w_ratios * self.rations


        w = (w_ratios[:, None] * self.base_sizes[None, :]).view(-1)
        h = (h_ratios[:, None] * self.base_sizes[None, :]).view(-1)

        base_anchors = torch.stack([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h],dim = 1)

        return base_anchors

    def generate(self, feature_map_size, stride = 16, device : Union[str, torch.device] = DEVICE):
        """
                根据特征图尺寸和步长，在原图尺度上生成所有锚点。

                Args:
                    feature_map_size (tuple): (height, width) of the feature map.
                    stride (int): 主干网络的总步长。
                    device (str): 生成的张量所在的设备。

                Returns:
                    torch.Tensor: 形状为 [H * W * 9, 4] 的所有锚点坐标。
        """
        # 0. 获取特征图的高和宽，并将基础锚点移动到正确的设备上
        # H 和 W 分别是特征图的高和宽，比如 (50, 50)
        H, W = feature_map_size
        # 将我们的9个“印章原型”准备好，放到计算设备上（比如GPU）
        base_anchors = self.base_anchors.to(device)

        # 1. 生成网格中心点的 x 和 y 坐标
        #    a. 使用 torch.arange 创建从 0 到 W-1 的序列，再乘以步长 stride
        #    b. 同样地，为 y 坐标创建序列
        shift_x = torch.arange(0, W, device = device) * stride
        shift_y = torch.arange(0, H, device = device) * stride
        # shift_x 会变成[0, 16, 32, ..., 784](一个长度为50的张量)。这定义了所有盖章中心的X坐标。
        # shift_y 会变成[0, 16, 32, ..., 784](一个长度为50的张量)。这定义了所有盖章中心的Y坐标。

        # 2. 使用 torch.meshgrid 创建网格
        #    indexing='ij' 参数确保输出的 y 和 x 的形状与图像习惯一致
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing ='ij')
        # 现在，如果我们取(shift_y[i, j], shift_x[i, j])，就能得到第i行j列那个网格点的精确坐标。


        # 3. 将网格中心点坐标展平并堆叠成 [H*W, 4] 的平移量
        #    a. 使用 .flatten() 将 shift_x 和 shift_y 展平
        #    b. 使用 torch.stack 将它们堆叠成 [H*W, 2] 的张量
        #    c. 使用 .repeat(1, 2) 将其复制成 [H*W, 4] 的形状 (dx, dy, dx, dy)
        shift_x = shift_x.flatten()
        shift_y = shift_y.flatten()
        shift = torch.stack([shift_x, shift_y], dim = 1).repeat(1, 2)


        # 4. 将基础锚点广播到所有网格中心点
        #    a. 将 base_anchors 的形状从 [9, 4] 变为 [1, 9, 4]
        #    b. 将 shifts 的形状从 [H*W, 4] 变为 [H*W, 1, 4]
        #    c. 将两者相加，PyTorch 会自动广播成 [H*W, 9, 4]
        #    d. 最后用 .view(-1, 4) 将其展平为最终的 [H*W*9, 4] 形状
        all_anchors = (base_anchors[None, : ,: ] + shift[:, None, :]).view(-1, 4)

        return all_anchors


if __name__ == '__main__':
    # 创建锚点生成器实例
    anchor_generator = AnchorGenerator()

    # 测试参数
    feat_map_height, feat_map_width = 50, 50
    network_stride = 16

    # 生成锚点
    anchors = anchor_generator.generate((feat_map_height, feat_map_width), stride=network_stride)

    # 验证输出
    expected_num_anchors = feat_map_height * feat_map_width * 9
    print(DEVICE)
    print(f"Feature map size: ({feat_map_height}, {feat_map_width})")
    print(f"Total anchors generated: {anchors.shape[0]}")
    print(f"Expected number of anchors: {expected_num_anchors}")
    print(f"Anchors tensor shape: {anchors.shape}")

    assert anchors.shape[0] == expected_num_anchors, "锚点总数不正确!"
    assert anchors.shape[1] == 4, "锚点维度应为4!"

    print("\n--- 抽样锚点 ---")
    # 打印第一个锚点（对应特征图(0,0)位置的第一个基础锚点）
    print("第一个锚点 (对应特征图位置 0,0):\n", anchors[0])
    # 打印第十个锚点（对应特征图(0,1)位置的第一个基础锚点）
    print("第十个锚点 (对应特征图位置 0,1):\n", anchors[9])

    print("\nAnchorGenerator 类已成功实现并通过测试!")
