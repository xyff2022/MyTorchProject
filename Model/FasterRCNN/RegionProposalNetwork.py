import torch
from torch import nn
from torch.nn import Conv2d, ReLU


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, num_anchors):
        super().__init__()
        """
               区域提议网络 (RPN) 的定义。

               Args:
                   in_channels (int): 输入特征图的通道数。根据你的ResNet-50，这里是 1024。
                   mid_channels (int): 中间层的通道数，通常为 512。
                   num_anchors (int): 每个特征图位置对应的锚点数量，通常是 9。
        """

        # 1. 定义一个3x3的滑窗卷积层。
        #    它的输入通道数是 `in_channels`，输出通道数是 `mid_channels`。
        #    为了保持特征图尺寸不变，kernel_size=3, stride=1, padding=1。
        self.conv = Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.relu = ReLU()

        # 2. 定义一个1x1的分类头。
        #    它的输入通道数是 `mid_channels`。
        #    它的输出通道数需要计算：每个锚点需要2个分数（前景/背景）。
        #    所以总输出通道数是 `num_anchors * 2`。
        self.cls_head = Conv2d(mid_channels, num_anchors * 2, 1)

        # 3. 定义一个1x1的回归头。
        #    它的输入通道数也是 `mid_channels`。
        #    它的输出通道数需要计算：每个锚点需要4个坐标偏移量。
        #    所以总输出通道数是 `num_anchors * 4`。
        self.reg_head = Conv2d(mid_channels, num_anchors * 4, 1)

        # 初始化权重，可以帮助模型更好地收敛
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feature_map):
        # forward方法的目标是接收特征图，输出原始的、但形状规整的位置（locs）和分数（scores）预测。

        # 获取输入的批次大小、高度和宽度，方便后续塑形
        B, C, H, W = feature_map.shape

        # 1. 让特征图首先通过3x3的滑窗卷积和ReLU激活函数
        features = self.relu(self.conv(feature_map))

        # 2. 将得到的中间特征，分别送入回归头和分类头
        rpn_locs = self.reg_head(features)
        rpn_scores = self.cls_head(features)

        # 3. 对输出进行塑形（Reshape），这是非常关键的一步
        #    原始形状: rpn_locs 是 [B, 36, H, W], rpn_scores 是 [B, 18, H, W]
        #    目标形状: rpn_locs 是 [B, H*W*9, 4], rpn_scores 是 [B, H*W*9, 2]
        #    a. 使用 .permute(0, 2, 3, 1) 将通道维度(C)换到最后
        #       形状变为: [B, H, W, 36] 和 [B, H, W, 18]
        rpn_locs = rpn_locs.permute(0, 2, 3, 1)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1)
        #    b. 使用 .contiguous().view() 进行最终塑形
        #       .contiguous() 确保张量在内存中是连续的
        #       .view(B, -1, 4) 会自动计算中间维度的大小 (H*W*9)
        rpn_locs = rpn_locs.contiguous().view(B, -1, 4)
        rpn_scores = rpn_scores.contiguous().view(B, -1, 2)

        return rpn_locs, rpn_scores


# rpn.py 文件末尾的测试代码
if __name__ == '__main__':
    # 假设 backbone 输出的特征图
    dummy_feature_map = torch.randn(1, 1024, 50, 50)

    # RPN 的参数
    in_channels = 1024
    mid_channels = 512
    num_anchors = 9

    # 实例化 RPN
    rpn = RegionProposalNetwork(in_channels, mid_channels, num_anchors)

    # 执行前向传播
    rpn_locs, rpn_scores = rpn(dummy_feature_map)

    # 打印和验证输出形状
    print("--- 正在测试 RPN __init__ 和 forward 方法 ---")
    print(f"输入特征图的形状: {dummy_feature_map.shape}")
    print(f"输出 rpn_locs 的形状: {rpn_locs.shape}")
    print(f"输出 rpn_scores 的形状: {rpn_scores.shape}")

    expected_locs_shape = (1, 50 * 50 * 9, 4)
    expected_scores_shape = (1, 50 * 50 * 9, 2)

    assert rpn_locs.shape == expected_locs_shape, "rpn_locs 形状不匹配!"
    assert rpn_scores.shape == expected_scores_shape, "rpn_scores 形状不匹配!"

    print("\nRPN 的 __init__ 和 forward 方法测试通过！")


