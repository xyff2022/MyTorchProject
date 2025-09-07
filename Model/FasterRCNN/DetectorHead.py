import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU

from Model.FasterRCNN.Pooling import RoIPool


class DetectorHead(nn.Module):
    def __init__(self,
                 num_classes,
                 pooler,
                 in_channels,
                 hidden_dim = 4096):
        """
                初始化检测头。

                Args:
                    num_classes (int): 物体类别的数量 (不包括背景)。
                    pooler (nn.Module): 我们在步骤四中实现的 RoIPooling 或 RoIAlign 模块。
                    in_channels (int): 输入特征图的通道数 (来自Backbone, e.g., 1024)。
                    hidden_dim (int): 全连接层的隐藏维度，通常为 4096。(暂时不用)
        """
        super().__init__()

        # --- 步骤 5.2.1：存储核心配置参数 ---
        self.num_classes = num_classes
        self.pooler = pooler
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # --- 步骤 5.2.2：定义共享的全连接层“尾部” ---

        # 1. 获取池化层的输出尺寸，例如 (7, 7)
        pool_size = self.pooler.output_size
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)

        # 2. 计算展平后的特征维度
        #    例如：1024 (channels) * 7 (height) * 7 (width) = 50176
        flatten_size = self.in_channels * pool_size[0] * pool_size[1]

        # 3. 使用 nn.Sequential 定义共享的“尾部”
        #    它包含两个全连接层和ReLU激活函数
        self.flatten = Sequential(
            Linear(flatten_size,self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU()
        )

        # --- 步骤 5.2.3：定义分类和回归的“头部” ---

        # 1. 定义分类分支“头部” (Classification Head)
        #    它的输入维度是“尾部”输出的 hidden_dim (4096)
        #    它的输出维度是 num_classes + 1 (例如 20 + 1 = 21)
        self.cls_head = Linear(self.hidden_dim, self.num_classes+1)

        # 2. 定义回归分支“头部” (Bounding Box Regression Head)
        #    它的输入维度也是 hidden_dim (4096)
        #    它的输出维度是 (num_classes + 1) * 4，为每个类别都预测4个坐标偏移
        self.bbox_head = Linear(self.hidden_dim, (self.num_classes + 1) * 4)


    def forward(self, feature_map, boxes):
        """
                检测头的前向传播。

                Args:
                    feature_map (torch.Tensor): 主干网络输出的特征图，形状 [B, C, H, W]。
                    boxes (torch.Tensor): 候选区域，形状 [N, 5]，格式为 (batch_idx, x1, y1, x2, y2)。
        """
        # --- 步骤 5.3.1：池化 ---
        # 直接调用我们在 __init__ 中定义的 roi_pool 或 roi_align 实例
        # 输入: feature_map [1, 1024, 50, 50], boxes [300, 5]
        # 输出: pooled_features [300, 1024, 7, 7]
        pooled_features = self.pooler(feature_map, boxes)

        # --- 步骤 5.3.2：展平与特征提炼 ---
        # 1. 展平池化后的特征
        #    torch.flatten(tensor, start_dim=1) 会将从第1个维度开始的所有维度压平
        #    输入: [300, 1024, 7, 7]
        #    输出: [300, 1024 * 7 * 7], 即 [300, 50176]
        flattened = torch.flatten(pooled_features, 1)


        # 2. 将展平后的特征送入共享的全连接层“尾部”进行提炼
        #    输入: [300, 50176]
        #    输出: [300, 4096(hidden_dim)]
        linear_features = self.flatten(flattened)

        # --- 步骤 5.3.3：最终预测 ---
        # 1. 将提炼后的特征送入分类头
        #    输入: [300, 4096]
        #    输出: [300, 21] (num_classes + 1)
        cls_scores = self.cls_head(linear_features)

        # 2. 将提炼后的特征送入回归头
        #    输入: [300, 4096]
        #    输出: [300, 84] ((num_classes + 1) * 4)
        bbox_offsets = self.bbox_head(linear_features)

        return cls_scores, bbox_offsets


# --- 用于测试本步骤代码的脚本 ---
if __name__ == '__main__':
    # ... (之前的测试代码可以注释掉或删除以保持整洁) ...

    print("\n--- 步骤 5.3.3 测试：实现 forward 方法 - 最终预测 ---")

    # 定义参数
    num_classes = 20
    in_channels = 1024
    hidden_dim = 4096
    batch_size = 1
    num_rois = 300

    # 实例化一个 RoI Pooler
    roi_pooler_instance = RoIPool(output_size=7, spatial_scale=1.0 / 16)
    # 实例化检测头
    detector_head = DetectorHead(
        num_classes=num_classes,
        pooler=roi_pooler_instance,
        in_channels=in_channels,
        hidden_dim=hidden_dim
    )

    # 创建假的输入
    dummy_feature_map = torch.randn(batch_size, in_channels, 50, 50)
    dummy_rois = torch.rand(num_rois, 5) * 800
    dummy_rois[:, 0] = 0

    # 调用完整的 forward 方法
    cls_scores, bbox_offsets = detector_head(dummy_feature_map, dummy_rois)

    print(f"输入 rois 形状: {dummy_rois.shape}")
    print(f"最终分类分数输出形状: {cls_scores.shape}")
    print(f"最终边界框偏移输出形状: {bbox_offsets.shape}")

    # 验证输出形状
    expected_scores_shape = (num_rois, num_classes + 1)
    expected_offsets_shape = (num_rois, (num_classes + 1) * 4)

    assert cls_scores.shape == expected_scores_shape, "分类分数输出形状不匹配!"
    assert bbox_offsets.shape == expected_offsets_shape, "边界框偏移输出形状不匹配!"

    print("\n步骤 5.3.3 测试通过！DetectorHead 的 forward 方法已全部完成！")
    print("整个步骤五（检测头）已成功实现！")
