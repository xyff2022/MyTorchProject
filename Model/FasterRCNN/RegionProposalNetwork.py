import torch
from torch import nn
from torch.nn import Conv2d, ReLU

from Model.FasterRCNN import Loss
from Model.FasterRCNN.AnchorTargetCreator import AnchorTargetCreator
from Model.FasterRCNN.ProposalCreator import ProposalCreator


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
        # 定义神经网络层

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

        # 实例化 RPN 自身所需的非网络组件
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_creator_train = ProposalCreator('training')
        self.proposal_creator_validation = ProposalCreator('validation')

        # 初始化权重，可以帮助模型更好地收敛
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feature_map, anchors, image_size, targets = None):
        """
                RPN的前向传播。现在它也负责计算自己的损失，并支持批处理。

                Args:
                    feature_map (torch.Tensor): 主干网络输出的特征图 [B, C, H, W]。
                    anchors (torch.Tensor): 生成的所有锚点 [H*W*9, 4]。
                    image_size (tuple): 原始图片的尺寸 (height, width)。
                    targets (list of dicts, optional): 训练时提供的标准答案，长度为B。

                Returns:
                    tuple:
                        - rpn_locs (torch.Tensor): 原始的位置预测 [B, H*W*9, 4]。
                        - rpn_scores (torch.Tensor): 原始的分数预测 [B, H*W*9, 2]。
                        - losses (dict): 包含RPN损失的字典 (训练时) 或空字典 (推理时)。
        """
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

        # 4. 如果是训练模式，循环计算批次中每张图片的损失
        rpn_losses = {}
        if self.training:
            assert targets is not None, "In training mode, targets must be provided to RPN"

            # --- 为批处理准备损失列表 ---
            rpn_loc_losses, rpn_cls_losses = [], []

            # --- 遍历批次中的每一项 ---
            for i in range(B):
                gt_bboxes = targets[i]["boxes"]

                # 为当前图片生成“标准答案”
                final_labels, final_loc_targets = self.anchor_target_creator(anchors, gt_bboxes, image_size)

                # 为当前图片计算损失
                # 注意我们这里取的是 rpn_locs[i] 和 rpn_scores[i]
                rpn_cls_loss = Loss.rpn_cls_loss(rpn_scores[i], final_labels)
                rpn_loc_loss = Loss.rpn_loc_loss(rpn_locs[i], final_loc_targets, final_labels)

                rpn_loc_losses.append(rpn_loc_loss)
                rpn_cls_losses.append(rpn_cls_loss)

            # --- 汇总整个批次的损失 ---
            # 将列表中的损失求和。除以批次大小B是为了得到平均损失，让学习率更稳定
            rpn_losses = {
                "rpn_cls_losses" : torch.stack(rpn_cls_losses).mean(),
                "rpn_loc_losses" : torch.stack(rpn_loc_losses).mean()
            }


        return rpn_locs, rpn_scores, rpn_losses


# rpn.py 文件末尾的测试代码
if __name__ == '__main__':
    # --- 1. 定义通用测试参数 ---
    dummy_feature_map = torch.randn(2, 1024, 50, 50) # Batch size = 2
    dummy_anchors = torch.randn(22500, 4)
    dummy_image_size = (800, 800)
    dummy_targets = [
        {'boxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32)},
        {'boxes': torch.empty((0, 4), dtype=torch.float32)}
    ]

    rpn = RegionProposalNetwork(in_channels=1024, mid_channels=512, num_anchors=9)

    # --- 2. 测试评估模式 (eval) ---
    print("--- 正在测试 RPN 评估模式 (eval) ---")
    rpn.eval() # 切换到评估模式
    with torch.no_grad():
        # 在评估模式下，不应传入 targets
        rpn_locs, rpn_scores, rpn_losses = rpn(dummy_feature_map, dummy_anchors, dummy_image_size)

    print(f"输出 rpn_locs 的形状: {rpn_locs.shape}")
    print(f"输出 rpn_scores 的形状: {rpn_scores.shape}")
    print(f"返回的损失字典: {rpn_losses}")

    # 验证形状和损失
    expected_locs_shape = (2, 50 * 50 * 9, 4)
    expected_scores_shape = (2, 50 * 50 * 9, 2)
    assert rpn_locs.shape == expected_locs_shape, "评估模式 rpn_locs 形状不匹配!"
    assert rpn_scores.shape == expected_scores_shape, "评估模式 rpn_scores 形状不匹配!"
    assert len(rpn_losses) == 0, "评估模式下，损失字典应为空!"
    print("评估模式测试通过！\n")


    # --- 3. 测试训练模式 (train) ---
    print("--- 正在测试 RPN 训练模式 (train) ---")
    rpn.train() # 切换回训练模式
    # 在训练模式下，必须传入 targets
    rpn_locs, rpn_scores, rpn_losses = rpn(dummy_feature_map, dummy_anchors, dummy_image_size, dummy_targets)

    print(f"输出 rpn_locs 的形状: {rpn_locs.shape}")
    print(f"输出 rpn_scores 的形状: {rpn_scores.shape}")
    print(f"返回的损失字典: {rpn_losses}")

    # 验证形状和损失
    assert rpn_locs.shape == expected_locs_shape, "训练模式 rpn_locs 形状不匹配!"
    assert rpn_scores.shape == expected_scores_shape, "训练模式 rpn_scores 形状不匹配!"
    assert "rpn_cls_losses" in rpn_losses and "rpn_loc_losses" in rpn_losses, "训练模式下，应返回损失!"
    print("训练模式测试通过！")

    print("\n\n--- RPN 模块所有单元测试通过！ ---")
