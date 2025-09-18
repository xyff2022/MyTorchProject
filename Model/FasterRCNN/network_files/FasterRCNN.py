import torch
from torch import nn

from Model.FasterRCNN.network_files.Anchors import AnchorGenerator
from Model.FasterRCNN.network_files.Backbone import ResNet50Backbone
from Model.FasterRCNN.network_files.DetectorHead import DetectorHead
from Model.FasterRCNN.network_files.ProposalCreator import ProposalCreator
from Model.FasterRCNN.network_files.RegionProposalNetwork import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(self, backbone: ResNet50Backbone,
                 rpn: RegionProposalNetwork,
                 detector_head: DetectorHead):
        super().__init__()
        """
                初始化完整的 Faster R-CNN 模型。

                Args:
                    backbone (nn.Module): 主干网络。
                    rpn (nn.Module): 区域提议网络。
                    detector_head (nn.Module): 检测头。
        """

        self.backbone = backbone
        self.rpn = rpn
        self.detector_head = detector_head

        # --- 非神经网络模块也在这里实例化 ---
        # 锚点生成器
        self.anchor_generate = AnchorGenerator()
        self.proposal_creator_train = ProposalCreator('training')
        self.proposal_creator_validation = ProposalCreator('validation')


    def forward(self, image, targets=None):
        """
            Faster R-CNN的前向传播。

            Args:
                image(torch.Tensor): 输入的图片张量，形状为[B, C, H, W]。
                targets: 训练时使用，包含真实边界框和标签等信息(暂时不用)。
        """

        # --- 步骤 6.3.1：特征提取 ---
        # 将图片送入主干网络，得到特征图
        # 输入: [B, 3, 800, 800]
        # 输出: [B, 1024, 50, 50]
        feature_map = self.backbone(image)

        # --- 步骤 6.3.2：生成候选区域 (Proposal Generation) ---
        # 生成锚点
        # 获取特征图尺寸，用于生成对应数量的锚点
        B, _, H, W =feature_map.shape
        image_size = image.shape[2:]
        anchors = self.anchor_generate.generate((H, W), device=image.device)

        # --- 步骤 6.3.3：RPN 预测 ---
        # 将特征图送入 RPN，得到原始的偏移量和分数
        # 输入:   [1, 1024, 50, 50]
        # 输出:   rpn_locs [1, 22500, 4], rpn_scores [1, 22500, 2],
        #        rpn_losses{"rpn_cls_losses","rpn_loc_losses"} or None
        rpn_locs, rpn_scores, rpn_losses = self.rpn(feature_map, anchors, image_size, targets)

        # --- 步骤 6.3.4: 选择 ProposalCreator 并生成提议 ---
        # self.training 是 nn.Module 自带的属性，model.train()时为True, model.eval()时为False
        if self.training:
            proposal_creator =self.proposal_creator_train
        else:
            proposal_creator = self.proposal_creator_validation

        # 循环为批次中的每张图片生成候选区域
        proposals = []

        # rpn_locs: [B, 22500, 4] -> [22500, 4]
        # rpn_scores: [B, 22500, 2] -> [22500, 1] (只取前景分)
        for i in range(B):
            # 将 RPN 的原始预测值解码成候选区域
            image_proposals = proposal_creator(rpn_locs[i],
                                        rpn_scores[i][:, 1],
                                        anchors,
                                        image.shape[2:])
            proposals.append(image_proposals)


        # --- 步骤 6.3.4：通过检测头进行最终预测 ---
        # 将特征图和 input 送入检测头,pool层在检测头里
        # 输入: feature_map [B, 1024, 50, 50], boxes [N, 5]
        # 输出: cls_scores [N, 21], bbox_offsets [N, 84]
        # N: 这是输入检测头的 RoI (候选区域) 的数量。这个值在训练和预测时是不同的：
        #   在训练时, N 等于 ProposalTargetCreator 采样出的数量（例如 128）。
        #   在预测时, N 等于 ProposalCreator 生成的候选区域数量（例如 2000）。
        detections, losses = self.detector_head(feature_map, proposals, image_size, targets)

        Loss = {}
        if self.training:
            Loss.update(rpn_losses)
            Loss.update(losses)

        return Loss if self.training else detections


# --- 用于测试本模块完整功能的脚本 ---
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
    # *** 修改：forward现在返回3个值，测试时只取前两个 ***
    rpn_locs, rpn_scores, _ = rpn(dummy_feature_map, torch.randn(22500, 4), (800, 800))


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