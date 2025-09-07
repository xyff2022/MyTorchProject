import torch
from torch import nn

from Model.FasterRCNN.AnchorTargetCreator import AnchorTargetCreator
from Model.FasterRCNN.Anchors import AnchorGenerator
from Model.FasterRCNN.Backbone import ResNet50Backbone
from Model.FasterRCNN.DetectorHead import DetectorHead
from Model.FasterRCNN.Pooling import RoIPool
from Model.FasterRCNN.ProposalCreator import ProposalCreator
from Model.FasterRCNN.ProposalTargetCreator import ProposalTargetCreator
from Model.FasterRCNN.RegionProposalNetwork import RegionProposalNetwork
from Model.ResNet_Pro import BottleneckResidualBlock


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
        # 提议生成器 (在推理时使用)
        # 注意：在训练和推理时，ProposalCreator 的参数不同
        # 我们在这里先创建一个用于推理的实例
        self.proposal_creator_train = ProposalCreator('train')
        self.proposal_creator_validation = ProposalCreator('validation')
        # --- 新增: 实例化两个"标准答案"生成器 ---
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

    def forward(self, image, targets=None):
        """
            Faster R-CNN的前向传播。

            Args:
                image(torch.Tensor): 输入的图片张量，形状为[B, C, H, W]。
                targets: 训练时使用，包含真实边界框和标签等信息(暂时不用)。
        """

        # --- 步骤 6.3.1：特征提取 ---
        # 将图片送入主干网络，得到特征图
        # 输入: [1, 3, 800, 800]
        # 输出: [1, 1024, 50, 50]
        feature_map = self.backbone(image)

        # --- 步骤 6.3.2：RPN 预测 ---
        # 将特征图送入 RPN，得到原始的偏移量和分数
        # 输入: [1, 1024, 50, 50]
        # 输出: rpn_locs [1, 22500, 4], rpn_scores [1, 22500, 2]
        rpn_locs, rpn_scores = self.rpn(feature_map)

        # --- 步骤 6.3.3：生成候选区域 (Proposal Generation) ---
        # 1. 生成锚点
        #    获取特征图尺寸，用于生成对应数量的锚点
        _, _, H, W =feature_map.shape
        anchors = self.anchor_generate.generate((H, W), device=image.device)

        # 2. 选择 ProposalCreator 并生成提议
        #    self.training 是 nn.Module 自带的属性，model.train()时为True, model.eval()时为False
        if self.training:
            proposal_creator =self.proposal_creator_train
        else:
            proposal_creator = self.proposal_creator_validation

        # 我们假设 batch_size=1，所以直接取第0个元素的预测结果
        # rpn_locs: [1, 22500, 4] -> [22500, 4]
        # rpn_scores: [1, 22500, 2] -> [22500, 1] (只取前景分)
        proposals = proposal_creator(rpn_locs[0],
                                     rpn_scores[0][:, 1],
                                     anchors,
                                     image.shape[2:])

        # 3. 准备 RoI Pooling 的输入 input
        #    为 proposals 添加 batch_index 列
        batch_indices = torch.zeros((proposals.shape[0], 1), device=image.device)
        boxes = torch.cat([batch_indices, proposals], dim=1)

        # --- 步骤 6.3.4：通过检测头进行最终预测 ---
        # 将特征图和 input 送入检测头,pool层在检测头里
        # 输入: feature_map [1, 1024, 50, 50], boxes [N, 5]
        # 输出: cls_scores [N, 21], bbox_offsets [N, 84]
        # N: 这是输入检测头的 RoI (候选区域) 的数量。这个值在训练和预测时是不同的：
        #   在训练时, N 等于 ProposalTargetCreator 采样出的数量（例如 128）。
        #   在预测时, N 等于 ProposalCreator 生成的候选区域数量（例如 2000）。
        cls_scores, bbox_offsets = self.detector_head(feature_map, boxes)

        # 在推理阶段，我们通常直接返回这些预测结果
        # 在训练阶段，我们会返回一个包含所有中间结果和最终结果的字典用于计算损失
        # 目前我们先只返回最终结果
        return cls_scores, bbox_offsets


# --- 用于测试本步骤代码的脚本 ---
if __name__ == '__main__':
    print("--- 步骤 6.3.4 测试：实现 forward 方法 - 最终预测 ---")

    # --- 1. 实例化所有子模块 ---
    resnet50_blocks = (3, 4, 6, 3)
    resnet50_channels = (64, 128, 256, 512)
    backbone_instance = ResNet50Backbone(
        BottleneckResidualBlock, resnet50_blocks, 3, resnet50_channels
    )
    rpn_instance = RegionProposalNetwork(in_channels=1024, mid_channels=512, num_anchors=9)
    roi_pooler_instance = RoIPool(output_size=7, spatial_scale=1.0 / 16)

    num_classes = 20
    detector_head_instance = DetectorHead(
        num_classes=num_classes,
        pooler=roi_pooler_instance,
        in_channels=1024
    )

    # --- 2. 实例化完整的 FasterRCNN 模型 ---
    model = FasterRCNN(
        backbone=backbone_instance,
        rpn=rpn_instance,
        detector_head=detector_head_instance
    )
    model.eval()

    # --- 3. 创建假的输入图片并进行前向传播 ---
    dummy_image = torch.randn(1, 3, 800, 800)

    # 调用完整的 forward 方法
    cls_scores, bbox_offsets = model(dummy_image)

    print(f"\n输入图片形状: {dummy_image.shape}")
    print(f"最终分类分数输出形状: {cls_scores.shape}")
    print(f"最终边界框偏移输出形状: {bbox_offsets.shape}")

    # 验证输出形状
    # RoI 的数量在验证模式下最多为 300
    num_rois = cls_scores.shape[0]
    expected_scores_shape = (num_rois, num_classes + 1)
    expected_offsets_shape = (num_rois, (num_classes + 1) * 4)

    assert cls_scores.shape == expected_scores_shape, "分类分数输出形状不匹配!"
    assert bbox_offsets.shape == expected_offsets_shape, "边界框偏移输出形状不匹配!"

    print("\n步骤 6.3.4 测试通过！Faster R-CNN 的完整前向传播已成功实现！")
    print("整个步骤六（整合模型）已成功完成！")




