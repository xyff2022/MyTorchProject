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
if __name__ == '__main__':
    print("--- FasterRCNN 最终组装测试 ---")

    # 为了测试，我们需要在这里手动实例化所有子模块
    from Model.FasterRCNN.Backbone import get_my_resnet50_backbone
    from Model.FasterRCNN.Pooling import RoIPool

    # 1. 实例化所有子模块
    backbone_instance = get_my_resnet50_backbone()
    # 注意：现在 RPN 和 DetectorHead 内部会自己实例化所需的组件
    rpn_instance = RegionProposalNetwork(in_channels=1024, mid_channels=512, num_anchors=9)
    roi_pooler_instance = RoIPool(output_size=7, spatial_scale=1.0 / 16)
    detector_head_instance = DetectorHead(
        num_classes=20,  # 假设是VOC数据集的20个类
        pooler=roi_pooler_instance,
        in_channels=1024
    )

    # 2. 实例化完整的 FasterRCNN 模型
    model = FasterRCNN(
        backbone=backbone_instance,
        rpn=rpn_instance,
        detector_head=detector_head_instance
    )

    print("模型组装成功！")

    # 3. 创建模拟数据 (包含一个没有物体的图片来测试健壮性)
    BATCH_SIZE = 2
    dummy_images = torch.randn(BATCH_SIZE, 3, 800, 800)
    dummy_targets = [
        {'bboxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32),
         'labels': torch.tensor([1], dtype=torch.int64)},
        # 第二张图片是纯背景，没有物体
        {'bboxes': torch.empty((0, 4), dtype=torch.float32),
         'labels': torch.empty((0), dtype=torch.int64)}
    ]

    # 4. 测试训练模式
    print("\n--- 测试训练模式 ---")
    model.train()
    losses = model(dummy_images, dummy_targets)
    print(f"返回的损失字典: {losses}")
    assert "loss_rpn_cls" in losses and "loss_roi_cls" in losses
    print("训练模式前向传播通过！(已处理纯背景图片)")

    # 5. 测试评估模式
    print("\n--- 测试评估模式 ---")
    model.eval()
    with torch.no_grad():
        detections = model(dummy_images)
    print(f"返回了 {len(detections)} 张图片的检测结果。")
    assert len(detections) == BATCH_SIZE
    print("评估模式前向传播通过！")
