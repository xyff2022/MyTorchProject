import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torchvision.ops import nms

from Model.FasterRCNN import Loss
from Model.FasterRCNN.Pooling import RoIPool
from Model.FasterRCNN.ProposalTargetCreator import ProposalTargetCreator


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

        # --- 为后处理步骤定义超参数 ---
        self.score_thresh = 0.05  # 初步筛选的置信度阈值
        self.nms_thresh = 0.5  # NMS的IoU阈值

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

        # --- 步骤 5.2.4：在 __init__ 中集成 ProposalTargetCreator ---
        # DetectorHead 现在自己负责在训练时采样和分配目标
        self.proposal_target_creator = ProposalTargetCreator()


    def forward(self, feature_map, proposals, image_size, targets = None):
        """
            升级后的 forward 方法，支持批处理。

            Args:
                feature_map (torch.Tensor): 主干网络输出的特征图 [B, C, H, W]。
                proposals (list[torch.Tensor]): 候选区域列表，长度为B。
                targets (list of dicts, optional): 训练时提供的标准答案，长度为B。

            Returns:
                tuple:
                - detections (list[dict]): 包含最终检测结果的列表 (推理时)。
                - losses (dict): 包含检测头损失的字典 (训练时)。
        """
        losses = {}
        if self.training:

            assert targets is not None, "In training mode, targets must be provided to detector head"

            # --- 训练路径：为批次中的每张图采样并分配目标 ---
            image_proposals, image_bboxes, image_labels = [], [], []
            for i in range(len(proposals)):
                # 为当前图片进行采样
                get_image_proposals = proposals[i]
                get_image_bboxes = targets[i]["bboxes"]
                get_image_labels = targets[i]["labels"]

                # proposal_target_creator 返回采样后的 RoI 和它们的目标
                image_roi, image_roi_locs, image_roi_label = (
                    self.proposal_target_creator(get_image_proposals, get_image_bboxes, get_image_labels))

                # 将结果存入列表
                image_proposals.append(image_roi)
                image_bboxes.append(image_roi_locs)
                image_labels.append(image_roi_label)

            # 将列表中的张量合并成一个大的批次
            pool_image_proposals = torch.cat(image_proposals, dim=0)
            image_bboxes = torch.cat(image_bboxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

        else:
            # --- 推理路径：直接使用所有 proposals ---
            pool_image_proposals = torch.cat(proposals, dim = 0)

        # --- 准备 RoI Pooling 的输入 ---
        # 为 rois_for_pooling 添加批次索引
        # 首先，我们需要知道每个 RoI 属于批次中的哪张图片
        roi_index = []
        for i, roi in enumerate(image_proposals if self.training else proposals):
            roi_index.append(torch.full([len(roi)], i, dtype=torch.float32, device= roi.device))
        roi_index = torch.cat(roi_index, dim=0)

        # 组合成 [K, 5] 的格式 (batch_index, x1, y1, x2, y2)
        pooling_bboxes = torch.cat([roi_index.unsqueeze(1), pool_image_proposals], dim=1)


        # --- 步骤 5.4.1：池化 ---
        # 直接调用我们在 __init__ 中定义的 roi_pool 或 roi_align 实例
        # 输入: feature_map [2, 1024, 50, 50], boxes [300, 5]
        # 输出: pooled_features [300, 1024, 7, 7]
        pooled_features = self.pooler(feature_map, pooling_bboxes)

        # --- 步骤 5.4.2：展平与特征提炼 ---
        # 1. 展平池化后的特征
        #    torch.flatten(tensor, start_dim=1) 会将从第1个维度开始的所有维度压平
        #    输入: [300, 1024, 7, 7]
        #    输出: [300, 1024 * 7 * 7], 即 [300, 50176]
        flattened = torch.flatten(pooled_features, 1)


        # 2. 将展平后的特征送入共享的全连接层“尾部”进行提炼
        #    输入: [300, 50176]
        #    输出: [300, 4096(hidden_dim)]
        linear_features = self.flatten(flattened)

        # --- 步骤 5.4.3：最终预测 ---
        # 1. 将提炼后的特征送入分类头
        #    输入: [300, 4096]
        #    输出: [300, 21] (num_classes + 1)
        cls_scores = self.cls_head(linear_features)


        # 2. 将提炼后的特征送入回归头
        #    输入: [300, 4096]
        #    输出: [300, 84] ((num_classes + 1) * 4)
        bbox_offsets = self.bbox_head(linear_features)


        # --- 7. 根据模式返回结果 ---
        if self.training:
            # 训练模式下，ProposalTargetCreator工作生成target用于教会模型如何生成预测
            # 计算损失
            roi_cls_loss = Loss.roi_cls_loss(cls_scores, image_labels)
            roi_loc_loss = Loss.roi_loc_loss(bbox_offsets, image_bboxes, image_labels)
            losses = {"roi_cls_loss" : roi_cls_loss,
                      "roi_loc_loss" : roi_loc_loss}
            return [], losses
        else:
            # 推理模式下，cls_scores和bbox_offsets就是模型
            # 并应用 NMS，然后将结果拆分回列表格式返回。
            # 目前为了保持流程完整，我们暂时返回空的检测结果。
            detections = self.postprocess_detections(cls_scores,
                                                     bbox_offsets,
                                                     pool_image_proposals,
                                                     roi_index,
                                                     image_size)

            return detections, {}

    def postprocess_detections(self,
                               cls_scores,
                               bbox_offsets,
                               pool_image_proposals,
                               roi_indices,
                               image_size):
        """
            对检测头的原始输出进行后处理，以获得最终的检测结果。

            Args:
                cls_scores (torch.Tensor): 形状 [K, num_classes+1] 的原始分数。
                bbox_offsets (torch.Tensor): 形状 [K, (num_classes+1)*4] 的原始偏移量。
                pool_image_proposals (torch.Tensor): 形状 [K, 4] 的 RoI。
                roi_indices (torch.Tensor): 形状 [K] 的批次索引。
                image_size (tuple): (height, width)。

            Returns:
                list[dict]: 每个图像的检测结果列表。
        """
        # --- 步骤 1: 解码边界框 ---
        # a. 重塑偏移量以便于操作
        bbox_offsets = bbox_offsets.view(bbox_offsets.size(0), -1, 4)

        # b. 计算 pool_image_proposals 的中心和宽高
        pool_image_proposals_w = pool_image_proposals[:, 2] - pool_image_proposals[:, 0]
        pool_image_proposals_h = pool_image_proposals[:, 3] - pool_image_proposals[:, 1]
        pool_image_proposals_x = pool_image_proposals[:, 0] + 0.5 * pool_image_proposals_w
        pool_image_proposals_y = pool_image_proposals[:, 1] + 0.5 * pool_image_proposals_h

        # c. 准备广播
        pool_image_proposals_w = pool_image_proposals_w.unsqueeze(1)
        pool_image_proposals_h = pool_image_proposals_h.unsqueeze(1)
        pool_image_proposals_x = pool_image_proposals_x.unsqueeze(1)
        pool_image_proposals_y = pool_image_proposals_y.unsqueeze(1)

        # d. 应用解码公式
        dx = bbox_offsets[..., 0]
        dy = bbox_offsets[..., 1]
        dh = bbox_offsets[..., 2]
        dw = bbox_offsets[..., 3]

        pre_x = dx * pool_image_proposals_w + pool_image_proposals_x
        pre_y = dy * pool_image_proposals_h + pool_image_proposals_y
        pre_w = pool_image_proposals_w * torch.exp(dw)
        pre_h = pool_image_proposals_h * torch.exp(dh)

        # e. 转换回 (x1, y1, x2, y2) 格式
        pre_bboxes = torch.zeros_like(bbox_offsets)
        pre_bboxes[..., 0] = pre_x - 0.5 * pre_w
        pre_bboxes[..., 1] = pre_y - 0.5 * pre_h
        pre_bboxes[..., 2] = pre_x + 0.5 * pre_w
        pre_bboxes[..., 3] = pre_y + 0.5 * pre_h

        # --- 步骤 2: 裁剪边界框到图像范围内 ---
        H, W = image_size
        pre_bboxes[..., 0].clamp_(min=0, max=W)
        pre_bboxes[..., 1].clamp_(min=0, max=H)
        pre_bboxes[..., 2].clamp_(min=0, max=W)
        pre_bboxes[..., 3].clamp_(min=0, max=H)

        # --- 步骤 3: 转换分数为概率 ---
        scores = nn.functional.softmax(cls_scores, dim = 1)

        output_detections = []

        # --- 步骤 4 & 5: 按图片进行过滤和NMS ---
        for index in torch.unique(roi_indices):
            # torch.unique返回不重复的值，即index是图像在batch中的索引
            img_mask = (index == roi_indices)
            img_boxes = pre_bboxes[img_mask]
            img_score = scores[img_mask]
            # 一维索引的长度只要和要索引的张量的第一个维度相同即可进行索引

            final_boxes, final_scores, final_labels = [], [], []

            # 遍历每个类别 (跳过背景类0)
            for i in range(1, self.num_classes + 1):
                score = img_score[:, i]
                bbox = img_boxes[:, i, :]
                # 被:索引的列表代表这一维度所有东西
                # 被一个值（如i）索引，这个维度就会被压缩
                # score形状[k]代表所有预测物体在某一类别的概率表
                # bbox形状[k,21]

                # 按分数阈值过滤
                mask = score > self.score_thresh
                score = score[mask]
                bbox = bbox[mask]

                if score.size(0) == 0:
                    continue

                # 执行NMS
                final_index = nms(bbox, score, self.nms_thresh)

                final_boxes.append(bbox[final_index])
                final_scores.append(score[final_index])
                final_labels.append(torch.full_like(score[final_index], i, dtype=torch.int64))

            # --- 步骤 6: 整理单张图片的最终结果 ---
            if len(final_boxes) > 0:
                one_final_net_bbox = torch.cat(final_boxes, dim = 0)
                one_final_net_scores = torch.cat(final_scores, dim = 0)
                one_final_net_labels = torch.cat(final_labels, dim=0)
            else:
                one_final_net_bbox = torch.empty((0, 4), device=pre_bboxes.device)
                one_final_net_scores = torch.empty((0,), device=scores.device)
                one_final_net_labels = torch.empty((0,), dtype=torch.int64, device=pre_bboxes.device)

            output_detections.append({
                'boxes': one_final_net_bbox,
                'scores': one_final_net_scores,
                'labels': one_final_net_labels,
            })

        return output_detections


# --- 用于测试本模块完整功能的脚本 ---
if __name__ == '__main__':
    print("--- DetectorHead 完整功能 (含批处理和后处理) 测试 ---")

    # 1. 设置测试环境
    BATCH_SIZE = 2
    NUM_CLASSES = 20
    IN_CHANNELS = 1024

    roi_pooler_instance = RoIPool(output_size=7, spatial_scale=1.0 / 16)
    detector_head = DetectorHead(
        num_classes=NUM_CLASSES,
        pooler=roi_pooler_instance,
        in_channels=IN_CHANNELS
    )

    # 2. 创建模拟数据
    dummy_feature_map = torch.randn(BATCH_SIZE, IN_CHANNELS, 50, 50)
    dummy_image_size = (800, 800)
    dummy_proposals = [
        torch.rand(300, 4) * 800,
        torch.rand(200, 4) * 800
    ]
    dummy_targets = [
        {'bboxes': torch.tensor([[10, 10, 100, 100], [200, 200, 350, 350]], dtype=torch.float32),
         'labels': torch.tensor([1, 5], dtype=torch.int64)},
        {'bboxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
         'labels': torch.tensor([8], dtype=torch.int64)}
    ]

    # --- 3. 测试训练模式 (与之前相同) ---
    print("\n--- 3. 测试训练模式 (model.train()) ---")
    detector_head.train()
    detections_train, losses_train = detector_head(dummy_feature_map, dummy_proposals, dummy_image_size, dummy_targets)
    assert isinstance(losses_train, dict) and "roi_cls_loss" in losses_train
    print("训练模式测试通过！")

    # --- 4. 测试评估模式 (现在会返回结果) ---
    print("\n--- 4. 测试评估模式 (model.eval()) ---")
    detector_head.eval()
    with torch.no_grad():
        detections_eval, losses_eval = detector_head(dummy_feature_map, dummy_proposals, dummy_image_size)

    # 验证返回类型和结构
    assert isinstance(detections_eval, list), "评估时应返回一个detections列表"
    assert len(detections_eval) == BATCH_SIZE, "Detections列表长度应等于批次大小"
    assert isinstance(losses_eval, dict) and len(losses_eval) == 0, "评估时损失字典应为空"

    print(f"返回了 {len(detections_eval)} 张图片的检测结果。")
    first_img_result = detections_eval[0]
    assert 'boxes' in first_img_result and 'scores' in first_img_result and 'labels' in first_img_result
    print("Detections 列表的结构正确 (包含 'boxes', 'scores', 'labels')。")
    print("评估模式测试通过！")

    print("\n\n--- 所有测试通过！DetectorHead 功能完整且正确。 ---")