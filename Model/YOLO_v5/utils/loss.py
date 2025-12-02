import math

import torch
import torch.nn as nn

from torch.nn import Module


def bbox_ciou(b1, b2):
    """
    计算CIoU
    Args:
        b1 (Tensor): 预测框, shape: [N, 4], format: [x, y, w, h]
        b2 (Tensor): 目标框, shape: [N, 4], format: [x, y, w, h]
    Returns:
        ciou (Tensor): CIoU值, shape: [N, 1]
    """
    # -----------------------------------------------------------#
    #   步骤 1: 准备工作与坐标转换
    # -----------------------------------------------------------#
    # 将 [x, y, w, h] 转换为 [x1, y1, x2, y2]
    b1_x1, b1_y1 = b1[:, 0] - b1[:, 2] / 2, b1[:, 1] - b1[:, 3] / 2
    b1_x2, b1_y2 = b1[:, 0] + b1[:, 2] / 2, b1[:, 1] + b1[:, 3] / 2
    b2_x1, b2_y1 = b2[:, 0] - b2[:, 2] / 2, b2[:, 1] - b2[:, 3] / 2
    b2_x2, b2_y2 = b2[:, 0] + b2[:, 2] / 2, b2[:, 1] + b2[:, 3] / 2

    # 计算每个框的面积(
    area1 = b1[..., 2] * b1[..., 3]
    area2 = b2[..., 2] * b2[..., 3]

    # -----------------------------------------------------------#
    #   步骤 2: 计算交集面积
    # -----------------------------------------------------------#
    # 找到交集矩形的左上角和右下角坐标
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # 计算交集宽高，并用clamp确保不为负数
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算交集面积
    inter = inter_w * inter_h

    # -----------------------------------------------------------#
    #   步骤 3: 计算IoU
    # -----------------------------------------------------------#
    # 计算并集面积
    union = area1 + area2 - inter
    # 计算IoU，加上一个极小值eps防止除零
    iou = inter / (union + 1e-6)

    # -----------------------------------------------------------#
    #   步骤 4: 计算惩罚项 (CIoU的核心)
    # -----------------------------------------------------------#

    d2 = (b2[..., 0] - b1[..., 0]) ** 2 + (b2[..., 1] - b1[..., 1]) ** 2
    c2 = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)) ** 2 + (
            torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)) ** 2 + 1e-6
    v = 4 * (torch.atan((b1[..., 2] / (b1[..., 3] + 1e-6))) - torch.atan(
        (b2[..., 2] / (b2[..., 3] + 1e-6)))) ** 2 / math.pi ** 2

    # 根据CIoU原论文的设计,alpha在一次前向传播中被当作一个固定的常量来使用.我们不希望梯度从最终的损失通过alpha再反向传播回iou
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    return iou - d2 / c2 - alpha * v


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    """
        标签平滑技术, 返回平滑后的正负标签值。
        Args:
            eps (float): 平滑因子。
        Returns:
            (float, float): (平滑后的正标签, 平滑后的负标签)
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputeCiou(Module):
    def __init__(self, model, cls_eps=0.1):
        super().__init__()
        # 获取检测头模块
        m = model.detect
        device = next(model.parameters()).device

        # 定义三种损失的权重
        self.box_weight = 0.05
        self.cls_weight = 0.5 * (20 / 80.)  # 分类损失权重会根据类别数进行调整
        self.obj_weight = 1.0

        # 定义二元交叉熵损失函数
        # pos_weight 给予正样本（有物体的）更高的权重
        self.BCE_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
        self.BCE_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

        # 超参数
        self.anchor_t = 4.0  # 锚框匹配阈值：GT框 与 锚框 的高宽比小于此阈值，才算匹配
        self.gain_ratio = 0.5
        self.anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        self.number_classes = m.number_classes  # 类别数
        self.number_detect = m.number_detect  # 检测层数
        self.anchors = m.anchors  # 锚框
        self.stride = m.stride  # 步长
        self.number_anchor = m.number_anchor

        self.class_positive, self.class_negative = smooth_BCE(eps=cls_eps)

    def forward(self, predictions, targets):
        device = predictions[0].device

        # 初始化三个损失值为零张量
        cls_loss = torch.zeros(1, device=device)  # 分类损失
        box_loss = torch.zeros(1, device=device)  # 定位损失 (CIoU)
        obj_loss = torch.zeros(1, device=device)  # 置信度损失

        # 调用decode_targets函数, 获取为当前批次数据量身定做的“靶子”
        tcls, tbox, indices, anchor = self.build_targets(predictions, targets)

        # =============================== 代码块 2: 遍历预测头并提取正样本 ===============================

        # 遍历三个预测层 (i 是索引 0, 1, 2)
        for i, prediction in enumerate(predictions):
            # stacked_indices 是一个 [4, number_true_box] 的张量
            stacked_indices = indices[i]
            # stacked_indices.shape [number_true_box,number_true_box,number_true_box,number_true_box]

            # 创建一个全零的置信度目标张量
            tobj = torch.zeros_like(prediction[..., 0], device=device)

            # 获取当前层正样本的数量
            number_pos_targets = stacked_indices.shape[1]

            if number_pos_targets:
                # 使用索引从预测张量pi中提取出所有正样本的预测
                # ps 的形状: [nb, num_classes + 5]
                # 【核心修改】: 在使用索引前，强制确保所有索引张量为 long 类型
                batch_idx, anchor_for_target, gj, gi = stacked_indices
                batch_idx = batch_idx.long()
                anchor_for_target = anchor_for_target.long()
                gj = gj.long()
                gi = gi.long()
                pos_sample = prediction[batch_idx, anchor_for_target, gj, gi]

                # =============================== 代码块 3: 计算定位损失 (CIoU Loss) ===============================

                # 1. 解码预测框, 得到在特征图尺度下的 xywh
                prediction_xy = pos_sample[..., :2].sigmoid() * 2 - 0.5
                # prediction_wh = torch.exp(pos_sample[..., 2:4]).clamp(max=1e3) * anchor[i]
                prediction_wh = (pos_sample[..., 2:4].sigmoid() * 2) ** 2 * anchor[i]
                prediction_box = torch.cat((prediction_xy, prediction_wh), 1)

                # 2. 准备目标框 (tbox中存储的已经是 [tx,ty,gw,gh] 格式)
                target_box = tbox[i]

                ciou = bbox_ciou(prediction_box, target_box)
                box_loss = box_loss + (1.0 - ciou).mean()

                # =============================== 代码块 4: 计算分类损失 (lcls) ===============================

                # 只有在类别数大于1时才计算分类损失
                if self.number_classes > 1:
                    target_class = torch.full_like(pos_sample[..., 5:], self.class_negative, device=device)
                    # target_class形状[number_pos_targets, num_classes]
                    target_class[range(number_pos_targets), tcls[i]] = self.class_positive
                    cls_loss += self.BCE_cls(pos_sample[..., 5:], target_class)

                # 在置信度目标张量tobj的对应位置上, 将值设为1
                tobj[batch_idx, anchor_for_target, gj, gi] = ciou.detach().clamp(0).to(tobj.dtype)

            obj_loss += self.BCE_obj(prediction[..., 4], tobj)

        return {"box_loss": box_loss, "obj_loss": obj_loss, "class_loss": cls_loss}

    def build_targets(self, predictions, targets):

        anchors = self.anchors
        number_anchor = self.number_anchor
        device = predictions[0].device

        # 获取真实标签的数量
        number_targets = targets.shape[0]
        # 初始化四个列表,用于存储每个预测层的目标
        tcls, tbox, indices, anchor = [], [], [], []

        # gain用于将归一化的targets坐标缩放到特征图尺度
        gain = torch.ones(7, device=device)  # 7: img_idx, class, x, y, w, h, anchor_idx

        # 创建一个从0到2的anchor索引张量,方便后续匹配
        # anchor_index shape: [3, 1] -> [3, number_targets], 内容是 [[0,0,..], [1,1,..], [2,2,..]]
        anchor_index = torch.arange(3, device=device).float().view(3, 1).repeat(1, number_targets)
        # repeat遇到维度不匹配，会从前面加维度

        # targets张量复制3份,每一份对应一个anchor,并附加上anchor索引
        # 最终 targets shape: [3, number_targets, 7]
        targets = torch.cat((targets.repeat(3, 1, 1), anchor_index[:, :, None]), dim=2)

        offset = torch.tensor(
            [
                [0, 0],  # 0: 中心网格 (对应 mask_center)
                [1, 0],  # 1: 左侧网格 (对应 mask_left, gi-1)
                [0, 1],  # 2: 上方网格 (对应 mask_top, gj-1)
                [-1, 0],  # 3: 右侧网格 (对应 mask_right, gi+1)
                [0, -1],  # 4: 下方网格 (对应 mask_bottom, gj+1)
            ], device=device
        ).float()

        # 遍历三个预测层 (i 是索引 0, 1, 2)
        for i, mask in enumerate(self.anchor_masks):
            # --- 直接从参数获取信息 ---
            # 1. 筛选出当前头使用的详细anchor列表,形状[3,2]
            current_anchors = anchors[mask]
            # 2. 获取当前头的步长
            current_stride = self.stride[i]
            # 3. 计算在当前特征图尺度下的anchor的尺寸 (current_anchor_size)
            current_anchor_size = (current_anchors / current_stride).to(device)

            # 【核心修改】: 从5D的prediction张量中正确获取H和W
            shape = predictions[i].shape  # [bs, num_anchors, H, W, 5+num_classes]
            # gain中的[x, y, w, h]应对应[W, H, W, H]
            gain[2:6] = torch.tensor([shape[3], shape[2], shape[3], shape[2]], device=device)

            # 将targets缩放到当前特征图尺度
            t = targets * gain

            if number_targets:
                # --- 核心匹配逻辑 ---
                # 1. 计算targets和current_anchor_size的宽高比例
                r = t[:, :, 4:6] / current_anchor_size[:, None]
                # targets[:, :, 4:6] 形状[3, number_targets, 2]，current_anchor_size[3,2],使用None在第0维后加入一维
                # r[3,number_targets,2]
                # 2. 使用传入的anchor_t作为阈值进行筛选
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t
                # .max(2)会寻找最后一个维度（维度2，即[宽变形量, 高变形量]）上的最大值。
                # 它会返回一个元组(values, indices)，我们用[0]取出其中的values。
                # .max(2)会压缩第2个维度，即最后一维，现在j[3,number_targets]
                t = t[j]
                # t的形状此时为[N_matches,7]即所有被匹配的样本及其7个"属性"

                # 邻近网格匹配
                gxy = t[:, 2:4]
                gij = gxy.long()

                # YOLOv5核心: 增加额外的网格作为正样本
                # 这是YOLOv5召回率高的一个重要原因
                gxy_offset = gxy - gij

                anchor_size = gain[2:4].long()

                left_up = gxy_offset < self.gain_ratio
                right_down = gxy_offset > 1 - self.gain_ratio

                valid_left_up = gij > 0
                valid_right_down = gij < (anchor_size - 1)

                left_mask = left_up[:, 0] & valid_left_up[:, 0]
                up_mask = left_up[:, 1] & valid_left_up[:, 1]
                right_mask = right_down[:, 0] & valid_right_down[:, 0]
                down_mask = right_down[:, 1] & valid_right_down[:, 1]

                center_mask = torch.ones_like(left_mask, dtype=torch.bool)

                mask = torch.stack((center_mask, left_mask, up_mask, right_mask, down_mask))
                # mask[5,N_matches],里面有number_true_box个True

                # t [N_matches, 7] -> t [5,N_matches, 7] ->[number_true_box,7]
                t = t.repeat((5, 1, 1))[mask].view(-1, 7)

                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[mask].view(-1, 2)

                # gxy[N_matches, 2] -> gxy[1, N_matches, 2]
                # offset[5,2] ->offset[5, 1, 2]
                #  [5, N_matches, 2]用mask取出number_true_box个物体
                # offsets[number_true_box,2]


            else:
                t = targets[0]
                # targets此前为[3, number_targets, 7]若number_targets=0,为[3, 0, 7]，取targets[0]为[0,7]
                offsets = torch.zeros(0, 2, device=t.device)

            # --- 提取信息 ---
            # 假设有物体，此时t[number_true_box,7]
            batch_idx, class_idx = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            # 此时gij为(number_true_box,2),为一对对的ij
            gi, gj = gij.T
            # 取出所有的单独的i和单独的j

            # --- 存储结果 ---
            anchor_for_target = t[:, 6].long()
            # 每个target匹配上的anchor索引 (0, 1, or 2)

            # 存储一个元组(图片索引 (这个正样本属于批次中的哪张图)。 Anchor 索引 (应该由当前网格的第几个 anchor 负责)。
            # gj: 网格 Y 坐标。 gi: 网格 X 坐标。)，它包含了定位一个正样本所需的全部索引信息。
            gj = gj.clamp(0, gain[3].item() - 1)
            gi = gi.clamp(0, gain[2].item() - 1)
            stack_indices = torch.stack((batch_idx, anchor_for_target, gj, gi), dim=0)
            indices.append(stack_indices)

            # tbox存储一个[N_matches, 4]的张量，这4列就是模型要学习的编码后的坐标目标
            # 4为模型预测的框的中心点相对于网格左上角的偏移量x和y，以及在当前尺度下框的绝对宽度和高度
            tbox.append(torch.cat((gxy - gij, gwh), 1))

            # 存储每个正样本匹配上的那个anchor在当前特征图尺度下的[width, height]
            anchor.append(current_anchor_size[anchor_for_target])

            tcls.append(class_idx)

        return tcls, tbox, indices, anchor