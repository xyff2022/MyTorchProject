import math

import torch
import torch.nn as nn


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


def decode_targets(predictions, targets, anchors, anchor_masks, strides, anchor_t):
    """
    (解耦版本)
    该函数是连接“真实世界标签”和“模型原始输出”的桥梁。
    它接收所有必要配置作为参数,不直接依赖于某个特定的模型类实例。
    Args:
        predictions (list): 一个包含三个预测头输出的列表。
        targets (Tensor): 一个单一的张量，它整合了整个批次中所有图片的所有真实物体的标签信息。[num_targets, 6]。
                        (Image Index)+(Class Index)+(Normalized x, y, w, h)
        anchors (Tensor): 包含所有9个anchor原始尺寸的张量, shape: [9, 2]。
        anchor_masks (list): 指定每个头使用哪些anchor的索引列表, e.g., [[6,7,8], [3,4,5], [0,1,2]]。
        strides (list): 包含三个预测头步长的列表, e.g., [32, 16, 8]。
        anchor_t (float): 用于匹配的anchor宽高比例阈值,用于增加正样本
    Returns:
        tcls, tbox, indices, anchor (list): 包含每个预测层目标的列表。
    """
    # 获取真实标签的数量
    number_targets = targets.shape[0]
    # 初始化四个列表,用于存储每个预测层的目标
    tcls, tbox, indices, anchor = [], [], [], []

    # gain用于将归一化的targets坐标缩放到特征图尺度
    gain = torch.ones(7, device=targets.device)  # 7: img_idx, class, x, y, w, h, anchor_idx

    # 创建一个从0到2的anchor索引张量,方便后续匹配
    # anchor_index shape: [3, 1] -> [3, number_targets], 内容是 [[0,0,..], [1,1,..], [2,2,..]]
    anchor_index = torch.arange(3,device=targets.device).float().view(3,1).repeat(1,number_targets)
    # repeat遇到维度不匹配，会从前面加维度

    # targets张量复制3份,每一份对应一个anchor,并附加上anchor索引
    # 最终 targets shape: [3, number_targets, 7]
    targets = torch.cat((targets.repeat(3,1,1),anchor_index[:,:,None]),dim=2)

    # 遍历三个预测层 (i 是索引 0, 1, 2)
    for i, mask in enumerate(anchor_masks):
        # --- 直接从参数获取信息 ---
        # 1. 筛选出当前头使用的详细anchor列表,形状[3,2]
        current_anchors = anchors[mask]
        # 2. 获取当前头的步长
        current_stride = strides[i]
        # 3. 计算在当前特征图尺度下的anchor的尺寸 (current_anchor_size)
        current_anchor_size = (current_anchors/current_stride).to(predictions[i].device)

        # 【核心修改】: 从5D的prediction张量中正确获取H和W
        shape = predictions[i].shape  # [bs, num_anchors, H, W, 5+num_classes]
        # gain中的[x, y, w, h]应对应[W, H, W, H]
        gain[2:6] = torch.tensor([shape[3], shape[2], shape[3], shape[2]], device=targets.device)

        # 将targets缩放到当前特征图尺度
        t = targets * gain

        if number_targets:
            # --- 核心匹配逻辑 ---
            # 1. 计算targets和current_anchor_size的宽高比例
            r = targets[:,:,4:6]/current_anchor_size[:,None]
            # targets[:, :, 4:6] 形状[3, number_targets, 2]，current_anchor_size[3,2],使用None在第0维后加入一维
            # r[3,number_targets,2]
            # 2. 使用传入的anchor_t作为阈值进行筛选
            j = torch.max(r, 1.0/r).max(2)[0] < anchor_t
            # .max(2)会寻找最后一个维度（维度2，即[宽变形量, 高变形量]）上的最大值。
            # 它会返回一个元组(values, indices)，我们用[0]取出其中的values。
            # .max(2)会压缩第2个维度，即最后一维，现在j[3,number_targets]
            t = t[j]
            # t的形状此时为[N_matches,7]即所有被匹配的样本及其7个"属性"
        else:
            t= targets[0]
            # targets此前为[3, number_targets, 7]若number_targets=0,为[3, 0, 7]，取targets[0]为[0,7]

        # --- 提取信息 ---
        batch_idx, class_idx = t[:,:2].long().T
        gxy = t[:,2:4]
        gwh = t[:,4:6]
        gij = gxy.long()
        # 此时gij为(N_matches,2),为一对对的ij
        gi,gj = gij.T
        # 取出所有的单独的i和单独的j

        # --- 存储结果 ---
        anchor_for_target = t[:, 6].long()
        # 每个target匹配上的anchor索引 (0, 1, or 2)

        # 存储一个元组(图片索引 (这个正样本属于批次中的哪张图)。 Anchor 索引 (应该由当前网格的第几个 anchor 负责)。
        # gj: 网格 Y 坐标。 gi: 网格 X 坐标。)，它包含了定位一个正样本所需的全部索引信息。
        gj=gj.clamp(0, gain[3].item() - 1)
        gi=gi.clamp(0, gain[2].item() - 1)
        stack_indices = torch.stack((batch_idx,anchor_for_target,gj,gi),dim=0)
        indices.append(stack_indices)

        # tbox存储一个[N_matches, 4]的张量，这4列就是模型要学习的编码后的坐标目标
        # 4为模型预测的框的中心点相对于网格左上角的偏移量x和y，以及在当前尺度下框的绝对宽度和高度
        tbox.append(torch.cat((gxy-gij, gwh),1))

        # 存储每个正样本匹配上的那个anchor在当前特征图尺度下的[width, height]
        anchor.append(current_anchor_size[anchor_for_target])

        tcls.append(class_idx)

    return tcls, tbox, indices, anchor


def compute_ciou_loss(predictions,
                      targets,
                      anchors_all,
                      anchor_masks,
                      strides,
                      anchor_t,
                      image_size,
                      cls_positive_weight=1.0,
                      obj_positive_weight=1.0,
                      num_classes = 20,
                      classes_eps = 0.0,
                      ciou_gain=3.54, cls_gain=37.4, obj_gain=64.3):
                      # ciou_gain=3.54, cls_gain=37.4, obj_gain=64.3这三个超参数最好不用
                      # ciou_gain = 0.05, # cls_gain = 0.3, # obj_gain = 0.7
    """
    计算YOLOv3的总损失, 该损失由三部分组成: 定位损失, 置信度损失和分类损失。

    Args:
        predictions (list): 模型的原始预测输出, 包含三个预测头的张量。
                  - p[0] shape: [bs, 3, 16, 16, nc+5]
                  - p[1] shape: [bs, 3, 32, 32, nc+5]
                  - p[2] shape: [bs, 3, 64, 64, nc+5]
        targets (torch.Tensor): 批次内所有真实标签, shape: [num_total_targets, 6]。
                                 每行格式为 [img_idx, cls_idx, x, y, w, h]。
        anchors_all (torch.Tensor): 全部9个anchor的原始像素尺寸, shape: [9, 2]。
        anchor_masks (list): Anchor的分配方案, e.g., [[6,7,8], [3,4,5], [0,1,2]]。
        strides (list): 三个预测头的步长, e.g., [32, 16, 8]。
        anchor_t (float): Anchor匹配的形状阈值, 用于build_targets。
        image_size(int): 当前训练的图像尺寸, 用于动态调整obj_gain。
        cls_positive_weight (float): 分类损失中BCE的正样本权重。
        obj_positive_weight (float): 置信度损失中BCE的正样本权重。
        num_classes (int): 数据集中的类别总数。
        classes_eps (float): 分类损失的标签平滑因子, 默认0.0 (不平滑)。
        ciou_gain (float): GIoU/CIoU损失的权重。
        cls_gain (float): 分类损失的权重。
        obj_gain (float): 置信度损失的权重。
    """
    # =============================== 代码块 1: 初始化与准备工作 ===============================
    device = predictions[0].device
    # 初始化三个损失值为零张量
    cls_loss = torch.zeros(1, device=device)  # 分类损失
    box_loss = torch.zeros(1, device=device)  # 定位损失 (CIoU)
    obj_loss = torch.zeros(1, device=device)  # 置信度损失

    # 定义损失函数, 使用传入的 cls_pw 和 obj_pw
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_positive_weight], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_positive_weight], device=device))

    # =============================== 关键改动: 调用 smooth_BCE ===============================
    class_positive, class_negative = smooth_BCE(eps=classes_eps)

    # 【新增】定义三个预测头的置信度损失平衡权重
    # 这是为了平衡不同尺度特征图上正负样本的巨大差异
    # [大物体头, 中物体头, 小物体头]
    obj_balance = [4.0, 1.0, 0.4]

    # 调用decode_targets函数, 获取为当前批次数据量身定做的“靶子”
    tcls,tbox,indices,anchor=decode_targets(predictions, targets, anchors_all, anchor_masks, strides, anchor_t)

    # =============================== 代码块 2: 遍历预测头并提取正样本 ===============================

    # 遍历三个预测层 (i 是索引 0, 1, 2)
    for i, prediction in enumerate(predictions):
        # stacked_indices 是一个 [4, N] 的张量
        stacked_indices = indices[i]

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
            pos_sample = prediction[batch_idx,anchor_for_target,gj,gi]



            # =============================== 代码块 3: 计算定位损失 (CIoU Loss) ===============================

            # 1. 解码预测框, 得到在特征图尺度下的 xywh
            prediction_xy = pos_sample[..., :2].sigmoid()
            prediction_wh = torch.exp(pos_sample[..., 2:4]).clamp(max=1e3) * anchor[i]
            prediction_box = torch.cat((prediction_xy,prediction_wh),1)

            # 2. 准备目标框 (tbox中存储的已经是 [tx,ty,gw,gh] 格式)
            target_box = tbox[i]

            ciou = bbox_ciou(prediction_box,target_box)
            box_loss = box_loss + (1.0 - ciou).mean()

            # =============================== 代码块 4: 计算分类损失 (lcls) ===============================

            # 只有在类别数大于1时才计算分类损失
            if num_classes > 1:
                target_class = torch.full_like(pos_sample[..., 5:], class_negative, device=device)
                # target_class形状[number_pos_targets, num_classes]
                target_class[range(number_pos_targets), tcls[i]] = class_positive
                cls_loss += BCEcls(pos_sample[..., 5:],target_class)

            # 在置信度目标张量tobj的对应位置上, 将值设为1
            tobj[batch_idx, anchor_for_target, gj, gi] = ciou.detach().clamp(0).to(tobj.dtype)

        obj_loss += BCEobj(prediction[..., 4], tobj) *obj_balance[i]

    # =============================== 关键改动: 应用ultralytics的权重策略 ===============================
    # 动态调整obj_gain和cls_gain
    obj_gain *= (image_size / 320) ** 2
    cls_gain *= num_classes / 80



    # 应用最终的损失权重
    box_loss *= ciou_gain
    obj_loss *= obj_gain
    cls_loss *= cls_gain

    return {"box_loss": box_loss, "obj_loss": obj_loss, "class_loss": cls_loss}

























