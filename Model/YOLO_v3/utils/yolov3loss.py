import math

import torch
import torch.nn
from torch.nn import Module, BCEWithLogitsLoss


class YOLOLoss(Module):
    def __init__(self, anchors,
                 num_classes,
                 input_shape,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                 ):
        super(YOLOLoss, self).__init__()
        # -----------------------------------------------------------#
        #   第一步: 存储模型的基本配置
        #   anchors: 所有的先验框
        #   num_classes: 类别数量
        #   input_shape: 输入图像的尺寸
        #   anchors_mask: 用于区分三个预测头的anchor索引
        #   ignore_threshold : 一个IoU阈值,用于在计算损失时忽略某些预测
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.ignore_threshold = 0.5

        # -----------------------------------------------------------#
        #   第二步: 定义损失的超参数
        #   balance: 不同预测头的置信度损失权重
        #   box_ratio: 定位损失的全局权重
        #   obj_ratio: 置信度损失的全局权重
        #   cls_ratio: 分类损失的全局权重
        # -----------------------------------------------------------#
        self.balance = [0.4, 1.0, 4]
        # 给大物体头（16x16）的损失乘以0.4（降权）.
        # 给中物体头（32x32）的损失乘以1.0（不变）.
        # 给小物体头（64x64）的损失乘以4.0（提权）.
        # 这样可以确保模型在所有尺度上都能得到有效地学习.
        # box_ratio: 定位损失(box_loss)的全局权重, 用于平衡各损失的量级
        self.box_ratio = 0.05
        # obj_ratio: 置信度损失(obj_loss)的全局权重, 根据输入尺寸动态缩放以适应不同数量的负样本
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        # cls_ratio: 分类损失(cls_loss)的全局权重, 根据类别数动态缩放以适应不同难度的分类任务
        self.cls_ratio = 1 * (num_classes / 80)
        # 通过权重,人为地放大了置信度损失的影响力,使其在梯度更新中占据主导地位.
        self.BCELoss = BCEWithLogitsLoss(reduction= "none")
        # 定义二元交叉熵损失函数.reduction="none"表示返回每个元素的损失,方便我们后续手动加权

    def wh_iou(self, wh1, wh2):
        """
        计算宽高IoU (不考虑位置)
        Args:
            wh1 (Tensor): 第一组框的宽高, shape: [N, 2]
            wh2 (Tensor): 第二组框的宽高, shape: [M, 2]
        Returns:
            iou (Tensor): 形状IoU, shape: [N, M]
        """
        # -----------------------------------------------------------#
        #   步骤 1: 重塑张量以进行广播
        # -----------------------------------------------------------#
        # wh1 shape: [N, 2] -> [N, 1, 2]
        wh1 = wh1.unsqueeze(1)
        # wh2 shape: [M, 2] -> [1, M, 2]
        wh2 = wh2.unsqueeze(0)

        # -----------------------------------------------------------#
        #   步骤 2: 计算交集面积
        #   torch.min会广播wh1和wh2到[N, M, 2]，然后逐元素比较
        #   .prod(2)会沿着最后一个维度(dim=2)将[min_w, min_h]相乘得到交集面积
        # -----------------------------------------------------------#
        inter = torch.min(wh1, wh2).prod(2)

        # -----------------------------------------------------------#
        #   步骤 3: 计算并集面积并返回IoU
        #   wh1.prod(2) -> 面积A, shape: [N, 1]
        #   wh2.prod(2) -> 面积B, shape: [1, M]
        #   两者相加再减去inter，会通过广播得到[N, M]的并集面积
        # -----------------------------------------------------------#
        return inter / (wh1.prod(2) + wh2.prod(2) - inter +1e-6)

    def bbox_ciou(self, b1, b2):
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

        d2=(b2[..., 0] - b1[..., 0]) **2 + (b2[..., 1] - b1[..., 1]) **2
        c2=(torch.max(b1_x2,b2_x2)-torch.min(b1_x1, b2_x1))**2+(torch.max(b1_y2,b2_y2)-torch.min(b1_y1,b2_y1))**2+1e-6
        v = 4*(torch.atan((b1[..., 2]/(b1[..., 3]+1e-6)))-torch.atan((b2[..., 2]/(b2[..., 3]+1e-6))))**2/math.pi**2

        # 根据CIoU原论文的设计,alpha在一次前向传播中被当作一个固定的常量来使用.我们不希望梯度从最终的损失通过alpha再反向传播回iou
        with torch.no_grad():
            alpha = v/(1-iou+v+1e-6)

        return iou-d2/c2-alpha*v





    def get_target(self, prediction, targets, anchors, in_w, in_h,
                   ):
        """
        为单个预测头的输出构建目标
        Args:
            prediction (Tensor): 单个尺度的模型预测输出, shape: [batch_size, 3, in_h, in_w, 5 + num_classes]
            targets (list): 一个批次的真实标签, [num_targets, 5([class_id, center_x, center_y, width, height])]
            anchors (Tensor): 当前尺度对应的3个先验框尺寸
            in_w (int): 特征图宽度
            in_h (int): 特征图高度
        Returns:
            mask (Tensor): 正样本掩码
            noobj_mask (Tensor): 负样本掩码
            tx, ty, tw, th (Tensor): 坐标和尺寸的目标值
            t_cls (Tensor): 类别目标值
            box_loss_scale (Tensor): 定位损失的权重
        """
        with torch.no_grad():
            # 获取批次大小
            batch_size = prediction.size(0)
            # 获取当前尺度下的先验框数量
            num_anchors = len(anchors)
            DEVICE = prediction.device

            # -----------------------------------------------------------#
            #   步骤 1: 初始化目标张量
            # -----------------------------------------------------------#
            # mask: (batch_size, 3, in_h, in_w)正样本掩码,值为1的位置表示该处的(网格, anchor)组合负责预测一个物体.
            # noobj_mask:(batch_size, 3, in_h, in_w)负样本掩码,值为1的位置表示该处是背景.
            # 一张全1的地图，默认所有地方都是背景。我们会在找到正样本和模糊样本的位置把它标记为0。
            # tx,ty,tw,th(batch_size, 3, in_h, in_w)分别存储真实框中心点x, y的偏移量目标值,以及宽高w,h的缩放比目标值
            # t_cls:(batch_size, 3, in_h, in_w, num_classes)存储真实框的类别目标值（经过one-hot编码）.
            # box_loss_scale:(batch_size, 3, in_h, in_w)存储每个正样本的定位损失权重,用于给小目标更大的权重.
            mask = torch.zeros(batch_size, num_anchors, in_h, in_w, requires_grad=False, device=DEVICE)
            noobj_mask = torch.ones(batch_size, num_anchors, in_h, in_w, requires_grad=False, device=DEVICE)
            tx = torch.zeros(batch_size, num_anchors, in_h, in_w, requires_grad=False, device=DEVICE)
            ty = torch.zeros(batch_size, num_anchors, in_h, in_w, requires_grad=False, device=DEVICE)
            tw = torch.zeros(batch_size, num_anchors, in_h, in_w, requires_grad=False, device=DEVICE)
            th = torch.zeros(batch_size, num_anchors, in_h, in_w, requires_grad=False, device=DEVICE)
            t_cls = torch.zeros(batch_size, num_anchors, in_h, in_w, self.num_classes,requires_grad=False, device=DEVICE)
            box_loss_scale = torch.zeros(batch_size, num_anchors, in_h, in_w, requires_grad=False, device=DEVICE)

            # -----------------------------------------------------------#
            #   步骤 2: 遍历批次中的每张图片
            # -----------------------------------------------------------#
            for b in range(batch_size):
                # -----------------------------------------------------------#
                #   步骤 3: 匹配真实框与先验框
                # -----------------------------------------------------------#
                # 如果这张图没有物体，则跳过
                if len(targets[b]) == 0:
                    continue

                # 将真实框的宽高映射到特征图尺度 (确保在正确的设备上创建)
                # target为yolo格式label
                gt_wh = targets[b][:, 3:5] * torch.tensor([in_w, in_h], dtype=torch.float,device=DEVICE)
                # targets[b][:, 3:5]的形状是[N, 2],torch.tensor([in_w, in_h], ...) 的形状是 [2]
                # 一个[N, 2]的张量和一个[2]的张量相乘时，它会自动“广播”那个较小的张量，把它扩展成[N, 2]的形状，让它们可以对齐相乘.

                # 计算真实框与先验框的形状IoU
                anchor_iou = self.wh_iou(gt_wh, anchors)
                # 找到与每个真实框形状最匹配的先验框索引
                best_ns = torch.argmax(anchor_iou, dim=1)

                # -----------------------------------------------------------#
                #   【新增代码】预计算真实框在特征图上的坐标
                # -----------------------------------------------------------#
                gt_boxes = torch.zeros_like(targets[b])
                gt_boxes[:, 0] = targets[b][:, 1] * in_w
                gt_boxes[:, 1] = targets[b][:, 2] * in_h
                # -----------------------------------------------------------#

                # -----------------------------------------------------------#
                #   步骤 4: 填充目标张量
                # -----------------------------------------------------------#
                for i, gt_box in enumerate(targets[b]):

                    # -----------------------------------------------------------#
                    #   【新增代码】处理模糊样本
                    # -----------------------------------------------------------#
                    # anchor_iou[i] 包含了当前真实框与3个先验框的形状IoU
                    # 通过与阈值比较，得到一个布尔张量, e.g., [False, True, True]
                    # True的位置代表对应的先验框是“模糊样本”，需要被忽略
                    ignored_anchors = anchor_iou[i] > self.ignore_threshold

                    # 获取真实框所在的网格坐标
                    gi = int(gt_boxes[i, 0])
                    gj = int(gt_boxes[i, 1])

                    # 安全检查，防止坐标越界
                    if gi < in_w and gj < in_h:
                        # 使用布尔张量 ignored_anchors 作为索引
                        # 将 noobj_mask 在 (gj, gi) 这个网格里
                        # 所有“模糊样本”对应的位置都设置为0
                        noobj_mask[b, ignored_anchors, gj, gi] = 0
                    # -----------------------------------------------------------#
                    #   【新增代码结束】
                    # -----------------------------------------------------------#



                    # 获取真实框的类别和坐标
                    gt_class = int(gt_box[0])
                    gt_x = gt_box[1] * in_w
                    gt_y = gt_box[2] * in_h
                    gt_w, gt_h = gt_wh[i]

                    # 网格的整数坐标
                    gt_i = int(gt_x)
                    gt_j = int(gt_y)

                    # 找到最佳anchor
                    best_n = best_ns[i]

                    # 确保网格坐标在有效范围内
                    if gt_i < in_w and gt_j < in_h:
                        # 标记正负样本
                        mask[b, best_n, gt_j, gt_i] = 1
                        noobj_mask[b, best_n, gt_j, gt_i] = 0

                        # 计算目标坐标和尺寸
                        tx[b, best_n, gt_j, gt_i] = gt_x - gt_i
                        ty[b, best_n, gt_j, gt_i] = gt_y - gt_j
                        tw[b, best_n, gt_j, gt_i] = math.log(gt_w / anchors[best_n][0])
                        th[b, best_n, gt_j, gt_i] = math.log(gt_h / anchors[best_n][1])

                        # 设置类别目标
                        t_cls[b, best_n, gt_j, gt_i, gt_class] = 1

                        # 设置定位损失的权重
                        box_loss_scale[b, best_n, gt_j, gt_i] = 2 - gt_box[3] * gt_box[4]
                        # 为小目标的定位损失赋予更高的权重
                        # 人为地放大了小物体在计算定位损失时的“话语权”

        # -----------------------------------------------------------#
        #   步骤 5: 返回所有目标张量
        # -----------------------------------------------------------#
        return mask, noobj_mask, tx, ty, tw, th, t_cls, box_loss_scale





    def forward(self, l_prediction, m_prediction, s_prediction, targets ):
        # -------------------------------------------------#
        # YOLOv3输出的预测为(batch_size, 3 * (num_classes + 5), grid_h, grid_w)
        # 其中grid_h, grid_w为16,32,或64
        # YOLOLoss类,为了方便后续处理,期望接收的输入形状是(batch_size, 3, grid_h, grid_w, num_classes + 5)
        # -------------------------------------------------#
        #   l_prediction: (batch_size, 3, 16, 16, 5 + num_classes)
        #   m_prediction: (batch_size, 3, 32, 32, 5 + num_classes)
        #   s_prediction: (batch_size, 3, 64, 64, 5 + num_classes)
        #   targets: (batch_size, num_targets, 5) 真实标签
        #   [tx, ty, tw, th, confidence, class_1_prob, class_2_prob, ...]
        # -------------------------------------------------#

        batch_size = l_prediction.size(0)
        in_h_l, in_w_l = l_prediction.shape[2:4]
        in_h_m, in_w_m = m_prediction.shape[2:4]
        in_h_s, in_w_s = s_prediction.shape[2:4]

        # 将先验框尺寸从像素单位转换为归一化单位 (0-1)
        scaled_anchors = torch.tensor(self.anchors, dtype=torch.float, device=l_prediction.device) / torch.tensor([self.input_shape[1], self.input_shape[0]], dtype=torch.float, device= l_prediction.device).view(1, 2)
        # 将以像素为单位的先验框（Anchor）尺寸，转换为相对于整张图片尺寸的比例（0到1之间）。
        # scaled_anchors就像一个包含了所有归一化尺寸的“总仓库”。形状[9, 2],包含了所有9个框的比例尺寸。

        # 将预测结果、宽高和先验框打包成列表，方便循环处理
        predictions = [l_prediction, m_prediction, s_prediction]
        in_ws = [in_w_l, in_w_m, in_w_s]
        in_hs = [in_h_l, in_h_m, in_h_s]

        # 初始化总损失
        loss_box, loss_conf, loss_cls = 0, 0, 0

        # 遍历三个尺度的预测
        for i, pred in enumerate(predictions):
            in_w, in_h = in_ws[i], in_hs[i]
            anchor_mask = self.anchors_mask[i]
            # self.anchors_mask: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            # i = 0时,anchor_mask变量现在就等于[6, 7, 8]
            # 获取当前尺度对应的归一化先验框
            anchors = scaled_anchors[anchor_mask]
            # anchors = scaled_anchors[[6, 7, 8]],从scaled_anchors这个 [9, 2] 的张量中，挑出第6行、第7行和第8行

            # -----------------------------------------------------------#
            #   步骤 2: 解码模型预测值
            # -----------------------------------------------------------#
            pred_dx = torch.sigmoid(pred[..., 0])
            pred_dy = torch.sigmoid(pred[..., 1])
            pred_dw = pred[..., 2]
            pred_dh = pred[..., 3]
            pred_conf = pred[..., 4]
            pred_cls = pred[..., 5:]

            # -----------------------------------------------------------#
            #   步骤 3: 构建目标
            # -----------------------------------------------------------#
            mask, noobj_mask, tx, ty, tw, th, t_cls, box_loss_scale = self.get_target(
                pred, targets, anchors, in_w, in_h)

            # -----------------------------------------------------------#
            #   步骤 4: 计算各项损失
            # -----------------------------------------------------------#
            # 4a. 定位损失 (CIoU Loss)
            if mask.sum() > 0:
                # 筛选出正样本的定位损失权重
                box_loss_scale_pos = box_loss_scale[mask == 1]
                # 布尔张量做索引，会把所有符合条件的值取出来，组成一维张量，这里取出了所有正样本的损失权重

                grid_x = (torch.linspace(0,in_w- 1,in_w,device=l_prediction.device)
                          .repeat(in_h,1).repeat(batch_size*3,1,1).view(pred_dx.shape))
                grid_y = (torch.linspace(0,in_h- 1,in_h,device=l_prediction.device)
                          .repeat(in_w,1).t().repeat(batch_size*3,1,1).view(pred_dy.shape))
                # 详细描述grid_x的生成过程:
                # 1. torch.linspace(0, in_w - 1, in_w):
                #    函数参数: start=0, end=in_w-1, steps=in_w
                #    生成一个一维张量，内容为 [0., 1., 2., ..., in_w-1]。
                # 2. .repeat(in_h, 1):
                #    函数参数: (in_h, 1)
                #    将上述一维张量在第0维(行)重复in_h次，在第1维(列)重复1次，
                #    生成一个二维张量，形状为(in_h, in_w)，内容为每一行都是[0,1,2...]。
                # 3. .repeat(batch_size * len(self.anchors), 1, 1):
                #    将二维张量在新的第0维重复 batch_size * 3 次。
                # 4. .view(x.shape):
                #    将张量形状重塑为与x完全一致的 (batch_size, 3, in_h, in_w)。
                # 最终生成grid_x: 一个四维张量，其任意一个元素 grid_x[b, 3, h, w] 的值都等于其最后一维的索引w。

                # 创建anchor宽高
                anchor_w = anchors[..., 0].view(1,3,1,1).repeat(batch_size, 1, in_h, in_w)
                # 最终生成anchor_w: 一个四维张量，其在[b, 3, h, w]位置的值，是第几个先验框的宽度。anchor_h同理。
                anchor_h = anchors[:, 1].view(1, 3, 1, 1).repeat(batch_size, 1, in_h, in_w)

                # 构建模型最终的预测框,这里的pred_bbox是模型预测的所有框的绝对坐标
                pred_bbox = torch.zeros((*tx.shape, 4), device=l_prediction.device)
                # pred_boxes(batch_size, 3, in_h, in_w, 4)其中4为[center_x, center_y, width, height]
                pred_bbox[..., 0] = pred_dx + grid_x
                pred_bbox[..., 1] = pred_dy + grid_y
                pred_bbox[..., 2] = torch.exp(pred_dw) * anchor_w
                pred_bbox[..., 3] = torch.exp(pred_dh) * anchor_h

                # 构建目标框
                target_bbox = torch.zeros((*tx.shape, 4), device=l_prediction.device)
                target_bbox[..., 0] = tx + grid_x
                target_bbox[..., 1] = ty + grid_y
                target_bbox[..., 2] = torch.exp(tw) * anchor_w
                target_bbox[..., 3] = torch.exp(th) * anchor_h
                # --- 构建结束 ---

                # 筛选出正样本的预测框和目标框
                pred_bbox_pos = pred_bbox[mask == 1]
                target_bbox_pos = target_bbox[mask == 1]

                # 计算CIoU并得到损失
                ciou = self.bbox_ciou(pred_bbox_pos, target_bbox_pos)
                ciou_loss = 1 - ciou

                # 加权并累加定位损失
                loss_box += (ciou_loss * box_loss_scale_pos).sum()

            # 4b. 置信度损失
            # 计算正样本的置信度损失
            loss_conf_pos = self.BCELoss(pred_conf[mask == 1], torch.ones_like(pred_conf[mask==1]))

            loss_conf_neg = self.BCELoss(pred_conf[noobj_mask.type(torch.bool)],
                                         torch.zeros_like(pred_conf[noobj_mask.type(torch.bool)]))
            # noobj_mask中，我们用1初始化，即1代表背景，0代表物体，我们找到物体后，mask被置为1同时将noobj_mask置为0
            loss_conf = loss_conf + self.balance[i] * (loss_conf_pos.sum()+loss_conf_neg.sum())

            # 4c. 分类损失 (只在正样本上计算)
            if mask.sum() > 0:
                loss_cls = loss_cls + self.BCELoss(pred_cls[mask==1], t_cls[mask==1]).sum()

        # -----------------------------------------------------------#
        #   步骤 5: 汇总总损失并返回
        # -----------------------------------------------------------#
        total_loss = self.box_ratio * loss_box + self.obj_ratio * loss_conf + self.cls_ratio * loss_cls

        return total_loss

