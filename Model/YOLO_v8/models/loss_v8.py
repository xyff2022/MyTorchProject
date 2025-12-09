import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
from torch import meshgrid

from torch.nn import Module

from .common_v8 import DFL


# 1. 计算 IoU (Intersection over Union) 的工具函数
#    这个函数支持 CIoU, DIoU, GIoU 等
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算两个框之间的 IoU / CIoU
    box1: (n, 4) 预测框
    box2: (n, 4) 真实框
    xywh: 如果为 True，输入格式为 (cx, cy, w, h)，否则为 (x1, y1, x2, y2)
    """
    # 获取坐标
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2
        b1_x1, b1_x2, b1_y1, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_x2, b2_y1, b2_y2 = box2.chunk(4, -1)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:
                v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU

    return iou

class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self,
                 topk: int = 13,
                 num_classes: int = 80,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 eps: float = 1e-9,
                 ciou = True):
        """模型先预测一次，然后根据**“预测得准不准”**（分类分数高 + IoU大）来动态选择最好的 k 个格子作为正样本.
        Args:
            topk (int): 每个真实物体(GT)最多选择多少个正样本。v8 默认为 10。
            num_classes (int): 类别数量。
            alpha (float): 对齐度量公式中，分类得分的权重幂次。默认为 0.5。
            beta (float): 对齐度量公式中，IoU 的权重幂次。默认为 6.0。
                          (注意：beta 很大，说明 v8 非常看重定位精度！)
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.background_idx = num_classes
        self.use_ciou = ciou

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        选择中心点落在 GT 边界框内部的锚点。
        Args:
            xy_centers: 所有锚点的中心坐标 (8400, 2)
            gt_bboxes: 真实框坐标 (bs, n_max_boxes, 4)
        Returns:
            (bs, 8400, n_max_boxes) 的布尔掩码。如果第 i 个锚点在第 j 个 GT 内部，则为 True。
        """

        # number_anchors = 8400
        number_anchors = xy_centers.size(0)
        # bs = 批次大小, number_boxes = 每张图最大的 GT 数量
        bs, number_boxes, _ = gt_bboxes.size()

        # 1. 获得 GT 的左上角 (lt) 和 右下角 (rb)
        #    gt_bboxes 格式通常是 xyxy 或 xywh，这里假设传入前已经转为 [x1, y1, x2, y2]
        #    view(-1, 1, 4) 是为了方便后面广播
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)

        # 2. 计算 锚点中心 与 GT 边界 的距离
        #    xy_centers[None] -> (1, number_anchors = 8400, 2)
        #    lt               -> (bs*number_boxes, 1, 2)
        #    我们构建一个巨大的距离矩阵：
        #    bbox_deltas: (bs, number_boxes, number_anchors = 8400, 4)
        #    4 个值分别代表：(anchor_x - x1), (anchor_y - y1), (x2 - anchor_x), (y2 - anchor_y)
        #    只要这 4 个值都大于 0，说明中心点在框内。
        bbox_deltas=torch.cat((xy_centers[None]-lt,rb-xy_centers[None]),dim=3
                              ).view(bs,number_boxes,number_anchors,-1)

        # 3. 判断是否都在框内 (amin > eps)
        #    .amin(3): 在第 3 维(4个距离)取最小值。如果最小值 > 0，说明都在框内。
        #    .gt_(eps): 生成布尔值
        return bbox_deltas.amin(3).gt_(eps)

    def get_box_metric(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts):
        """
        计算对齐度量: alignment_metrics = s^alpha * u^beta
        """
        # pd_scores: 预测分数 (bs, 8400, num_classes)
        # pd_bboxes: 预测框 (bs, 8400, 4)
        # gt_labels: 真实标签 (bs, number_gt_bboxes, 1)
        # gt_bboxes: 真实框 (bs, number_gt_bboxes, 4)
        # mask_in_gts: 上一步得到的掩码 (bs, 8400, number_gt_bboxes)

        bs = pd_scores.size(0)
        na = pd_scores.size(1)  # 8400
        number_gt_bboxes = gt_bboxes.size(1)

        # ============================================================
        # 核心步骤 A: 准备广播 (Broadcasting) 计算 IoU
        # ============================================================
        # 我们要计算 8400 个预测框 和 number_gt_bboxes 个真实框 两两之间的 IoU。
        # 这需要把它们扩展成 (bs, 8400, number_gt_bboxes, 4) 的形状。

        # pd_bboxes: (bs, 8400, 4) -> unsqueeze(2) -> (bs, 8400, 1, 4)
        # 含义：对于每个锚点，只有 1 个预测框，准备好和 number_gt_bboxes 个 GT 比较。
        pd_boxes = pd_bboxes.unsqueeze(2)

        # gt_bboxes: (bs, number_gt_bboxes, 4) -> unsqueeze(1) -> (bs, 1, number_gt_bboxes, 4)
        # 含义：对于每个 GT，把它复制 8400 份，准备和每个锚点比较。
        gt_boxes = gt_bboxes.unsqueeze(1)

        # 计算 IoU
        # bbox_iou 返回形状: (bs, 8400, number_gt_bboxes, 1) -> squeeze(-1) -> (bs, 8400, number_gt_bboxes)
        if self.use_ciou:  # v8 默认 True
            # 计算 CIoU
            iou = bbox_iou(pd_boxes, gt_boxes, xywh=False, CIoU=True).squeeze(-1)
        else:
            iou = bbox_iou(pd_boxes, gt_boxes, xywh=False).squeeze(-1)

        # 限制 IoU 大于 0 (防止计算 log 或幂次出错)
        iou = iou.clamp(0)

        # ============================================================
        # 核心步骤 B: 提取 GT 对应类别的分数 ("隔山打牛")
        # ============================================================

        # gt_labels: (bs, number_gt_bboxes, 1) -> squeeze -> (bs, number_gt_bboxes)
        # 存的是类别的索引，比如 [0, 5, ...] 代表第 0 类和第 5 类
        gt_labels = gt_labels.long().squeeze(-1)

        # 扩展 label 维度以匹配 pd_scores
        # (bs, number_gt_bboxes) -> (bs, number_gt_bboxes, 1) -> repeat -> (bs, 8400, number_gt_bboxes)
        # 这一步是为了造一个索引矩阵，告诉 PyTorch 每个位置该取哪个通道的分数
        # 含义：对于 8400 个锚点，针对第 j 个 GT，我们都应该去查第 label[j] 个类别的分数
        target_labels = gt_labels.unsqueeze(-1).repeat(1, na, 1)

        # 调整 pd_scores 形状
        # (bs, 8400, 80) -> transpose -> (bs, 80, 8400)
        pd_scores_tran = pd_scores.transpose(1, 2)

        # 输出的形状和 index 一模一样，只有在 dim 那个维度上，坐标由 index 里的值决定，其他维度的坐标保持不变。
        target_scores = torch.gather(pd_scores_tran, 1, target_labels)
        # out (bs, 8400, number_gt_bboxes)

        # 3. 计算对齐度量 (公式)  t = s^\alpha \times u^\beta
        align_metric = target_scores.pow(self.alpha) * iou.pow(self.beta)


        # align_metric * mask_in_gts (第 1 个返回值):
        # 用途：这是真正用来做 select_topk (挑前10名) 的分数。包含了“能力”和“资格”的双重考量。
        # iou (第 2 个返回值):
        # 用途：用来计算后续的归一化系数。即使有些点被 mask 掉了，保留原始 IoU 数据有时为了统计或调试方便，
        # align_metric (第 3 个返回值):
        # 用途：这是原始的、没被 mask 阉割过的分数。通常用于调试或者某些特殊变体的 loss 计算，
        return align_metric * mask_in_gts, iou, align_metric

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        选取每个 GT 的 Top-K 个 Anchor。
        """
        # metrics: (bs, number_anchors, number_gt_bboxes) - 每个 anchor 对每个 GT 的得分
        # topk_mask (bs, number_gt_bboxes, 1)

        # 1. 获取维度信息
        num_anchors = metrics.shape[1]  # 8400
        # metrics.shape[2] 是 number_gt_bboxes (GT的数量)

        # ============================================================
        # 阶段 A: Top-K 选拔 (谁分高谁入围)
        # ============================================================

        # torch.topk(input, k, dim)
        # input: metrics (bs, number_anchors, number_gt_bboxes)
        # k: self.topk
        # dim: 1 (沿着 number_anchors 个锚点的维度选，即为每个 GT 选出最好的锚点)

        # topk_metrics: (bs, self.topk, number_gt_bboxes) -> 选出来的分数
        # topk_idxs:    (bs, self.topk, number_gt_bboxes) -> 选出来的锚点索引 (0~number_anchors-1)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=1, largest=largest)

        # 2. 生成 Top-K 掩码 (is_in_topk)
        #    我们想要一个 (bs, 8400, number_gt_bboxes) 的矩阵，选中的位置是 1，没选中的是 0
        is_in_topk = torch.zeros_like(metrics, dtype=torch.long)
        for b in range(len(metrics)):
            # 在 Python 中，当你对一个多维数组使用 len() 函数时，它永远只返回第 0 维（第一维）的长度。
            # bs  b 代表当前是第几张图 (0, 1, 2 ... bs-1)
            # scatter_ 是 gather 的逆操作，把 1 填回 topk_idxs 指定的位置
            #     tensor.scatter_(dim, index, src)
            # 含义：将 src 的值，根据 index（索引位置），写入到 tensor（目标张量）中。
            # 下划线 _：代表 In-place（原地操作），即它会直接修改 tensor 本身的数据，而不会创建新的副本。
            is_in_topk[b].scatter_(1, topk_idxs[b], 1)

        # 3. 过滤无效 GT
        # topk_mask 形状: (bs, number_gt_bboxes)
        if topk_mask is not None:
            is_in_topk = is_in_topk * topk_mask.transpose(1, 2)

        # 4. 再次过滤：metric 几乎为 0 的不要 (通常是背景)
        is_in_topk = torch.where(metrics>1e-9, is_in_topk, torch.zeros_like(is_in_topk))

        # ==================== 冲突处理 (核心) ====================
        # 场景：两个物体靠得很近，第 555 号锚点同时进入了 物体A 和 物体B 的 Top-10。
        # is_in_topk.sum(-1) > 1 意味着某一行(某个anchor)有多个 1
        # mask  (bs, number_anchors)的布尔掩码。True 表示这个点有冲突
        mask = is_in_topk.sum(dim=-1) > 1

        if mask.any():
            # Any (任何一个)。
            # 功能：检查张量（Tensor）中是否至少包含一个 True（或者非 0 值）。

            # 1. 找到冲突锚点心中“最爱”的那个 GT
            # metrics: (bs, 8400, max_GT)
            # argmax(-1): 在 GT 维度上找最大值的索引。
            # 结果 conflict_gt_idx: (bs, 8400)
            # 含义: 每个锚点分数最高的那个 GT 是第几个？
            conflict_layer = is_in_topk.clone()
            conflict_layer[~mask] = 0
            # 在 conflict_layer 这张表上，把所有没有冲突的锚点数据全部清零。

            # 对于冲突的 anchor，看它对哪个 GT 的 metric 最高
            # argmax(-1) 返回 metric 最大的那个 GT 的索引
            conflict_gt_idx = metrics.argmax(-1)
            # conflict_gt_idx：形状 (bs, 8400)。
            # 它告诉我们：对于每一个锚点（不管有没有冲突），它心中得分最高的那个 GT 是谁。

            # 构建一个新的掩码，只有胜出的那个 GT 是 1
            not_conflict_mask = torch.zeros_like(is_in_topk)
            for b in range(len(metrics)):
                not_conflict_mask[b].scatter_(1, conflict_gt_idx[b].unsqueeze(-1), 1)

            # 更新 is_in_topk
            # 逻辑：(非冲突区域保持原样) + (冲突区域替换为胜出者)
            is_in_topk = (is_in_topk * ~mask.unsqueeze(-1)) + (not_conflict_mask * ~mask.unsqueeze(-1))

        # =======================================================

        # 5. 生成最终输出
        # final_mask: (bs, 8400) - 只要是任意一个 GT 的正样本，就是 True
        # is_in_topk: (bs, 8400, max_GT)
        # sum(-1): 沿着最后一个维度求和 -> (bs, 8400)
        final_mask = is_in_topk.sum(dim=-1) > 0
        # 如果是 0：说明这个锚点对任何 GT 都不是正样本
        # 如果是 1：说明这个锚点被分配给了某一个 GT
        # 物理含义：final_mask 是一张“身份卡”。它不关心你负责哪个物体，只关心“你是不是正样本”**。

        # target_gt_idx: (bs, 8400) - 每个 anchor 到底负责第几个 GT？
        # 如果不是正样本，这个索引无意义 (后面会被 mask 掉)
        target_gt_idx = is_in_topk.argmax(dim=-1)
        # 告诉系统，每个锚点对应的那个 1 到底在第几列

        return target_gt_idx, final_mask, is_in_topk

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, final_mask):
        """
        根据分配结果，提取对应的 GT 标签和坐标。
        """
        # gt_labels: (bs, number_gt_bboxes, 1)
        # target_gt_idx: (bs, 8400) - 每个样本分配到的 GT 索引
        # final_mask: (bs, 8400) - 正样本掩码

        # 1. 获取维度信息
        bs, number_anchors = final_mask.shape
        number_gt_bboxes = gt_bboxes.shape[1]  # 比如 10

        # 2. 生成 Batch 偏移量 (Global Indexing)
        #    batch_index: [[0], [1], [2]...] (形状: bs, 1)
        batch_index = torch.arange(end=bs, dtype=torch.int64, device=gt_labels.device)[:, None]

        #    target_gt_idx 变成了全局唯一的索引
        #    第 0 张图的索引不变 (0~9)
        #    第 1 张图的索引变成 (10~19)
        target_gt_idx = batch_index * number_gt_bboxes + target_gt_idx
        # (BatchSize, 8400) 它的数值不再是简单的 0, 1, 2（局部编号），而是变成了跨越整个 Batch 的 全局编号。

        # 3. 提取 标签 (Class)
        #    先把 gt_labels (bs, number_gt_bboxes, 1) 拍扁成 (bs*number_gt_bboxes) 的一维数组
        #    然后用全局索引 target_gt_idx 去“抓药”
        #    结果 target_cls: (bs, 8400)
        target_cls = gt_labels.long().flatten()[target_gt_idx]
        # 当我们执行 gt_labels.long().flatten()[target_gt_idx] 时，PyTorch 会做以下操作：
        # 遍历 target_gt_idx 这个大表里的每一个格子。
        # 查找：看格子里写的数字是几（比如 3）。
        # 提取：去 flat_labels 仓库的第 3 号位置把货物（数值 5）拿出来。
        # 填入：把拿到的 5 填回到结果张量的对应位置。

        # 4. 提取 坐标 (BBox)
        #    先把 gt_bboxes (bs, 10, 4) 拍扁成 (bs*10, 4)
        #    然后抓药
        #    结果 target_bboxes: (bs, 8400, 4)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # 5. 生成 One-Hot 分数
        #    把类别 ID (比如 2) 变成向量 [0, 0, 1, 0...]
        #    结果 target_scores: (bs, 8400, 20)
        target_scores = F.one_hot(target_cls, self.num_classes)

        return target_cls, target_bboxes, target_scores

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, gt_mask):
        """
        Args:
            pd_scores: 预测分数 (bs, 8400, 20)
            pd_bboxes: 预测框 (bs, 8400, 4)
            anc_points: 锚点中心 (8400, 2)
            gt_labels: (bs, number_gt_bboxes, 1)
            gt_bboxes: (bs, number_gt_bboxes, 4)
            gt_mask: (bs, number_gt_bboxes, 1) - 标记哪些是真实的 GT (因为 batch 中可能有 padding)
        """

        self.bs = pd_scores.size(0)
        self.number_gt_bboxes = gt_bboxes.size(1)

        # 0. 异常处理：如果没有 GT (空图)，直接返回全 0
        if self.number_gt_bboxes == 0:
            return (torch.full_like(pd_scores[..., 0], self.background_idx).to(pd_scores.device),
                    torch.zeros_like(pd_bboxes).to(pd_bboxes.device),
                    torch.zeros_like(pd_scores).to(pd_scores.device),
                    torch.zeros_like(pd_scores[..., 0]).to(pd_scores.device),
                    torch.zeros_like(pd_scores[..., 0]).to(pd_scores.device))

        # 1. 第一步：粗筛 (select_candidates_in_gts)
        #    mask_in_gts: 形状 (bs, 8400, number_gt_bboxes)，如果在 GT 内则为 True
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, self.eps)

        # (为了兼容后续变量名) mask_pos 表示那些有效的、且在 GT 内部的位置
        mask_pos = mask_in_gts * gt_mask

        # 2. 第二步：计算度量 (get_box_metric)
        #    align_metric: 对齐度量值 t (已经乘以了 mask_in_gts)
        #    overlaps: IoU (未过滤)(bs, 8400, number_gt_bboxes)
        #    mask_in_gts * mask_gt: 确保只计算有效的 GT（过滤 padding 和框外的）
        align_metric, overlaps, _ = self.get_box_metric(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_pos)

        # 3. 第三步：精选 Top-K
        #    target_gt_idx: 正样本分配给了哪个 GT
        #    fg_mask: 正样本掩码
        target_gt_idx, final_mask, is_in_topk=self.select_topk_candidates(align_metric,topk_mask=gt_mask.bool())

        # 4. 第四步：生成目标
        target_cls, target_bboxes, target_scores=self.get_targets(gt_labels,gt_bboxes,target_gt_idx,final_mask)

        # ==================== 软标签生成 (Soft Label) ====================
        # 对齐度量 (align_metric) 的值域是不确定的 (比如 0.1 ~ 0.8)
        # 我们希望把它归一化，作为 BCE Loss 的 Target

        # 确保只计算正样本
        align_metric = align_metric * is_in_topk

        # 找到每个 GT 内部最大的 metric 值 (Max t)
        # max_metrics: (bs, 8400, 1) -> 注意：这里维度取决于具体实现，逻辑上是找到归属该 GT 的最大值
        max_metrics = align_metric.amax(dim=-1, keepdim=True)
        # .amax() 是 "Array Maximum" 的缩写，它的作用非常纯粹：只返回指定维度上的最大值，而不返回索引。

        # 找到每个 GT 内部最大的 IoU 值 (Max IoU)
        max_iou = (overlaps * is_in_topk).amax(dim=-1, keepdim=True)

        # 归一化公式: t_norm = t * (max_iou / max_t)
        norm_align_metric = (max_metrics * align_metric / (max_iou + 1e-9)).amax(dim=-1, keepdim=True)
        # (max_metrics * align_metric / (max_iou + 1e-9))形状(bs, 8400, number_gt_bboxes)
        # 表示每一个锚点和每一个 GT 之间的归一化后的得分）
        # 但最后一列number_gt_bboxes被处理后，只有一个得分

        # 最终的 target_scores = one_hot_class * normalized_metric
        # 比如：本来是 [0, 1, 0] (猫)，现在变成 [0, 0.85, 0] (置信度为 0.85 的猫)
        target_scores = target_scores * norm_align_metric

        return target_cls, target_bboxes, target_scores, final_mask.bool(), norm_align_metric


class v8loss(Module):
    def __init__(self, model):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.detect.stride
        self.nc = model.detect.nc
        self.number_out = model.detect.no
        self.reg_max = model.detect.reg_max
        self.device = next(model.parameters()).device

        self.box_gain = 7.5
        self.cls_gain = 0.5
        self.dfl_gain = 1.5

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

        self.dfl = DFL(self.reg_max)

        # ============== 创建 DFL 期望计算的索引张量 ==============
        # 这个张量用于计算离散分布的期望值
        # 例如 reg_max=16 时，dfl_proj = [0, 1, 2, ..., 15]
        # 后续通过 matmul(softmax(pred), dfl_proj) 得到期望距离
        # 使用 register_buffer 注册为 buffer，这样它会:
        # 1. 随模型保存/加载
        # 2. 随 .to(device) 自动移动到正确设备
        # 3. 但不会被优化器更新 (不是可学习参数)
        self.register_buffer(
            'dfl_proj',
            torch.arange(self.reg_max, dtype=torch.float)
        )

    def preprocess(self, targets, batch_size, scale_tensor):
        # targets (n,6) scale_tensor[640, 640, 640, 640]
        # ==========================================
        # 1. 异常处理：空 Batch
        # ==========================================
        # targets.shape[0] 是物体总数。如果为0，说明这批图片全是背景，没有任何物体。
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            # ==========================================
            # 2. 确定维度：这批图里，物体最多的一张有几个？
            # ==========================================
            # i 是 targets 的第 0 列，也就是 "图片索引" (0, 0, 0, 1, 1, 2...)
            image_idx = targets[:, 0]

            # unique(return_counts=True) 是关键函数。
            # 它可以统计每个索引出现的次数。
            # x = torch.tensor([0, 0, 0, 1, 1])
            # 假设这是 targets[:, 0] (图片索引)意思是: 第 0 张图有 3 个物体，第 1 张图有 2 个物体
            # unique_elements, counts = x.unique(return_counts=True)
            # (unique_elements). tensor([0, 1])含义: 这个 Batch 里包含了 第0张图 和 第1张图
            # (counts)tensor([3, 2]) 0的出现了3次(第0张图有3个物体)  1的出现了2次(第1张图有2个物体)
            _, counts = image_idx.unique(return_counts=True)
            # tensor.max() (没有参数，或者只有 keepdim 参数)全局最大值 (Global Max)


            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            # 此时 out 全是 0。
            # 形状: (BatchSize, Max_Objects, 5)
            # 最后一维的 5 代表: [class_id, x, y, w, h] (img_index 已经被转化为 batch 维度了)

            # ==========================================
            # 3. 填表：搬运数据
            # ==========================================
            # 遍历这批数据中的每一张图片
            for i in range(batch_size):
                # 创建一个布尔掩码 matches
                # 如果 targets 第 0 列等于当前图片索引 j，则为 True
                match = (i == image_idx)

                # 计算这张图实际有几个物体
                n = match.sum()

                # 如果这张图有物体 (n > 0)
                if n:
                    # 核心搬运代码：
                    # targets[matches, 1:] -> 选出属于这张图的所有行，并且丢掉第0列(img_index)
                    # out[i, :n]           -> 把数据填入 out 的第 j 行，的前 n 个位置
                    # 剩下的位置 (n 之后) 保持初始化时的 0，这就是 Padding
                    out[i, :n] = targets[match, 1:]

            # ==========================================
            # 4. 坐标变换：反归一化
            # ==========================================
            # 此时 out[..., 1:5] 里的坐标还是 0~1 的相对值 (x, y, w, h)
            # scale_tensor 比如是 [640, 640, 640, 640]
            # .mul_ 是原地乘法
            # 结果：坐标变成了像素值 (例如 320.5, 320.5, 100, 100)
            # 这样做是为了方便后续计算 IoU (IoU 对尺度不敏感，但 DFL 和中心点距离计算通常用像素尺度)
            out[..., 1:5] = out[..., 1:5].mul_(scale_tensor)


        return out

    def forward(self, predictions, targets):
        """
        计算损失函数的入口

        Args:
            predictions: 模型输出的特征图列表 [P3, P4, P5]
                   P3 shape: (Batch, 64+nc, 80, 80)
                   P4 shape: (Batch, 64+nc, 40, 40)
                   P5 shape: (Batch, 64+nc, 20, 20)
            targets: 真实标签数据 (Ground Truth)
                   通常是 (M, 6) 的 Tensor -> [img_idx, cls_id, x, y, w, h]

        Returns:
            total_loss: 标量, 用于反向传播
            loss_items: (3,) 的 Tensor, 包含 [box_loss, cls_loss, dfl_loss] 用于显示
        """

        loss = torch.zeros(3, device=self.device)  # 初始化损失容器 [box, cls, dfl]
        feats = predictions[1] if isinstance(predictions, tuple) else predictions  # 确保 preds 是特征图列表

        # =============================================================
        # 1. 准备工作: 生成 Anchors 并重塑预测结果
        # =============================================================

        # 生成所有特征层对应的锚点坐标 (x, y) 和步长张量
        # anchor_points: (8400, 2)    8400 = 6400(P3) + 1600(P4) + 400(P5)
        # stride_tensor: (8400, 1)
        anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)

        # =============================================================
        # 第 2 步: 拼接与维度调整
        # =============================================================
        # 将三个特征图拼接成一个大张量
        # 原始: P3 (bs, 84, 80, 80), P4 (bs, 84, 40, 40), P5 (bs, 84, 20, 20)
        #    view: (bs, 84, h, w) -> (bs, 84, h*w)
        #    split: 分离 reg_dist (64通道) 和 cls_score (20通道)
        pred = torch.cat([xi.view(xi.shape[0], self.number_out, -1) for xi in feats], dim=2)
        pred_distri, pred_scores = pred.split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (bs, 8400, nc)

        # 获取数据类型和 batch 大小
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # 计算原始图像尺寸 (用于坐标缩放)
        # feats[0].shape[2:] = (80, 80) for P3 层
        # 乘以 stride[0]=8 得到原图尺寸 (640, 640)
        image_size = torch.tensor(feats[0].shape[2:], dtype=dtype, device=self.device) * self.stride[0]

        # =============================================================
        # 第 3 步: 处理 GT Targets
        # =============================================================
        # scale_tensor: [h, w, h, w] 用于缩放 cx, cy, w, h
        # 注意顺序是 [1, 0, 1, 0] 因为 shape[2:] 返回 (h, w)
        target = self.preprocess(targets, batch_size, image_size[[1, 0, 1, 0]])

        # 分离标签和坐标
        # targets: (bs, max_obj, 5) -> [cls, cx, cy, w, h]
        # gt_labels: (bs, max_obj, 1)
        # gt_bboxes: (bs, max_obj, 4) -> [cx, cy, w, h]
        gt_labels, gt_bboxes = target.split((1, 4), dim=2)

        gt_mask = gt_bboxes.sum(dim=2, keepdim=True).gt_(0)
        # (batch, max_obj, 1)[bool]

        # =============================================================
        # 第 4 步: 解码预测框 (DFL)
        # =============================================================
        pred_ltrb = self.dfl(pred_distri)  # (bs, 4, 8400)
        pred_ltrb = pred_ltrb.permute(0, 2, 1)  # (bs, 8400, 4)

        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (bs, 8400, 64)

        # decoded_bboxes: (bs, 8400, 4) [xyxy 格式，像素尺度]
        decoded_bboxes=distance2bbox(pred_ltrb,anchor_points.unsqueeze(0),xywh=False)*stride_tensor.unsqueeze(0)

        # =============================================================
        # 第 5 步: 转换 GT 坐标格式 (xywh -> xyxy)
        # =============================================================
        # gt_bboxes 是 [cx, cy, w, h] 格式
        # Assigner 需要 [x1, y1, x2, y2] 格式
        gt_bboxes_xyxy = torch.zeros_like(gt_bboxes)
        gt_bboxes_xyxy[..., 0] = gt_bboxes[..., 0] - gt_bboxes[..., 2] / 2  # x1 = cx - w/2
        gt_bboxes_xyxy[..., 1] = gt_bboxes[..., 1] - gt_bboxes[..., 3] / 2  # y1 = cy - h/2
        gt_bboxes_xyxy[..., 2] = gt_bboxes[..., 0] + gt_bboxes[..., 2] / 2  # x2 = cx + w/2
        gt_bboxes_xyxy[..., 3] = gt_bboxes[..., 1] + gt_bboxes[..., 3] / 2  # y2 = cy + h/2

        # =============================================================
        # 第 6 步: 动态标签分配 (Task-Aligned Assigner)
        # =============================================================
        target_cls, target_bboxes, target_scores, final_mask, _ = self.assigner(
            pred_scores.sigmoid(),  # 分类分数 (过 sigmoid)
            decoded_bboxes,  # 预测框 (xyxy, 像素尺度)
            anchor_points * stride_tensor,  # 锚点 (像素尺度)
            gt_labels,  # GT 标签
            gt_bboxes_xyxy,  # GT 框 (xyxy)
            gt_mask  # 有效 GT 掩码
        )

        # 计算目标分数总和 (用于归一化，防止除零)
        target_scores_sum = target_scores.sum().clamp(min = 1.0)

        # =============================================================
        # 第 7 步: 计算 Loss
        # =============================================================
        if final_mask.sum()>0:
            # ---------------------------------------------------------
            # (a) Box Loss (CIoU Loss)
            # ---------------------------------------------------------
            # 1 - CIoU 作为损失
            ciou = bbox_iou(
                decoded_bboxes[final_mask],# 预测框 (只取正样本)
                target_bboxes[final_mask],# 目标框 (只取正样本)
                xywh=False,
                CIoU=True
            )
            loss[0] = (1.0 - ciou).sum()

            # ---------------------------------------------------------
            # (b) DFL Loss
            # ---------------------------------------------------------
            # 将 target_bboxes (xyxy, 像素尺度) 转换回 ltrb 距离 (特征图尺度)

            # 1.获取正样本对应的锚点和步长
            # anchor_points 原始形状: (8400, 2) - 特征图尺度
            # stride_tensor 原始形状: (8400, 1)
            # final_mask 形状: (bs, 8400) - 布尔掩码，True 表示正样本

            # 需要把 anchor_points 扩展到 batch 维度，然后用 final_mask 索引
            # expand: 共享内存（只是不同的"视图"）
            anchor_points_expanded = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
            # (8400, 2) → (1, 8400, 2) → (bs, 8400, 2)

            stride_expanded = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            # (8400, 1) → (1, 8400, 1) → (bs, 8400, 1)

            # 只取正样本
            dfl_anchor = anchor_points_expanded[final_mask]  # (num_pos, 2)
            dfl_stride = stride_expanded[final_mask]  # (num_pos, 1)
            dfl_target_bboxes = target_bboxes[final_mask]  # (num_pos, 4) [xyxy 像素尺度]

            # 2.将 xyxy 坐标转回 ltrb 距离
            #   xyxy 坐标 [x1, y1, x2, y2] (像素尺度)
            #       锚点坐标 [ax, ay] (特征图尺度)
            #       步长 stride
            #
            #   ltrb 距离 [l, t, r, b] (特征图尺度)
            target_ltrb = torch.cat([
                dfl_anchor[:, 0:1] - dfl_target_bboxes[:, 0:1] / dfl_stride,  # l
                dfl_anchor[:, 1:2] - dfl_target_bboxes[:, 1:2] / dfl_stride,  # t
                dfl_target_bboxes[:, 2:3] / dfl_stride - dfl_anchor[:, 0:1],  # r
                dfl_target_bboxes[:, 3:4] / dfl_stride - dfl_anchor[:, 1:2]  # b
            ], dim=1)

            # 3: 限制范围
            # DFL 只能预测 [0, reg_max) 范围内的距离
            # reg_max=16 表示最大预测 14.99 个格子的距离
            target_ltrb = target_ltrb.clamp(0, self.reg_max - 1 - 0.01)

            # 计算 DFL Loss
            loss[2] = self.dfl_loss(
                pred_distri[final_mask].view(-1, 4, self.reg_max),
                target_ltrb
            )

        # ---------------------------------------------------------
        # (c) Cls Loss (BCE Loss)
        # ---------------------------------------------------------
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum()

        # =============================================================
        # 第 8 步: 归一化与加权
        # =============================================================
        loss[0] *= self.box_gain / target_scores_sum  # Box Loss
        loss[1] *= self.cls_gain / target_scores_sum  # Cls Loss
        loss[2] *= self.dfl_gain / target_scores_sum  # DFL Loss

        # 返回总损失和各项损失
        return loss.sum() * batch_size, loss.detach()

    def dfl_loss(self, pred_dist, target_ltrb):
        """
        Distribution Focal Loss

        【核心思想】
        不是直接回归一个连续值，而是学习一个离散分布。
        目标是让分布的期望值等于真实距离。

        Args:
            pred_dist: (num_pos, 4, reg_max)
                       模型对 4 个方向各预测 reg_max 个 logits
                       例如: (100, 4, 16) - 100个正样本，4个方向，16个分布值

            target_ltrb: (num_pos, 4)
                         真实的 ltrb 距离 (特征图尺度)
                         例如: (100, 4) - 100个正样本的 [l, t, r, b]

        Returns:
            loss: 标量
        """

        # 假设 target_ltrb 某个值是 2.3
        tl = target_ltrb.long()  # floor(2.3) = 2 (左邻居)
        tr = tl + 1  # 2 + 1 = 3 (右邻居)

        # 计算权重(线性插值)
        wl = tr - target_ltrb  # 3 - 2.3 = 0.7 (左权重)
        wr = 1 - wl             # 1 - 0.7 = 0.3 (右权重)

        # F.cross_entropy(input, target)
        # input: (N, C) - N 个样本，C 个类别
        # target: (N,) - N 个样本的类别索引
        # pred_dist: (num_pos, 4, 16)
        # 需要 reshape 成 (num_pos*4, 16) 来计算 CE
        ce_l = F.cross_entropy(
            pred_dist.view(-1, self.reg_max),
            tl.view(-1),
            reduction='none'
        ).view(tl.shape)  # (num_pos, 4)

        ce_r = F.cross_entropy(
            pred_dist.view(-1, self.reg_max),
            tr.view(-1),
            reduction='none'
        ).view(tl.shape)

        # 最终 DFL Loss mean(-1),在最后一个维度取平均值
        loss = (ce_l * wl + ce_r * wr).mean(-1, keepdim=True).sum()
        #       ↑         ↑
        #    左邻居损失   右邻居损失
        #    权重 0.7    权重 0.3

        return loss




    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """
        生成锚点 (Anchor Points) 和 步长张量 (Stride Tensor)

        Args:
            feats: 列表，包含三个特征图 [P3, P4, P5]
                   例如 P3 shape: (bs, 84, 80, 80)
            strides: 列表，包含三个步长 [8, 16, 32]
                     例如 P3 步长为 8，意味着特征图上 1 个单位 = 原图 8 个像素
            grid_cell_offset: 网格中心偏移量，0.5 表示中心

        Returns:
            anchor_points: (Total_Anchors, 2) 所有层的网格中心点 (cx, cy)
            stride_tensor: (Total_Anchors, 1) 所有层的步长
        """
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device

        for i, stride in enumerate(strides):
            # 获取当前特征图的宽 (w) 和 高 (h)
            # 例如 P3 层: _, _, 80, 80 = feats[0].shape
            _, _, h, w = feats[i].shape

            # 生成 x 和 y 的坐标序列
            # sx (x 坐标): 生成从 0 到 w-1 的序列，并加上偏移量 0.5。这是每个格子中心的 x 坐标。
            # sy (y 坐标): 生成从 0 到 h-1 的序列，并加上偏移量 0.5。
            sx = torch.arange(0, w, 1, dtype=dtype, device=device) + grid_cell_offset
            sy = torch.arange(0, h, 1, dtype=dtype, device=device) + grid_cell_offset

            # 生成网格矩阵
            # meshgrid (网格化): 将 sx 和 sy 结合，生成一个完整的网格。
            # indexing='ij' 参数很重要，它决定了返回的顺序是 (y, x) 还是 (x, y)。
            # 这里 sy 在前，sx 在后，使用 'ij' 索引，生成的 sy 形状是 (h, w)，sx 形状也是 (h, w)。
            # 此时 sy 矩阵里的值是每一行的 y 坐标，sx 矩阵里的值是每一列的 x 坐标。
            sy, sx = meshgrid(sy, sx, indexing='ij')

            # 堆叠并压扁，得到 (Total_Cells_In_Layer, 2)
            # torch.stack((sx, sy), -1): 在最后一个维度堆叠。形状变为 (h, w, 2)。每个元素都是 [x, y]。
            # .view(-1, 2): 压扁前两个维度。形状变为 (h*w, 2)。
            # 例如 80*80 = 6400，所以这一层会生成 6400 个 (cx, cy) 坐标点。
            anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))

            # 对于这一层的所有 h * w 个点，它们的缩放倍数都是相同的（即当前的 stride）。我们需要创建一个同样长度的张量来存储这个值。
            # torch.full(...): 创建一个形状为 (h*w, 1) 的张量，里面所有的值都填充为 stride（例如 8）。
            stride_tensor.append(torch.full((h * w, 1), stride, device=device, dtype=dtype))

        # 拼接列表中的张量
        return torch.cat(anchor_points), torch.cat(stride_tensor)


def distance2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox