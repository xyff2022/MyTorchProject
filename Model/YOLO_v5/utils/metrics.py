# 导入所有必需的库
import torch
import torchvision
from torchmetrics.detection import MeanAveragePrecision
import math

from torchvision.ops import nms
from tqdm import tqdm


def xywh2xyxy_torch(x):
    """
    (Torch version) 将中心点格式 [x_center, y_center, w, h]
    转换为 [x1, y1, x2, y2]

    Args:
        x (Tensor): [N, 4] 形状的张量, 格式为 xywh

    Returns:
        y (Tensor): [N, 4] 形状的张量, 格式为 xyxy
    """
    # 1. 创建一个和输入 x 形状相同、值全为 0 的张量 y，用于存放结果
    y = torch.zeros_like(x, device=x.device)

    # 2. 计算 x1 (左上角 x)
    # x[:, 0] 是所有框的 x_center
    # x[:, 2] 是所有框的 width
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x

    # 3. 计算 y1 (左上角 y)
    # x[:, 1] 是所有框的 y_center
    # x[:, 3] 是所有框的 height
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y

    # 4. 计算 x2 (右下角 x)
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x

    # 5. 计算 y2 (右下角 y)
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    # 6. 返回转换后的张量
    return y


def xywhn2xyxy_torch(x, w=640, h=640):
    """
    (Torch version) 将归一化的 [x_center_n, y_center_n, w_n, h_n]
    转换为像素坐标 [x1, y1, x2, y2]

    Args:
        x (Tensor): [N, 4] 格式的 Bounding boxes (xywhn)
        w (int): 图像宽度
        h (int): 图像高度

    Returns:
        y (Tensor): [N, 4] 格式的 Bounding boxes (xyxy)
    """

    # 1. 创建一个和输入 x 形状相同、值全为 0 的张量 y
    y = torch.zeros_like(x, device=x.device)

    # 2. 计算 x1
    # (x[:, 0] - x[:, 2] / 2) 得到归一化的 x1_n
    # ... * w 将其转换为像素坐标
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x

    # 3. 计算 y1
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # top left y

    # 4. 计算 x2
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x

    # 5. 计算 y2
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # bottom right y

    # 6. 返回结果
    return y


# ==============================================================================
# 核心辅助函数 (解码, NMS, 坐标变换)
# ==============================================================================
def decode_outputs(predictions, anchors_all, num_classes, anchor_masks, strides):
    """
        解码YOLOv5模型返回的三个原始特征图。
        Args:
            predictions (list of torch.Tensor): 模型的三个原始输出.
            anchors_all : 所有的anchor box尺寸.
            num_classes (int): 类别数.
            anchor_masks (list): 每个输出层使用的anchor索引.
            strides (list): 每个输出层的步长.

        Returns:
            torch.Tensor: 解码后的所有预测框, shape [batch_size, N, 5 + num_classes].
                          格式为 [cx, cy, w, h, obj_conf, class_conf, ...].
    """

    device = predictions[0].device
    decoded_predictions = []

    for i, prediction in enumerate(predictions):
        batch_size, _, grid_h, grid_w, _ = prediction.shape
        stride = strides[i]

        # Grid offsets
        grid_x, grid_y = torch.meshgrid(torch.arange(grid_w, device=device), torch.arange(grid_h, device=device),
                                        indexing='xy')
        grid = torch.stack((grid_x, grid_y), 2).view(1, 1, grid_h, grid_w, 2).float()

        anchor_for_layer = anchors_all[anchor_masks[i]]
        anchor_wh = anchor_for_layer.view(1, len(anchor_masks[i]), 1, 1, 2)

        # 复制-份 prediction，因为我们将进行 in-place (原地) 操作
        p = prediction.clone()

        # (sigmoid(tx) * 2 - 0.5 + grid) * stride
        p[..., :2] = (p[..., :2].sigmoid() * 2. - 0.5 + grid) * stride

        # (sigmoid(twh) * 2)**2 * anchor
        p[..., 2:4] = (p[..., 2:4].sigmoid() * 2.) ** 2 * anchor_wh

        # (obj_conf 和 class_conf)
        p[..., 4:] = p[..., 4:].sigmoid()
        # --- 结束修改 ---

        # Flatten and append
        decoded_predictions.append(p.view(batch_size, -1, 5 + num_classes))

    return torch.cat(decoded_predictions, 1)


def non_max_suppression(predictions, conf_threshold=0.25, iou_threshold=0.45, max_det=300):
    """
        对解码后的预测执行NMS
        Args:
            predictions:处理完的预测,shape(batch_size, -1, 5 + num_classes)
            conf_threshold:置信度阈值
            iou_threshold:交并比阈值
            max_det:最大检测数量
    """
    # 准备一个空列表，长度为 batch_size
    device = predictions[0].device
    out_put = [torch.zeros((0, 6), device=device)] * predictions.shape[0]

    # 遍历批次中的每张图
    for idx, prediction in enumerate(predictions):
        # [!!!] 核心修改 (v5 逻辑): "multi_label" (多标签) 过滤

        # 1. 计算所有类别的最终得分
        # prediction 形状是 [N, 5 + num_classes] (e.g., [25200, 25])
        # prediction[:, 4:5] 是 obj_conf, shape [25200, 1]
        # prediction[:, 5:] 是 cls_conf, shape [25200, 20]
        box_score = prediction[:, 4:5] * prediction[:, 5:]  # 结果 shape [25200, 20]

        # 2. 找到所有得分 > conf_threshold 的 (box_index, class_index) 组合
        # 假设第100个框的"人"(类0)得分0.9, "瓶子"(类4)得分0.8
        # 且 conf_threshold = 0.25
        # .nonzero() 会返回:
        # tensor([[100, 0],   <-- (框100, 类别0)
        #         [100, 4],   <-- (框100, 类别4)   ...])
        # .nonzero() 会扫描这个 [25200, 20] 的 True/False 表，找出所有值为 True 的格子的“坐标”
        # T后，i拿到所有框的索引，j拿到所有类别的索引
        # j现在表示类别
        i, j = (box_score > conf_threshold).nonzero(as_tuple=False).T

        M = i.shape[0]  # 候选框总数
        NMS_CANDIDATE_LIMIT = 50000  # 限制 NMS 最多处理 50k 个候选框
        if M > NMS_CANDIDATE_LIMIT:
            # 仅保留得分最高的 top-k 个候选框
            scores = box_score[i, j]  # 获取这 M 个框的实际分数
            _, topk_indices = torch.topk(scores, k=NMS_CANDIDATE_LIMIT)

            # 用 top-k 的索引来更新 i 和 j
            i = i[topk_indices]
            j = j[topk_indices]

        # 3. 组合成 [M, 6] 的格式 (M 是找到的组合总数)
        # 以上述例子为例
        # 转换所有被选中的box的格式，现在box 是 xyxy 格式
        box = xywh2xyxy(prediction[i, :4])
        final_prediction = torch.cat((
            box,  # 拿出框 100 的 xywh
            box_score[i, j, None],  # 拿出框 100 的 相乘后的conf (分别是 0.9 和 0.8)
            j[:, None].float()  # 拿出类别 (分别是 0 和 4)
        ), dim=1)

        if not final_prediction.shape[0]:
            continue

        # 4. [!!!] 核心修改 (v3 逻辑): "Batched NMS" (按类别 NMS)
        # C 是一个基于类别 ID 的巨大偏移量
        C = final_prediction[:, 5:6] * 4096  # 4096 是一个足够大的偏移量

        # (box + C) 使得 NMS 只在同-类别的框之间进行
        box_offset = final_prediction[:, :4] + C
        scores = final_prediction[..., 4]

        # NMS 现在在 box_offset 上运行，它能正确区分不同类别
        keep_indices = nms(box_offset, scores, iou_threshold)

        # 5. 过滤
        final_prediction = final_prediction[keep_indices]

        # 6. 限制最大检测数量
        if final_prediction.shape[0] > max_det:
            final_prediction = final_prediction[:max_det]

        out_put[idx] = final_prediction

    return out_put


def xywh2xyxy(x):
    """将 [center_x, center_y, width, height] 格式转为 [x_min, y_min, x_max, y_max]"""
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


