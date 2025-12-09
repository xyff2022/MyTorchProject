
import torch
import numpy as np
from torchvision.ops import nms


# =============================================================================
# Part 1: 坐标转换函数 (从 YOLOv5 复用)
# =============================================================================

def xywh2xyxy(x):
    """
    将边界框从 [center_x, center_y, width, height] 格式
    转换为 [x_min, y_min, x_max, y_max] 格式

    【为什么需要这个转换】
    - YOLO 模型预测的是 xywh 格式 (中心点 + 宽高)
    - NMS 和 IoU 计算需要 xyxy 格式 (两个角点)

    Args:
        x (Tensor | ndarray): 边界框，形状 (N, 4) 或 (..., 4)
                              格式: [cx, cy, w, h]

    Returns:
        y (Tensor | ndarray): 转换后的边界框，格式: [x1, y1, x2, y2]

    【数学公式】
    x1 = cx - w/2  (中心点向左移动半个宽度 = 左边界)
    y1 = cy - h/2  (中心点向上移动半个高度 = 上边界)
    x2 = cx + w/2  (中心点向右移动半个宽度 = 右边界)
    y2 = cy + h/2  (中心点向下移动半个高度 = 下边界)

    【示例】
    输入: [100, 100, 50, 80]  # 中心(100,100), 宽50, 高80
    输出: [75, 60, 125, 140]  # 左上(75,60), 右下(125,140)

                     50
           ┌─────────────────┐
           │    (100,100)    │
        80 │        ●        │
           │                 │
           └─────────────────┘
          (75,60)        (125,140)
    """
    # 克隆输入，避免修改原始数据
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # x1 = cx - w/2
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    # y1 = cy - h/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    # x2 = cx + w/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    # y2 = cy + h/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y


def xyxy2xywh(x):
    """
    将边界框从 [x_min, y_min, x_max, y_max] 格式
    转换为 [center_x, center_y, width, height] 格式

    【用途】
    - 将 NMS 输出转换回 YOLO 格式
    - 某些可视化工具需要 xywh 格式

    Args:
        x (Tensor | ndarray): 边界框，格式: [x1, y1, x2, y2]

    Returns:
        y (Tensor | ndarray): 转换后的边界框，格式: [cx, cy, w, h]

    【数学公式】
    cx = (x1 + x2) / 2  (左右边界的中点 = 中心 x)
    cy = (y1 + y2) / 2  (上下边界的中点 = 中心 y)
    w  = x2 - x1        (右边界 - 左边界 = 宽度)
    h  = y2 - y1        (下边界 - 上边界 = 高度)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # cx = (x1 + x2) / 2
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    # cy = (y1 + y2) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    # w = x2 - x1
    y[..., 2] = x[..., 2] - x[..., 0]
    # h = y2 - y1
    y[..., 3] = x[..., 3] - x[..., 1]

    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    将归一化的 [cx_n, cy_n, w_n, h_n] 格式
    转换为像素坐标 [x1, y1, x2, y2] 格式

    【为什么需要这个函数】
    - YOLO 标签文件中的坐标是归一化的 (0~1)
    - 实际计算 IoU 和可视化需要像素坐标

    Args:
        x (Tensor | ndarray): 归一化坐标 (N, 4)，格式 [cx_n, cy_n, w_n, h_n]
                              所有值范围是 [0, 1]
        w (int): 图像宽度 (像素)
        h (int): 图像高度 (像素)
        padw (int): 水平方向的填充偏移量 (用于 letterbox)
        padh (int): 垂直方向的填充偏移量 (用于 letterbox)

    Returns:
        y (Tensor | ndarray): 像素坐标 (N, 4)，格式 [x1, y1, x2, y2]

    【数学公式】
    x1 = w * (cx_n - w_n/2) + padw
    y1 = h * (cy_n - h_n/2) + padh
    x2 = w * (cx_n + w_n/2) + padw
    y2 = h * (cy_n + h_n/2) + padh

    【示例】
    输入: [0.5, 0.5, 0.1, 0.2], w=640, h=640, padw=0, padh=0

    cx = 0.5 * 640 = 320
    cy = 0.5 * 640 = 320
    box_w = 0.1 * 640 = 64
    box_h = 0.2 * 640 = 128

    输出: [288, 256, 352, 384]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # x1 = w * (cx_n - w_n/2) + padw
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw
    # y1 = h * (cy_n - h_n/2) + padh
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh
    # x2 = w * (cx_n + w_n/2) + padw
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw
    # y2 = h * (cy_n + h_n/2) + padh
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh

    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=1e-3):
    """
    将像素坐标 [x1, y1, x2, y2] 格式
    转换为归一化的 [cx_n, cy_n, w_n, h_n] 格式

    【用途】
    - 将检测结果转换为 YOLO 标签格式
    - 数据增强后需要重新归一化坐标

    Args:
        x (Tensor | ndarray): 像素坐标 (N, 4)，格式 [x1, y1, x2, y2]
        w (int): 图像宽度
        h (int): 图像高度
        clip (bool): 是否将结果裁剪到 [0, 1] 范围
        eps (float): 防止除零的小常数

    Returns:
        y (Tensor | ndarray): 归一化坐标 (N, 4)，格式 [cx_n, cy_n, w_n, h_n]

    【数学公式】
    cx_n = ((x1 + x2) / 2) / w
    cy_n = ((y1 + y2) / 2) / h
    w_n  = (x2 - x1) / w
    h_n  = (y2 - y1) / h
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # cx_n = center_x / width
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / (w + eps)
    # cy_n = center_y / height
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / (h + eps)
    # w_n = box_width / image_width
    y[..., 2] = (x[..., 2] - x[..., 0]) / (w + eps)
    # h_n = box_height / image_height
    y[..., 3] = (x[..., 3] - x[..., 1]) / (h + eps)

    if clip:
        # 确保所有值在 [0, 1] 范围内
        if isinstance(y, torch.Tensor):
            y = y.clamp(0, 1)
        else:
            np.clip(y, 0, 1, out=y)

    return y


# =============================================================================
# Part 2: YOLOv8 专用 NMS (新增)
# =============================================================================

def non_max_suppression_v8(
        predictions,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        max_det=300,
        nc=20
):
    """
    YOLOv8 非极大值抑制 (Non-Maximum Suppression)

    【什么是 NMS】
    模型对同一个物体可能产生多个重叠的检测框。
    NMS 的作用是保留置信度最高的框，去除与其高度重叠的其他框。

    【YOLOv8 vs YOLOv5 的关键区别】
    - YOLOv5: 输出格式是 [x, y, w, h, objectness, cls0, cls1, ...]
              最终置信度 = objectness × class_conf
    - YOLOv8: 输出格式是 [x, y, w, h, cls0, cls1, ...]  (无 objectness!)
              最终置信度 = class_conf (直接使用)

    Args:
        predictions (Tensor): 模型推理输出
                            形状: (batch_size, num_anchors, 4 + nc)
                            格式: [x, y, w, h, cls0_conf, cls1_conf, ...]

                            【重要】num_anchors = 8400 (80×80 + 40×40 + 20×20)

        conf_thres (float): 置信度阈值，低于此值的检测将被过滤
                           默认 0.25，范围 [0, 1]
                           值越大，保留的框越少（更严格）

        iou_thres (float): IoU 阈值，用于 NMS 去重
                          默认 0.45，范围 [0, 1]
                          值越大，保留的重叠框越多

        classes (list[int] | None): 要保留的类别列表
                                   例如 [0, 2, 3] 表示只保留第 0、2、3 类
                                   None 表示保留所有类别

        agnostic (bool): 是否进行类别无关的 NMS
                        True:  所有类别一起做 NMS (不同类别的框也会互相抑制)
                        False: 每个类别单独做 NMS (推荐)


        max_det (int): 每张图像最多保留的检测数量
                      默认 300，防止检测过多导致后处理变慢

        nc (int): 类别数量，用于解析预测张量
                 0 表示自动从 prediction 形状推断

    Returns:
        output (list[Tensor]): 每张图像的检测结果列表
                              每个元素形状: (num_detections, 6)
                              格式: [x1, y1, x2, y2, confidence, class_id]

    【处理流程】
    1. 解析预测张量，分离坐标和类别置信度
    2. 按置信度过滤低分框
    3. xywh -> xyxy 坐标转换
    4. 执行 NMS 去除重叠框
    5. 限制最大检测数量

    【示例】
    >>> model.eval()
    >>> with torch.no_grad():
    ...     pred = model(images)  # (1, 8400, 84) for nc=80
    >>> results = non_max_suppression_v8(pred, conf_thres=0.25, iou_thres=0.45)
    >>> for det in results:
    ...     print(det.shape)  # (num_boxes, 6)
    """

    # =========================================================================
    # 0. 参数校验和初始化
    # =========================================================================

    # 检查置信度阈值的合法性
    assert 0 <= conf_thres <= 1, f'conf_thres 必须在 [0, 1] 范围内, 当前值: {conf_thres}'
    assert 0 <= iou_thres <= 1, f'iou_thres 必须在 [0, 1] 范围内, 当前值: {iou_thres}'

    # 获取设备和数据类型
    device = predictions.device

    # 获取 batch size
    bs = predictions.shape[0]

    # 推断类别数量 (如果未指定)
    # prediction 形状: (bs, num_anchors, 4 + nc)
    # 所以 nc = prediction.shape[2] - 4
    nc = nc or (predictions.shape[2] - 4)

    # 初始化输出列表，每个元素对应一张图像
    output = [torch.zeros((0, 6), device=device)] * bs

    # =========================================================================
    # 1. 遍历 batch 中的每张图像
    # =========================================================================

    for i, prediction in enumerate(predictions):
        # prediction 形状: (num_anchors, 4 + nc) 例如 (8400, 84)

        # ---------------------------------------------------------------------
        # 1.1 提取置信度并进行初步过滤
        # ---------------------------------------------------------------------

        # 【YOLOv8 关键区别】直接使用类别置信度，无 objectness
        # x[:, 4:] 是所有类别的置信度，形状 (8400, nc)
        # .amax(1) 在类别维度取最大值，得到每个框的最高置信度
        # 形状: (8400,)

        # 单标签模式(每个框只能属于一个类别)
        # 获取每个框的最大类别置信度和对应类别索引
        max_conf, max_conf_idx = prediction[:, 4:].max(dim = 1)
        # max_conf: (8400,) - 每个框的最大置信度
        # max_conf_idx: (8400,) - 每个框的预测类别索引

        # 过滤低置信度的框
        mask = max_conf > conf_thres

        # 应用掩码
        prediction = prediction[mask]
        max_conf = max_conf[mask]  # 保留对应的置信度
        max_conf_idx = max_conf_idx[mask]  # 保留对应的类别索引

        # 如果过滤后没有剩余框，跳到下一张图
        if prediction.shape[0] == 0:
            continue

        # ---------------------------------------------------------------------
        # 1.2 坐标转换: xywh -> xyxy
        # ---------------------------------------------------------------------

        # 提取边界框坐标 (前 4 个值)
        box = prediction[:, :4]  # (8400, 4) 格式 [cx, cy, w, h]

        # 转换为 xyxy 格式
        box = xywh2xyxy(box)  # (8400, 4) 格式 [x1, y1, x2, y2]

        # ---------------------------------------------------------------------
        # 1.3 组合成最终格式: [x1, y1, x2, y2, max_conf, class]
        # ---------------------------------------------------------------------

        # 确保 max_conf 和 max_conf_idx 是正确的形状
        # (8400,) -> (8400, 1)
        if max_conf_idx.dim() == 1:
            max_conf_idx= max_conf_idx.unsqueeze(1)
        if max_conf.dim() == 1:
            max_conf= max_conf.unsqueeze(1)
        # 拼接: [box(4), conf(1), class(1)] -> (n, 6)
        nms_prediction = torch.cat((box, max_conf, max_conf_idx.float()), dim = 1)

        # ---------------------------------------------------------------------
        # 1.4 执行 NMS
        # ---------------------------------------------------------------------

        # 【Batched NMS 技巧】
        # 为了让不同类别的框不互相抑制，我们给每个类别的框加上一个巨大的偏移
        # 这样同类别的框坐标接近，不同类别的框坐标相距很远
        c = nms_prediction[:, 5:6] * 4096
        nms_boxes = nms_prediction[:, :4] + c

        # 提取置信度用于 NMS 排序
        scores = nms_prediction[:, 4]  # (n,)

        # 调用 torchvision 的 NMS 函数
        # 返回保留框的索引
        keep = nms(nms_boxes, scores, iou_thres)

        # 限制最大检测数量
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        # 保存结果
        output[i] = nms_prediction[keep]

    return output


# =============================================================================
# Part 3: NMS 辅助函数
# =============================================================================

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    将边界框从一个图像尺寸缩放到另一个图像尺寸

    【用途】
    推理时使用 letterbox 将图像调整为模型输入尺寸 (如 640x640)，
    得到的检测框坐标是相对于 640x640 的。
    此函数将坐标缩放回原始图像尺寸。

    Args:
        img1_shape (tuple): 模型输入尺寸 (height, width)，如 (640, 640)
        boxes (Tensor): 检测框 (n, 4)，格式 [x1, y1, x2, y2]
        img0_shape (tuple): 原始图像尺寸 (height, width)
        ratio_pad (tuple | None): (ratio, (padw, padh))
                                 如果提供，直接使用；否则自动计算

    Returns:
        boxes (Tensor): 缩放后的检测框

    【示例】
    模型输入: 640x640 (letterbox 后)
    原始图像: 1920x1080
    检测框: [100, 100, 200, 200] (相对于 640x640)

    缩放后: 检测框坐标对应到 1920x1080
    """

    if ratio_pad is None:
        # 计算缩放比例 (取较小的，确保图像完全在画布内)
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        # 计算填充量
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,  # padw
               (img1_shape[0] - img0_shape[0] * gain) / 2)  # padh
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    # 减去填充
    boxes[..., [0, 2]] -= pad[0]  # x1, x2 减去水平填充
    boxes[..., [1, 3]] -= pad[1]  # y1, y2 减去垂直填充

    # 除以缩放比例
    boxes[..., :4] /= gain

    # 裁剪到图像边界内
    clip_boxes(boxes, img0_shape)

    return boxes


def clip_boxes(boxes, shape):
    """
    将边界框裁剪到图像边界内

    Args:
        boxes (Tensor): 边界框 (n, 4)，格式 [x1, y1, x2, y2]
        shape (tuple): 图像形状 (height, width)

    【为什么需要裁剪】
    缩放后的坐标可能超出图像边界，需要裁剪到有效范围内
    """

    if isinstance(boxes, torch.Tensor):
        # x1, x2 裁剪到 [0, width]
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        # y1, y2 裁剪到 [0, height]
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:
        # NumPy 版本
        boxes[..., 0] = np.clip(boxes[..., 0], 0, shape[1])
        boxes[..., 2] = np.clip(boxes[..., 2], 0, shape[1])
        boxes[..., 1] = np.clip(boxes[..., 1], 0, shape[0])
        boxes[..., 3] = np.clip(boxes[..., 3], 0, shape[0])