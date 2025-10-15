import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import cv2



# 导入torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# 导入torchvision的NMS
from torchvision.ops import nms

# ==============================================================================
# 核心辅助函数 (解码, NMS, 坐标变换)
# ==============================================================================
def decode_outputs(predictions, anchors_all, num_classes, anchor_masks, strides):
    """
        解码YOLOv3模型返回的三个原始特征图。
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

    for i,prediction in enumerate(predictions):
        batch_size, _, grid_h,grid_w = prediction.shape
        stride = strides[i]
        prediction = prediction.view(batch_size,len(anchor_masks[i]),num_classes+5,grid_h,grid_w).permute(
            0, 1, 3, 4, 2).contiguous()
        # (batch_size, 3, H, W, 5 + num_classes)

        # Grid offsets
        grid_x, grid_y = torch.meshgrid(torch.arange(grid_w, device=device), torch.arange(grid_h, device=device),
                                        indexing='xy')
        grid = torch.stack((grid_x, grid_y), 2).view(1, 1, grid_h, grid_w, 2).float()

        anchor_for_layer = anchors_all[anchor_masks[i]]
        anchor_wh = anchor_for_layer.view(1, len(anchor_masks[i]),1,1,2)

        # Decode
        prediction[..., :2] = (prediction[..., :2].sigmoid() + grid) * stride
        prediction[..., 2:4] = torch.exp(prediction[..., 2:4]).clamp(max=1e3) * anchor_wh
        prediction[..., 4:] = prediction[..., 4:].sigmoid()

        # Flatten and append
        decoded_predictions.append(prediction.view(batch_size, -1, 5 + num_classes))

    return torch.cat(decoded_predictions, 1)

def xywh2xyxy(x):
    """将 [center_x, center_y, width, height] 格式转为 [x_min, y_min, x_max, y_max]"""
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def scale_resize_coords(resized_shape, box, original_shape):
    # 计算宽度和高度各自的缩放比例
    gain_w = resized_shape[1] / original_shape[1]  # gain_w = resized_w / original_w
    gain_h = resized_shape[0] / original_shape[0]  # gain_h = resized_h / original_h

    box[..., [0, 2]] = box[..., [0, 2]] / gain_w
    box[..., [1, 3]] = box[..., [1, 3]] / gain_h

    box[..., [0, 2]] = box[..., [0, 2]].clamp(0, original_shape[1])  # x1, x2
    box[..., [1, 3]] = box[..., [1, 3]].clamp(0, original_shape[0])  # y1, y2
    return box





def non_max_suppression(predictions, conf_threshold=0.25, iou_threshold=0.45, max_det=300):
    """
        对解码后的预测执行NMS
        Args:
            predictions:处理完的预测,shape(batch_size, -1, 5 + num_classes)
            conf_threshold:置信度阈值
            iou_threshold:交并比阈值
            max_det:最大检测数量
    """
    device = predictions[0].device
    out_put = [torch.zeros((0,6),device=device)]* predictions.shape[0]

    for idx, prediction in enumerate(predictions):
        prediction = prediction[prediction[...,4]>conf_threshold]
        if not prediction.shape[0]: continue

        prediction[..., 5:] = prediction[..., 4:5] * prediction[..., 5:]
        box = xywh2xyxy(prediction[..., :4])

        max_conf, max_idx = torch.max(prediction[..., 5:], dim=1,keepdim=True)

        final_prediction = torch.cat((box, max_conf, max_idx.float()),dim=1)[max_conf.view(-1)>conf_threshold]
        # [M,(x1, y1, x2, y2, final_confidence, class_index)] [M,6]

        if not final_prediction.shape[0]: continue

        # ======================================================================
        # --- 按类别NMS的核心修改 (Batched NMS) ---
        # ======================================================================
        # 1. 创建一个基于类别ID的偏移量
        # C是一个大常数, 确保不同类别的框在坐标上不会重叠
        # 例如，类别0的框坐标不变, 类别1的框坐标+C, 类别2的框坐标+2C ...
        C = final_prediction[:, 5:6] * 4096  # 4096是一个足够大的偏移量

        scores =final_prediction[..., 4]
        box_offest = final_prediction[..., :4] + C

        keep_indices = nms(box_offest, scores, iou_threshold)
        final_prediction = final_prediction[keep_indices]

        if final_prediction.shape[0] > max_det:
            final_prediction = final_prediction[:max_det]

        out_put[idx] = final_prediction

    return out_put

