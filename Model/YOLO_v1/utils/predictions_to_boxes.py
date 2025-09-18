import torch
from torchvision.ops import nms

def predictions_to_boxes(predictions, image_size,threshold = 0.5, S = 7, B = 2, C = 20, ):
    """
    将YOLOv1模型的原始输出张量转换为最终的边界框列表。

    Args:
        predictions (torch.Tensor): 模型的输出张量，形状为 (S, S, C + B*5)。不考虑batchsize
        threshold (float): 置信度阈值，低于此值的预测将被忽略。
        S (int): 网格尺寸。
        B (int): 每个网格单元预测的边界框数量。
        C (int): 类别数。
        image_size:

    returns:
        eval_list: list，长度为N。每个元素是一个字典，包含一张图片的最终预测结果：
              {'boxes': torch.Tensor, 'scores': torch.Tensor, 'labels': torch.Tensor}
    """

    # --- 步骤 1 & 2: 解码张量并收集所有批次中的有效预测框 ---

    # 将预测张量从 (1470) 变形为 (49, 30) 以方便遍历
    predictions = predictions.reshape(S * S, C + B * 5)

    all_bboxes = []
    for cell_idx in range(S * S):# 遍历 49 个网格单元
        grid_row = cell_idx // S
        grid_col = cell_idx % S
        # 解码类别信息
        # 获取当前单元格的20个类别概率
        classes = predictions[cell_idx, :C]
        # 找到概率最高的类别ID和对应的概率值
        best_score, best_class_idx = torch.max(classes, dim = 0)

        # 4. 遍历该单元格内的所有预测器 (B=2)
        # --- 对每个bounding box进行解码 ---
        for b in range(B):
            # 获取当前预测器的物体置信度P(Object)
            # 第一个预测器的置信度在索引20，第二个在索引25
            confidence = predictions[cell_idx, C + b * 5]

            # 计算最终置信度并应用阈值
            final_confidence = confidence * best_score
            if final_confidence > threshold:
                # 5. 解码坐标
                # 获取 [x_cell, y_cell, w_norm, h_norm]
                box_pred = predictions[cell_idx, C + b * 5 + 1 : C + b * 5 + 5]

                # 将相对坐标转换为绝对像素坐标 (假设图片尺寸为448x448)


                x_center = (grid_col + box_pred[0]) * (image_size / S)
                y_center = (grid_row + box_pred[1]) * (image_size / S)
                width = box_pred[2] * image_size
                height = box_pred[3] * image_size

                # 转换为 (xmin, ymin, xmax, ymax) 格式
                xmin = x_center - width * 0.5
                ymin = y_center - height * 0.5
                xmax = x_center + width * 0.5
                ymax = y_center + height * 0.5

                all_bboxes.append(
                    [
                        best_class_idx.item(),
                        final_confidence.item(),
                        xmin.item(),
                        ymin.item(),
                        xmax.item(),
                        ymax.item(),
                     ]
                )


    return all_bboxes




