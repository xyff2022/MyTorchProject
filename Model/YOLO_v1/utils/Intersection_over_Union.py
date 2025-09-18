import torch


def intersection_over_union(box_pred, box_label, box_format = "yolo"):
    """
        计算预测框和真实框之间的交并比(IoU)

        args:
            box_pred (torch.Tensor): 预测的边界框张量，形状为 (N, S, S, 4)，N是批量大小。
            box_label (torch.Tensor): 真实的边界框张量，形状为 (N, S, S, 4)。
            box_format (str): 边界框的格式。可选值为 "yolo" ([x, y, w, h]) 或 "voc" ([x1, y1, x2, y2])。
                         默认为 "yolo"。
        return:
            tensor: 每个框对应的IoU值, 形状为 (N, S, S, 1)
        """
    if box_format == "yolo":
        box_pred_x1 = box_pred[..., 0:1] - box_pred[..., 2:3] / 2
        # 取出x轴靠近左边的点的坐标
        box_pred_x2 = box_pred[..., 0:1] + box_pred[..., 2:3] / 2
        # 取出x轴靠近右边的点的坐标
        box_pred_y1 = box_pred[..., 1:2] - box_pred[..., 3:4] / 2
        # 取出y轴靠近下边的点的坐标
        box_pred_y2 = box_pred[..., 1:2] + box_pred[..., 3:4] / 2
        # 取出y轴靠近上边的点的坐标
        box_label_x1 = box_label[..., 0:1] - box_label[..., 2:3] / 2
        box_label_x2 = box_label[..., 0:1] + box_label[..., 2:3] / 2
        box_label_y1 = box_label[..., 1:2] - box_label[..., 3:4] / 2
        box_label_y2 = box_label[..., 1:2] + box_label[..., 3:4] / 2

    x1 = torch.max(box_label_x1, box_pred_x1)
    x2 = torch.min(box_label_x2, box_pred_x2)
    y1 = torch.max(box_label_y1, box_pred_y1)
    y2 = torch.min(box_label_y2, box_pred_y2)

    intersection_width = (x2 - x1).clamp(0)
    intersection_height = (y2 - y1).clamp(0)
    intersection = intersection_width * intersection_height

    box_pred_area = abs((box_pred_x2 - box_pred_x1) * (box_pred_y2 - box_pred_y1))
    box_label_area = abs((box_label_x2 - box_label_x1) * (box_label_y2 - box_label_y1))

    union = box_pred_area + box_label_area - intersection + 1e-6
    iou = intersection / union

    return iou


if __name__ == '__main__':
    # 1. 定义真实框 (Ground Truth Label)
    # 假设数据集中有一个物体的标注框。
    # 格式为 [x_center, y_center, width, height]，所有值都已归一化。
    # 这个框的中心在图片的 (0.5, 0.5) 位置，宽高都是图片的40%。
    box_label = torch.tensor([
        [0.5, 0.5, 0.4, 0.4]  # 这是一个包含单个真实框的批次
    ])

    # 2. 定义模型的预测框 (Model's Prediction)
    # 假设模型对这个物体做出了一个比较接近但不完美预测。
    # 预测框的中心稍微偏右下方，宽度稍大，高度稍小。
    box_pred = torch.tensor([
        [0.52, 0.48, 0.44, 0.36]  # 这是一个包含单个预测框的批次
    ])

    # 3. 调用函数并传入输入值
    # 我们使用默认的 "midpoint" 格式，因为我们的输入就是YOLO格式。
    iou = intersection_over_union(box_pred, box_label, box_format="yolo")

    # 4. 打印结果
    print(f"真实框 (Ground Truth): {box_label.numpy()}")
    print(f"预测框 (Prediction): {box_pred.numpy()}")
    print(f"计算出的IoU值为: {iou.numpy()}")
