import torch


def calculate_iou(x, y):
    """
    计算两组边界框之间的 IoU 矩阵。

        Args:
            x (torch.Tensor): 预测的边界框，形状为 [N, 4]。
            y (torch.Tensor): 真实的边界框，形状为 [M, 4]。

        Returns:
            torch.Tensor: IoU 矩阵，形状为 [N, M]。
    """
    # 扩展维度以利用广播机制
    # x: [N, 4] -> [N, 1, 4]
    # y: [M, 4] -> [1, M, 4]
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    # 计算交集的坐标
    intersection_min = torch.max(x[..., :2], y[..., :2])  #->[N,M,2(xmin,ymin)]
    intersection_max = torch.min(x[..., 2:], y[..., 2:])  #->[N,M,2(xmax,ymax)]

    # 计算交集的宽和高，如果 non-positive 则设为0  ->[N,M,2]
    intersection_wh = (intersection_max - intersection_min).clamp(min = 0)

    # 计算交集面积  ->[N,M]
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    # 计算各自的面积
    x_area = ((x[..., 2] - x[..., 0]) *
                        (x[..., 3] - x[..., 1]))   #-> [N, 1]
    y_area = ((y[..., 2] - y[..., 0]) *
                   (y[..., 3] - y[..., 1]))   #-> [1, M]

    # 计算并集面积
    union = x_area + y_area - intersection

    # 计算 IoU，避免除以零
    iou = intersection / (union + 1e-9)  #-> [N, M]

    return iou