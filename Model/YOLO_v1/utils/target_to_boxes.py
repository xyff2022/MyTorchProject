import torch

def target_to_boxes(target, image_size=448, S=7, C=20):
    """
    一个专门解析YOLOv1真实标签(target)张量的函数，将其转换为边界框列表。

    这个函数逻辑非常直接，因为它不需要处理多个候选框或置信度阈值。

    Args:
        target (torch.Tensor): 真实标签张量，形状为 (S, S, C + 5) 或 (S, S, 30)，
                               但只会使用前 C+5 的部分。
        image_size (int): 图像的尺寸 (例如 448)，用于将相对坐标转换回像素坐标。
        S (int): 网格尺寸 (通常为 7)。
        C (int): 类别数 (通常为 20)。

    Returns:
        list: 包含这张图片所有真实边界框的列表。
              每个元素格式为 [class_id, confidence, x1, y1, x2, y2]。
              这里的 confidence 对于真实标签来说，永远是 1.0。
    """
    # 初始化一个空列表，用于存放解码出的真实边界框
    all_bboxes = []

    # --- 1. 遍历所有网格单元 ---
    # 我们使用两个嵌套循环来遍历 7x7 网格中的每一个单元格
    for i in range(S):  # i 代表网格的行 (row)
        for j in range(S):  # j 代表网格的列 (col)

            # --- 2. 检查该单元格是否有物体 ---
            # 按照约定，target张量在索引 C (即第20)的位置存放置信度信息。
            # 如果值为1，说明这个网格单元负责检测一个真实物体。
            if target[i, j, C] == 1:

                # --- 3. 如果有物体，则解码信息 ---
                # a. 获取类别ID
                # 类别信息存储在索引 0 到 C-1 的位置，是一个 one-hot 向量。
                # 我们使用 torch.argmax() 来找到值为1的那个位置的索引，即类别ID。
                class_id = torch.argmax(target[i, j, :C]).item()

                # b. 获取坐标 [x_cell, y_cell, w_norm, h_norm]
                # 坐标信息存储在索引 C+1 到 C+4 的位置。
                box_coords = target[i, j, C+1 : C+5]

                # c. 将坐标从相对值转换为绝对像素坐标
                # 原始坐标是相对于网格单元的，我们需要将其转换回整张图的像素坐标。
                x_center = (j + box_coords[0]) * (image_size / S)
                y_center = (i + box_coords[1]) * (image_size / S)
                width = box_coords[2] * image_size
                height = box_coords[3] * image_size

                # d. 将中心点坐标和宽高转换为 (xmin, ymin, xmax, ymax) 格式
                # 这是评估时通常需要的格式。
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2

                # --- 4. 将解码后的框添加到列表中 ---
                # 我们将所有信息打包成一个列表，并添加到总列表中。
                # 注意：真实标签的置信度我们直接设为 1.0。
                all_bboxes.append(
                    [
                        class_id,
                        1.0,
                        xmin.item(),
                        ymin.item(),
                        xmax.item(),
                        ymax.item(),
                    ]
                )

    # --- 5. 返回包含所有真实框的列表 ---
    return all_bboxes



