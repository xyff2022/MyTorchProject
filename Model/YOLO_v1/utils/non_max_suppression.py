import torch

from Model.YOLO_v1.utils.calculate_iou import calculate_iou


def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    对边界框列表执行非极大值抑制 (NMS)。

    Args:
        bboxes (list): 包含所有预测框的列表，
                       每个元素格式为 [class_id, confidence, x1, y1, x2, y2]。
        iou_threshold (float): 用于抑制重叠框的IoU阈值。
        threshold (float): 置信度阈值，过滤掉低置信度的框。

    Returns:
        list列表，形状为
        {[class_id, confidence, x1, y1, x2, y2],
        [class_id, confidence, x1, y1, x2, y2],
        [class_id, confidence, x1, y1, x2, y2],
        [class_id, confidence, x1, y1, x2, y2],}
    """

    # 确保 bboxes 是一个列表
    assert type(bboxes) == list

    # 1. 按类别ID对所有边界框进行分组
    bboxes_by_class = {}
    for box in bboxes:
        class_id = box[0]
        if class_id not in bboxes_by_class:
            bboxes_by_class[class_id] = []
        bboxes_by_class[class_id].append(box)

    final_bboxes = []

    # 2. 对每个类别独立进行NMS
    for class_id, class_bboxes in bboxes_by_class.items():
        class_bboxes = sorted(class_bboxes,key = lambda x: x[1], reverse=True)

        while class_bboxes:
            chosen_bbox = class_bboxes.pop(0)
            # chosen_bboxes [class_id, confidence, x1, y1, x2, y2]

            if chosen_bbox[1] < threshold:
                continue

            final_bboxes.append(chosen_bbox)

            if not class_bboxes:
                continue

            # 1. 将所有剩下的框转换为一个张量
            remaining_boxes_tensor = torch.tensor([box[2:] for box in class_bboxes])
            chosen_box_tensor = torch.tensor(chosen_bbox[2:]).unsqueeze(0)

            # 2. 一次性计算IoU
            # chosen_box_tensor shape: (1, 4)
            # remaining_boxes_tensor shape: (N, 4)
            # ious shape: (1, N)
            ious = calculate_iou(chosen_box_tensor, remaining_boxes_tensor)

            # 3. 筛选出IoU低于阈值的框
            # ious.squeeze(0) < iou_threshold 会返回一个布尔张量，例如 [True, False, True]
            boxes_to_keep = ious.squeeze(0) < iou_threshold

            # 4. 使用布尔索引来更新列表
            class_bboxes = [box for i, box in enumerate(class_bboxes) if boxes_to_keep[i]]

    return final_bboxes



