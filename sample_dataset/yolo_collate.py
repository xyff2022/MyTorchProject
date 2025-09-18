import torch
from torch.utils.data import DataLoader

# 导入您之前提供的 VocDataset 类
# 假设 voc_to_yolo_dataset.py 文件与此文件在同一目录下
from Model.YOLO_v1.voc_to_yolo_dataset import VocDataset2YOLO


def yolo_collate_fn(batch, S=7, C=20):
    """
    为 YOLOv1 模型自定义的 collate_fn 函数。
    将 Dataset 返回的 (image, target_dict) 列表，打包成模型训练所需的批处理张量。

    参数:
        batch (list): 一个列表，其中每个元素都是一个元组 (image, target_dict)。
                      - image 是一个 PyTorch 张量。
                      - target_dict 是一个包含 "boxes" 和 "labels" 的字典。
        S (int): YOLOv1 模型的网格尺寸。
        C (int): 数据集的类别总数。

    返回:
        tuple: 一个包含两个元素的元组 (images_batch, yolo_target_tensor)。
               - images_batch: 形状为 (N, 3, H, W) 的图片批处理张量。
               - yolo_target_tensor: 形状为 (N, S, S, C+5) 的 YOLO 格式目标张量。
    """
    # --- 步骤 1: 分离图片和标签 ---
    # 初始化两个空列表，分别用于存放一个批次中的所有图片和所有 target 字典
    images = []
    targets = []

    # 遍历 dataloader 传来的原始批次数据
    for sample in batch:
        # 将图片部分添加到 images 列表
        images.append(sample[0])
        # 将 target 字典部分添加到 targets 列表
        targets.append(sample[1])

    # --- 步骤 2: 堆叠图片张量 ---
    # 使用 torch.stack 将图片列表堆叠成一个大的张量
    # 例如，如果 batch_size=4, 图片是 3x448x448，堆叠后 images_batch 的形状就是 (4, 3, 448, 448)
    images_batch = torch.stack(images, 0)

    # --- 步骤 3: 初始化 YOLO 目标张量 ---
    # 获取批次大小 N (batch_size)
    batch_size = len(targets)
    # 获取数据增强/变换后的图像高度 H 和宽度 W
    # 我们假设批次内所有图像尺寸都相同，这通常由 transform 中的 Resize 操作保证
    _, _, H, W = images_batch.shape
    # 创建一个全零的目标张量，准备接收转换后的 YOLO 格式标签
    # 形状为 (批次大小, 网格行数, 网格列数, 类别数 + 5个边界框参数)
    yolo_target_tensor = torch.zeros(batch_size, S, S, C + 5)

    # --- 步骤 4: 遍历批次，填充 YOLO 目标张量 ---
    # 使用 enumerate 遍历批次中的每一个 target 字典，idx 是其在批次中的索引 (0, 1, 2, ...)
    for idx, target_dict in enumerate(targets):
        # 获取当前样本的所有边界框，形状为 (num_boxes, 4)
        boxes = target_dict["boxes"]
        # 获取当前样本的所有类别标签，形状为 (num_boxes,)
        labels = target_dict["labels"]

        # 遍历当前图片中的每一个边界框
        for i in range(len(boxes)):
            # 获取单个边界框的坐标 (xmin, ymin, xmax, ymax)
            # 注意：这里的坐标是相对于图像尺寸 (H, W) 的像素值
            xmin, ymin, xmax, ymax = boxes[i]
            # 获取对应的类别标签
            label = labels[i]

            # --- 步骤 4a: 坐标转换 ---
            # 将 (xmin, ymin, xmax, ymax) 转换为归一化的 (center_x, center_y, width, height)
            width_norm = (xmax - xmin) / W
            height_norm = (ymax - ymin) / H
            center_x_norm = (xmin + xmax) / (2 * W)
            center_y_norm = (ymin + ymax) / (2 * H)

            # --- 步骤 4b: 计算网格信息 ---
            # 计算物体中心点所在的网格单元的行(grid_j)和列(grid_i)
            # 注意 x 对应列，y 对应行
            grid_i = int(S * center_x_norm)
            grid_j = int(S * center_y_norm)

            # 计算中心点在单元格内部的相对坐标 (x_cell, y_cell)
            x_cell = S * center_x_norm - grid_i
            y_cell = S * center_y_norm - grid_j

            # --- 步骤 4c: 填充张量 ---
            # 检查该单元格是否已被占用 (YOLOv1的核心规则：一个单元格只负责一个物体)
            # yolo_target_tensor[idx, grid_j, grid_i, C] 是置信度位
            if yolo_target_tensor[idx, grid_j, grid_i, C] == 0:
                # 1. 将置信度位设置为 1，表示这个网格包含一个物体
                yolo_target_tensor[idx, grid_j, grid_i, C] = 1

                # 2. 进行 one-hot 编码，将对应类别的索引位设置为 1
                yolo_target_tensor[idx, grid_j, grid_i, label] = 1

                # 3. 将4个坐标信息打包成一个张量
                coords = torch.tensor([x_cell, y_cell, width_norm, height_norm])

                # 4. 将坐标张量填入目标张量的相应位置 (C+1 到 C+5)
                yolo_target_tensor[idx, grid_j, grid_i, C + 1:] = coords

    # --- 步骤 5: 返回最终结果 ---
    return images_batch, yolo_target_tensor


if __name__ == '__main__':
    # ================== 这是一个如何使用的示例 ==================
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from functools import partial

    # 假设这是您的类别列表
    VOC_CLASSES = ["person", "car", "dog", "cat"]  # 示例，共4个类别

    # 1. 定义数据增强/转换
    # 注意：在 VocDataset 中，boxes 是以像素坐标 (xmin, ymin, xmax, ymax) 传入的
    # 所以 bbox_params 的 format 应该是 'pascal_voc'
    transform = A.Compose([
        A.Resize(448, 448),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # 2. 实例化数据集 (请替换为您的真实路径)
    # 为了能运行此示例，您需要创建一个假的目录结构和文件
    # image_folder/1.jpg, label_folder/1.xml

    TRAIN_IMG_DIR = r"../data/PASCAL_VOC/train/images"
    TRAIN_LABEL_DIR = r"../data/PASCAL_VOC/train/labels"
    TRAIN_CSV_FILE = r"../data/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
    try:
        dataset = VocDataset2YOLO(
            TRAIN_CSV_FILE ,
            TRAIN_IMG_DIR ,
            TRAIN_LABEL_DIR ,
            transform=transform
        )

        # 3. 使用 partial 创建一个绑定了 S 和 C 参数的 collate_fn
        # 这样 DataLoader 在调用它时就不需要我们再手动传 S 和 C 了
        collate_fn_with_params = partial(yolo_collate_fn, S=7, C=len(VOC_CLASSES))

        # 4. 创建 DataLoader，并将自定义的 collate_fn 传入
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn_with_params
        )

        # 5. 从 DataLoader 中获取一个批次的数据进行验证
        images, yolo_targets = next(iter(data_loader))

        # 6. 打印形状以验证结果
        print("测试成功！")
        print(f"图片批处理张量形状: {images.shape}")
        print(f"YOLO目标张量形状: {yolo_targets.shape}")

        # 验证形状是否符合预期
        assert images.shape == (4, 3, 448, 448)
        assert yolo_targets.shape == (4, 7, 7, len(VOC_CLASSES) + 5)
        print("\ncollate_fn 测试通过！输出形状正确，可以直接用于YOLOv1损失计算。")

    except (FileNotFoundError, IndexError) as e:
        print("\n测试失败，请确保：")
        print("1. 您已将 'path/to/your/images' 和 'path/to/your/labels' 替换为真实路径。")
        print("2. 您的数据集文件夹中至少包含一个样本。")
        print(f"原始错误: {e}")
