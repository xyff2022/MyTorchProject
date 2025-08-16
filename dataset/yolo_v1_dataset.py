import os

import cv2
import torch
from torch.utils.data import Dataset


class YoloV1Dataset(Dataset):
    def __init__(self, image_folder, label_folder, s=7, c=20, transform=None):
        """
            初始化YOLOv1的数据集类。

            参数:
                image_folder (str): 图片文件所在的文件夹路径。
                label_folder (str): 标签文件所在的文件夹路径。
                S (int): 网格尺寸 (默认为7)。
                C (int): 类别总数 (默认为20)。
                transform (albumentations.Compose): 应用于图片和边界框的数据增强流水线。
        """
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.S = s
        self.C = c
        self.transform = transform
        self.image_names = os.listdir(self.image_folder)

    def __len__(self):
        """返回数据集中图片的数量。"""
        return len(self.image_names)

    def __getitem__(self, index):
        """
            根据索引获取一个处理好的样本 (图片, 目标张量)。
        """
        # --- 步骤 1 & 2: 定位并读取原始数据 ---
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_folder, image_name)

        # 使用OpenCV读取图片，因为Albumentations基于它
        image = cv2.imread(image_path)

        # 将BGR格式转换为更通用的RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = image_name.split(".")[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_name)

        # 从标签文件中读取所有边界框
        bounding_box = []
        with open(label_path, "r", encoding= "UTF-8") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bounding_box.append([class_id, center_x, center_y, width, height])


        # --- 步骤 3: 应用数据增强 ---

            # Albumentations需要将bounding_box和它们的类别分开传入
            # 为保证后续使用的bounding_box的一致性，将数据分离提取出来
            class_labels = [box[0] for box in bounding_box]
            bounding_box = [box[1:] for box in bounding_box]

        if self.transform:
            transformed = self.transform(
                image = image,
                bboxes = bounding_box,
                class_labels = class_labels
            )
            image = transformed["image"]
            bounding_box = transformed["bboxes"]
            class_labels = transformed["class_labels"]

        # --- 步骤 4: 构建YOLO格式的Target张量 ---
        # 初始化一个全零的目标张量，形状为 (S, S, C+5)
        # C个类别概率 + 1个置信度 + 4个坐标值
        target = torch.zeros((self.S, self.S, self.C + 5))

        for box_idx, box in enumerate(bounding_box):
            class_label = class_labels[box_idx]
            x, y, width, height = box

            # 计算物体中心点所在的网格单元的行(i)和列(j)
            i = int(self.S * x)
            j = int(self.S * y)

            # 计算中心点在单元格内部的相对坐标 (center_x, center_y)
            # 单元格偏移量”,通过分而治之,极大地降低了学习难度,稳定了训练过程,并充分利用了卷积网络的空间特性
            center_x = self.S * x - i
            center_y = self.S * y - j

            # 检查该单元格是否已被占用 (YOLOv1的核心规则：一个单元格只负责一个物体)
            if target[i, j, self.C] == 0:
                target[i, j, self.C] = 1
                target[i, j, int(class_label)] = 1
                target[i, j, self.C + 1 : self.C + 5] =torch.tensor(
                    [center_x, center_y, width, height]
                )

        return image, target


# --- 单元测试 ---
if __name__ == '__main__':
    import albumentations as a
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    # 1. 定义一个简单的数据增强和预处理流水线用于测试
    transform = a.Compose(
        [
            a.Resize(width = 448, height = 448),
            a.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
            a.ToTensorV2()
        ],
        bbox_params = a.BboxParams(format = "yolo", label_fields = ['class_labels'])
    )
    # 2. 实例化数据集
    # 使用 try-except 来处理路径不存在的情况
    try:
        train_dataset = YoloV1Dataset(r"E:\py project\torchProject\data\HelmetDataset-YOLO-Train\images",
                                      r"E:\py project\torchProject\data\HelmetDataset-YOLO-Train\labels",
                                      7, 20, transform)

        data_loder = DataLoader(train_dataset, 4, True)

        images, targets = next(iter(data_loder))
        print("测试成功！")
        print("一个批次的图片形状:", images.shape)
        print("一个批次的目标张量形状:", targets.shape)

        # 验证形状是否符合预期
        assert images.shape == (4, 3, 448, 448)
        assert targets.shape == (4, 7, 7, 25)  # S=7, C=20 -> 20+5=25
        print("\nDataset和DataLoader测试通过！输出形状正确，可以用于训练。")

    except FileNotFoundError:
        print("\n测试失败：请确保路径存在，")








