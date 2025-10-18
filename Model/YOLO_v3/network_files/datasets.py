import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================== 主类: YOLOv3Dataset ===============================

class YOLOv3Dataset(Dataset):
    def __init__(self,
                 img_root,  # 指向 images 文件夹的路径
                 label_root,  # 指向 labels 文件夹的路径
                 img_size=512,
                 transform=False):
        """
        使用 Albumentations 的 YOLOv3 数据集加载器。

        Args:
            img_root (str): 包含所有图像的文件夹路径。
            label_root (str): 包含所有标签文件的文件夹路径。
            img_size (int): 训练和验证时图像的目标尺寸。
            transform (bool): 是否应用数据增强。
        """
        # ---------------------- 1. 初始化和路径处理 ----------------------
        self.img_root = img_root
        self.label_root = label_root
        self.img_size = img_size

        # 自动扫描图片文件夹, 获取所有图片文件名
        self.image_names = sorted([name for name in os.listdir(img_root)
                                   if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        # ---------------------- 2. 定义图像增强管道 ----------------------
        if transform:
            # 训练时的增强策略
            self.transform = A.Compose([

                # --- 几何变换 ---
                # 更强大的仿射变换组合，模拟物体在图像中的各种姿态
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.7,
                                   border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5),

                # --- 颜色和质量变换 ---
                # 增强的颜色抖动
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02, p=0.7),
                # 随机调整亮度和对比度
                A.RandomBrightnessContrast(p=0.5),
                # 随机应用一种图像质量下降的效果，提高模型对模糊、噪点的鲁棒性
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(p=0.5),
                    A.ISONoise(p=0.5),
                ], p=0.3),

                # --- 结构性变换与正则化 ---
                # 随机裁剪图像的一部分，同时保证 BBox 安全
                A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, erosion_rate=0.2, p=0.5),
                # Cutout/CoarseDropout: 随机遮挡图像块，强迫模型学习上下文信息
                A.CoarseDropout(max_holes=8, max_height=int(img_size * 0.1), max_width=int(img_size * 0.1),
                                min_holes=2, fill_value=0, mask_fill_value=None, p=0.5),
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # 验证时的处理策略
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        """返回数据集中图片的数量。"""
        return len(self.image_names)

    def __getitem__(self, index):
        """根据索引获取一个处理好的样本 (图片, 扁平化的标签列表)。"""
        # ---------------------- 1. 读取原始数据 ----------------------
        image_name = self.image_names[index]
        image_path = os.path.join(self.img_root, image_name)

        image = cv2.imread(image_path)
        # 新增: 获取原始图像尺寸
        h0, w0 = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.label_root, label_name)

        bboxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    values = [float(v) for v in line.strip().split()]
                    bboxes.append(values)

            # ---------------------- 2. 应用增强 ----------------------
        class_labels = [box[0] for box in bboxes]
        yolo_bboxes = [box[1:] for box in bboxes]

        transformed = self.transform(image=image, bboxes=yolo_bboxes, class_labels=class_labels)

        transformed_image = transformed['image']

        # --- [核心修改] 手动进行 [0, 1] 归一化 ---
        # A.Normalize 已被移除，我们在这里进行正确的缩放
        transformed_image = transformed_image.float() / 255.0

        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        # ---------------------- 3. 格式化输出 (YOLOv3格式) ----------------------
        number_labels = len(transformed_class_labels)
        labels_out = torch.zeros((number_labels, 6))

        if number_labels > 0:
            # 重新组合 [类别, x, y, w, h]
            labels_np = np.hstack((np.array(transformed_class_labels).reshape(-1, 1),
                                   np.array(transformed_bboxes)))
            labels_out[..., 1:] = torch.from_numpy(labels_np)

        return transformed_image, labels_out, image_path, ((h0, w0),)

    @staticmethod
    def collate_fn(batch):
        """
        自定义的批次打包函数, 用于处理YOLOv3的标签。
        """
        images, labels, path, shapes = zip(*batch)  # 解包一个批次的数据

        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                label[:, 0] = i

        # 过滤掉没有标签的样本, 并将所有标签拼接成一个大张量
        valid_labels = [label for label in labels if label.shape[0] > 0]

        if len(valid_labels) > 0:
            return torch.stack(images, 0), torch.cat(valid_labels, 0), path, shapes
        else:
            return torch.stack(images, 0), torch.zeros((0, 6)), path, shapes



