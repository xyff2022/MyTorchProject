import os
import torch
import xmltodict
import cv2  # 使用 OpenCV 读取图片以配合 albumentations
from torch.utils.data import Dataset


class VocDataset2YOLO(Dataset):
    def __init__(self, image_folder, label_folder, classes_list, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_names = os.listdir(self.image_folder)
        self.classes_list = classes_list

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # 1. --- 获取文件名和路径 ---
        # 根据传入的索引，从图片名称列表中获取对应的文件名
        image_name = self.image_names[index]
        # 拼接出完整的图片文件路径
        image_path = os.path.join(self.image_folder, image_name)

        # 2. --- 读取和预处理图片 ---
        # 使用 OpenCV 读取图片，返回值是 BGR 格式的 NumPy 数组
        image = cv2.imread(image_path)
        # 将图片的颜色空间从 BGR (OpenCV 默认) 转换为 RGB (更通用)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. --- 读取和解析标注文件 ---
        # 根据图片文件名 (如 '0001.jpg') 生成对应的 XML 标注文件名 (如 '0001.xml')
        label_name = image_name.split(".")[0] + ".xml"
        # 拼接出完整的标注文件路径
        label_path = os.path.join(self.label_folder, label_name)

        # 初始化用于存放边界框坐标和类别标签的列表
        boxes = []
        labels = []

        # 使用 try-except 结构来捕获可能发生的解析错误，增强代码的健壮性
        try:
            # 打开 XML 文件并读取其全部内容
            with open(label_path, "r", encoding="utf-8") as f:
                label_content = f.read()
            # 使用 xmltodict 库将 XML 字符串内容解析成 Python 字典
            label_dict = xmltodict.parse(label_content)
            # 安全地获取 'annotation' 标签，如果不存在则返回一个空字典
            annotation = label_dict.get('annotation', {})
            # 安全地获取 'object' 标签，如果不存在则返回一个空列表
            objects_data = annotation.get('object', [])

            # 兼容性处理：如果 XML 中只有一个 <object>，xmltodict 会将其解析为字典，而不是列表。
            # 这行代码确保 objects_data 永远是一个列表，方便后续统一处理。
            if not isinstance(objects_data, list):
                objects_data = [objects_data]

            # 遍历列表中的每一个物体标注
            for obj in objects_data:
                # 获取物体的类别名称，例如 "person" 或 "car"
                class_name = obj.get('name')
                # 如果这个类别名称不在我们预定义的类别列表中，则跳过这个物体
                if class_name not in self.classes_list:
                    continue

                # 在预定义的类别列表中找到该名称对应的索引 (ID)
                class_id = self.classes_list.index(class_name)

                # 获取边界框 'bndbox' 的字典
                bndbox = obj.get('bndbox', {})
                # 从字典中获取边界框的四个坐标值
                xmin = float(bndbox.get('xmin', 0))
                ymin = float(bndbox.get('ymin', 0))
                xmax = float(bndbox.get('xmax', 0))
                ymax = float(bndbox.get('ymax', 0))

                # 将解析出的坐标和类别 ID 添加到相应的列表中
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

        except Exception as e:
            # 如果在解析过程中发生任何错误，打印错误信息
            print(f"\n[错误] 解析文件失败: {label_path}, 原因: {e}")
            # 并确保返回空的标注，防止程序崩溃
            boxes = []
            labels = []

        # 4. --- 应用数据增强 ---
        # 检查是否在创建 Dataset 实例时传入了 transform (数据增强) 流水线
        if self.transform:
            # 调用 albumentations 的 transform 流水线，同时传入图片、边界框和标签
            # albumentations 会自动同步对图片和边界框的几何变换
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            # 从返回的字典中获取增强后的图片
            image = transformed['image']
            # 获取同步更新后的边界框
            boxes = transformed['bboxes']
            # 获取对应的标签 (通常不会改变，但这是标准做法)
            labels = transformed['labels']

        # 5. --- 格式化最终输出 ---
        # 将处理好的边界框和标签整理成一个字典，这是模型所期望的格式
        target = {
            # 将边界框列表转换为 float32 类型的 PyTorch 张量
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            # 将类别 ID 列表转换为 long 类型的 PyTorch 张量
            "labels": torch.tensor(labels, dtype=torch.long)
        }

        # 返回增强后的图片和格式化后的标注字典
        return image, target
