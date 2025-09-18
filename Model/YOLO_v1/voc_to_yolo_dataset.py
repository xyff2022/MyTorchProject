import os
import xml.etree.ElementTree

import albumentations as A
import cv2
import numpy
import pandas
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class VocDataset2YOLO(Dataset):
    def __init__(self, csv_file : str, img_dir : str, label_dir : str,  transform=None):
        self.img_dir = img_dir
        self.annotations = pandas.read_csv(csv_file, header=None, names=['filename'])
        self.label_dir = label_dir
        self.transform = transform
        self.B = 2
        self.S = 7
        self.C = 20
        self.class_list = {
            "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
            "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9,
            "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13,
            "person": 14, "pottedplant": 15, "sheep": 16, "sofa": 17,
            "train": 18, "tvmonitor": 19
        }

        # Bug修复：增加了测试代码需要的反向映射
        self.idx_to_class = {v: k for k, v in self.class_list.items()}
        self.classes = 20

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. --- 获取文件名和路径 ---
        # 根据传入的索引，从图片名称列表中获取对应的文件名
        filename = str(self.annotations.loc[index, 'filename'])
        # 拼接出完整的图片文件路径
        image_path = os.path.join(self.img_dir, filename + ".jpg")
        label_path = os.path.join(self.label_dir, filename + ".xml")

        # 2. --- 读取和预处理图片 ---
        # 使用 OpenCV 读取图片，返回值是 BGR 格式的 NumPy 数组
        image = cv2.imread(image_path)

        original_image = image.copy()

        # 将图片的颜色空间从 BGR (OpenCV 默认) 转换为 RGB (更通用)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---  解析XML并转换坐标 ---
        # 解析XML文件
        tree = xml.etree.ElementTree.parse(label_path)
        # 它会自动打开这个路径对应的XML文件，读取其中的全部内容，并将其解析成一个树状的结构（ElementTree对象）。
        root = tree.getroot()


        # 初始化用于存放边界框坐标和类别标签的列表
        bboxes = []
        labels = []

        # 使用 try-except 结构来捕获可能发生的解析错误，增强代码的健壮性
        try:
            # 遍历列表中的每一个物体标注
            for obj in root.findall("object"):
                # 获取物体的类别名称，例如 "person" 或 "car"
                class_name = obj.find("name").text
                # .text返回字符串

                # 如果这个类别名称不在我们预定义的类别列表中，则跳过这个物体
                if class_name not in self.class_list:
                    continue

                # 在预定义的类别列表中找到该名称对应的索引 (ID)
                class_id = self.class_list[class_name]

                # 获取边界框 'bndbox' 的字典
                bndbox = obj.find("bndbox")
                # 从字典中获取边界框的四个坐标值
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # 将解析出的坐标和类别 ID 添加到相应的列表中
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

        except Exception as e:
            # 如果在解析过程中发生任何错误，打印错误信息
            print(f"\n[错误] 解析文件失败: {label_path}, 原因: {e}")
            # 并确保返回空的标注，防止程序崩溃
            bboxes = []
            labels = []
        original_bboxes = bboxes.copy()

        # 4. --- 应用数据增强 ---
        # 检查是否在创建 Dataset 实例时传入了 transform (数据增强) 流水线
        if self.transform:
            # 调用 albumentations 的 transform 流水线，同时传入图片、边界框和标签
            # albumentations 会自动同步对图片和边界框的几何变换
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            # 从返回的字典中获取增强后的图片
            image = transformed['image']
            # 获取同步更新后的边界框
            bboxes = transformed['bboxes']
            # 获取对应的标签 (通常不会改变，但这是标准做法)
            labels = transformed['labels']

        # --- 编码为目标张量 ---
        target = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        # 获取数据增强后的图像大小
        _, image_height, image_width = image.shape


        # 我们要把bboxes里的[xmin, ymin, xmax, ymax]转换成[x_center,y_center,width,height]
        for box_id in range(len(bboxes)):
            box = bboxes[box_id]
            class_id = int(labels[box_id])


            xmin, ymin, xmax, ymax = box

            # 归一化和坐标转换
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height


            # 计算物体中心点所在的网格单元的行(i)和列(j)
            i = int(self.S * y_center)
            j = int(self.S * x_center)

            # 计算中心点在单元格内部的相对坐标 (center_x, center_y)
            # 单元格偏移量”,通过分而治之,极大地降低了学习难度,稳定了训练过程,并充分利用了卷积网络的空间特性
            center_x = self.S * x_center - j
            center_y = self.S * y_center - i

            if target[i, j, self.C ] == 0:
                target[i, j, self.C ] = 1
                target[i, j, self.C +1: self.C + 5] = torch.tensor(
                    [center_x, center_y, width, height]
                )
                target[i, j, class_id] = 1

        # target = {
        #     # 将边界框列表转换为 float32 类型的 PyTorch 张量
        #     "bboxes": torch.tensor(bboxes, dtype=torch.float32),
        #     # 将类别 ID 列表转换为 long 类型的 PyTorch 张量
        #     "labels": torch.tensor(labels, dtype=torch.long)
        # }

        # 返回增强后的图片和格式化后的标注字典
        return image, target,original_image, original_bboxes


# --- 集成的测试代码 ---
if __name__ == "__main__":
    # 1. 设置参数 (这是你需要根据你的实际情况修改的部分)
    # !!! ==> 修改这里: 指向你的 PASCAL VOC 数据集的 train.txt 文件
    TRAIN_CSV_FILE = r"../../data/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
    # !!! ==> 修改这里: 指向你的图片文件夹
    IMG_DIR = r"../../data/VOCdevkit/VOC2012/JPEGImages"
    # !!! ==> 修改这里: 指向你的 XML 标签文件夹
    LABEL_DIR = r"../../data/VOCdevkit/VOC2012/Annotations"

    # 其他参数
    S = 7
    B = 2
    C = 20
    IMAGE_SIZE = 448

    # 2. 定义数据增强
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']),
    )

    # 3. 主执行逻辑
    # Bug修复: 实例化正确的类名 VOCDataset
    test_dataset = VocDataset2YOLO(
        csv_file=TRAIN_CSV_FILE,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,  # 可以保持随机，因为__getitem__会返回所有需要的信息
    )

    # 获取一个样本，现在包含4个返回值
    image_tensor, label_matrix, original_image, original_bboxes = next(iter(test_loader))

    # --- 4. 新增的验证与可视化部分 ---
    print("\n--- 开始解码和验证 Target Tensor ---")

    label_matrix = label_matrix.squeeze(0)

    # 测试代码现在可以正确找到置信度标志了
    object_cells = torch.where(label_matrix[..., C ] == 1)
    num_objects = len(object_cells[0])

    if num_objects == 0:
        print("这张图片中没有找到任何在标签中的物体 (可能是一个bug，或该图片确实无标签)。")
    else:
        print(f"在这张图片中找到了 {num_objects} 个物体:")

        for i in range(num_objects):
            grid_row = object_cells[0][i]
            grid_col = object_cells[1][i]

            class_id = torch.argmax(label_matrix[grid_row, grid_col, 0:C]).item()
            class_name = test_dataset.idx_to_class[class_id]

            coords = label_matrix[grid_row, grid_col, C+1:C + 5]
            x_cell, y_cell, width_norm, height_norm = coords

            print(f"\n物体 #{i + 1}:")
            print(f"  - 类别: {class_name} (ID: {class_id})")
            print(f"  - 所在网格 (row, col): ({grid_row}, {grid_col})")
            print(
                f"  - 编码坐标 [x_cell, y_cell, w, h]: [{x_cell:.3f}, {y_cell:.3f}, {width_norm:.3f}, {height_norm:.3f}]")

    # --- 5. 可视化验证 ---
    # 将 PyTorch 张量转换回 OpenCV 图像格式
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
    image = image.astype(numpy.uint8)
    image_vis = numpy.ascontiguousarray(image, dtype=numpy.uint8)

    # 绘制解码后的框 (绿色)
    for i in range(num_objects):
        grid_row = object_cells[0][i]
        grid_col = object_cells[1][i]

        coords = label_matrix[grid_row, grid_col, C+1:C + 5]
        x_cell, y_cell, width_norm, height_norm = coords
        class_id = torch.argmax(label_matrix[grid_row, grid_col, 0:C]).item()
        class_name = test_dataset.idx_to_class[class_id]

        x_center_abs = (grid_col.item() + x_cell.item()) / S * IMAGE_SIZE
        y_center_abs = (grid_row.item() + y_cell.item()) / S * IMAGE_SIZE
        width_abs = width_norm.item() * IMAGE_SIZE
        height_abs = height_norm.item() * IMAGE_SIZE

        xmin = int(x_center_abs - width_abs / 2)
        ymin = int(y_center_abs - height_abs / 2)
        xmax = int(x_center_abs + width_abs / 2)
        ymax = int(y_center_abs + height_abs / 2)


        cv2.rectangle(image_vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
        cv2.putText(image_vis, f"Decoded: {class_name}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2)

    # --- 6. 新增: 绘制原始标签框 (红色) ---
    # 注意: 原始框是对应于原始图像尺寸的，我们需要将它们缩放到与增强后图像相同的尺寸
    original_image = original_image.squeeze(0).numpy()
    orig_h, orig_w, _ = original_image.shape

    # 遍历原始bboxes
    for box in original_bboxes:
        xmin_orig, ymin_orig, xmax_orig, ymax_orig = box

        # 将原始坐标缩放到 IMAGE_SIZE
        xmin = int(xmin_orig.item() / orig_w * IMAGE_SIZE)
        ymin = int(ymin_orig.item() / orig_h * IMAGE_SIZE)
        xmax = int(xmax_orig.item() / orig_w * IMAGE_SIZE)
        ymax = int(ymax_orig.item() / orig_h * IMAGE_SIZE)

        cv2.rectangle(image_vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    print("\n--- 可视化结果 ---")
    print("绿色框 = 从 Target Tensor 解码出的结果")
    print("红色框 = 从 XML 文件读取的原始标签 (Ground Truth)")
    print("如果两种颜色的框完美重合，说明数据处理和编码完全正确！")

    # --- 修改: 恢复使用 cv2.imshow() 进行实时可视化 ---
    # cv2.imshow 需要 BGR 格式
    cv2.imshow("Validation Image", cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
    print("\n按任意键关闭图片窗口。")
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗