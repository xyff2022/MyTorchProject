# 文件路径: torchProject/Model/YOLO_v1/train/Yolov1Train.py

# ==================== PURE PYTHON PATH SETUP ====================
import sys
from pathlib import Path

# 计算出项目根目录 (torchProject) 的绝对路径
# Yolov1Train.py -> train/ -> YOLO_v1/ -> Model/ -> torchProject/
# 所以我们需要向上跳 4 层才能到根目录，因此使用 parents[3]
FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT)) # 或者 sys.path.append(str(ROOT))


import cv2
import torch
from albumentations import ToTensorV2
from torch import optim
import albumentations as A
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from Model.YOLO_v1.utils.non_max_suppression import non_max_suppression
from Model.YOLO_v1.utils.predictions_to_boxes import predictions_to_boxes
from Model.YOLO_v1.utils.target_to_boxes import  target_to_boxes
from Model.YOLO_v1.voc_to_yolo_dataset import VocDataset2YOLO
# 【新增】: 导入 torchmetrics 用于计算 COCO mAP

from Model.YOLO_v1.network_files.yolov1 import YOLOv1, YoloLoss


# --- 1. 初始化 ---
# --- 设置超参数和全局配置 ---
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 120
NUM_EPOCHS = 100 # 训练的总轮数
WEIGHT_DECAY = 5e-4
# 【新增】: 定义预热阶段的周期数
WARMUP_EPOCHS = 3
# 【新增】: 定义训练结束时，学习率衰减到的最终比例
FINAL_LR_SCALE = 0.01

IOU_THRESHOLD = 0.5   # iou_threshold(float): 用于抑制重叠框的IoU阈值。
THRESHOLD = 0.25      # threshold(float): 置信度阈值，过滤掉低置信度的框。


# YOLOv1 模型和数据的特定参数
S = 7
B = 2
IMAGE_SIZE = 448

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

LOAD_MODEL = False # 【新增】: 是否从检查点加载模型继续训练

# 定义检查点文件路径
CHECKPOINT_FILE = "yolov1_checkpoint.pth.tar"
BEST_CHECKPOINT_FILE = "yolov1_best.pth.tar" # 最好模型的保存路径

# 数据集路径
TRAIN_IMG_DIR = r"../../../data/VOCdevkit/VOC2012/JPEGImages"
TRAIN_LABEL_DIR = r"../../../data/VOCdevkit/VOC2012/Annotations"
VAL_IMG_DIR = r"../../../data/VOCdevkit/VOC2012/JPEGImages"
VAL_LABEL_DIR = r"../../../data/VOCdevkit/VOC2012/Annotations"

PASCAL_VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                      "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                      "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

C = len(PASCAL_VOC_CLASSES)

TRAIN_CSV_FILE = r"../../../data/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
VAL_CSV_FILE =r"../../../data/VOCdevkit/VOC2012/ImageSets/Main/val.txt"
# --- 2. 定义数据增强和加载 ---

# 定义用于训练集的数据增强流水线
# (这里可以加入更多您想尝试的增强，如ShiftScaleRotate, ColorJitter等)
def get_transforms(train=True):
    """
    根据是训练阶段还是验证阶段，返回相应的数据增强/转换流程。

    参数:
        train (bool): 如果为True，则返回训练用的数据增强流程；否则返回验证用的。
    """
    if train:
        # 定义训练时使用的数据增强流程
        transform = A.Compose([
            # 将图片尺寸调整为 448x448
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            # 以50 % 的概率随机调整图片的亮度、对比度、饱和度和色相
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.4, p=0.5),
            # 以50 % 的概率随机进行平移、缩放和旋转
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            # 以50 % 的概率水平翻转图片
            A.HorizontalFlip(p=0.5),
            # 以10 % 的概率对图片进行模糊处理
            A.Blur(blur_limit=3, p=0.1),
            # 使用ImageNet的均值和标准差对图片进行归一化
            A.Normalize(mean=imagenet_mean, std=imagenet_std, max_pixel_value=255, ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # 定义验证/测试时使用的数据增强流程（通常只包含必要的尺寸调整和标准化）
        transform = A.Compose([
            # 将图片尺寸调整为 448x448
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            # 标准化图片像素值
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            # 将图片和标签转换为PyTorch张量
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    return transform


# --- 【新增】步骤 5.1: 创建学习率调度器 ---
def linear_warmup_decay(warmup_epochs, total_epochs, final_lr_scale):
    """
    生成一个用于 LambdaLR 的函数，该函数根据当前周期计算学习率的乘数因子。
    """

    def lr_lambda(current_epoch):
        """
        根据当前周期数返回学习率的乘数。
        """
        # 预热阶段 (Warm-up Phase)
        if current_epoch < warmup_epochs:
            # 学习率从接近0线性增加到1倍 (即初始学习率)
            return float(current_epoch) / float(max(1, warmup_epochs))
        # 线性衰减阶段 (Linear Decay Phase)
        else:
            # 计算从1倍衰减到 final_lr_scale 倍，在 (total_epochs - warmup_epochs) 个周期内完成
            # remaining_epochs: 衰减阶段总共有多少个周期
            remaining_epochs = total_epochs - warmup_epochs
            # current_decay_epoch: 当前处于衰减阶段的第几个周期 (从0开始)
            current_decay_epoch = current_epoch - warmup_epochs
            # scale: 计算一个从1.0线性下降到0.0的比例因子
            scale = 1.0 - (current_decay_epoch / float(max(1, remaining_epochs)))
            # 最终的乘数是 final_lr_scale 和 1.0 之间的线性插值
            return final_lr_scale + (1.0 - final_lr_scale) * scale

    # 返回这个内部定义的函数
    return lr_lambda



# --- 3. 封装单次训练循环的函数 ---
def train_one_epoch(dataloader, model, optimizer, loss_fn, scaler):
    """
       执行一个完整的训练周期 (epoch)。

       Args:
           dataloader (DataLoader): 训练数据加载器。
           model (Module): 要训练的模型。
           optimizer (Optimizer): 优化器。
           loss_fn (Module): 损失函数。
           scaler(GradScaler): 用于混合精度训练的缩放器
    """

    # 将模型设置为训练模式，这会启用Dropout等层
    model.train()
    # 使用tqdm来创建一个可视化的进度条
    loop = tqdm(dataloader, leave=True, desc="Training")
    mean_loss = []


    for batch_idx, (images, targets) in enumerate(loop):
        # 将数据移动到指定的设备 (GPU或CPU)
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # 使用混合精度进行前向传播
        with autocast():

            # 1. 前向传播: 将图片送入模型得到预测结果
            prediction = model(images)

            # 2. 计算损失: 比较预测结果和真实标签
            loss = loss_fn(prediction, targets)
        mean_loss.append(loss.item())

        # 3. 反向传播: 清空梯度 -> 计算梯度 -> 更新权重
        optimizer.zero_grad()
        # 反向传播 (由scaler处理)
        scaler.scale(loss).backward()
        # 更新权重 (由scaler处理)
        scaler.step(optimizer)
        # 更新scaler的缩放因子
        scaler.update()

        # 更新tqdm进度条的后缀信息，实时显示当前损失
        loop.set_postfix(loss=loss.item())
    # 打印当前epoch的平均损失
    print(f"当前Epoch的平均损失为: {sum(mean_loss) / len(mean_loss):.4f}")

# --- 5. 模型保存与加载 ---
def save_checkpoint(state, filename=CHECKPOINT_FILE):
    """
    保存模型检查点。
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    加载模型检查点。
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def get_map_torchmetrics(data_loader, model, device = DEVICE):
    """
    在验证集上使用 torchmetrics 计算并返回 COCO mAP 分数。
    """
    # 1. 初始化 metric
    metric = MeanAveragePrecision(box_format="xyxy").to(device)

    # 确保模型处于评估模式
    model.eval()
    for images, targets in tqdm(data_loader, desc="Evaluating COCO mAP"):
        with torch.no_grad():
            # 2. 获取模型预测
            images = images.to(device)
            predictions = model(images)
            predictions = predictions.to(device)

        predictions_for_metric = []
        targets_for_metric = []

        # 遍历批次中的每一张图片
        for i in range(BATCH_SIZE):
            # --- 处理预测 ---
            # 使用我们之前写的函数进行解码和NMS
            boxes = predictions_to_boxes(predictions[i], IMAGE_SIZE, THRESHOLD)
            final_boxes = non_max_suppression(boxes, IOU_THRESHOLD, THRESHOLD)

            # 将解码后的结果打包成torchmetrics需要的字典格式
            # --- Bug修复：显式处理 pred_boxes 为空的情况 ---
            if final_boxes:
                # 如果有预测框，正常转换
                predictions_for_metric.append({
                    "boxes": torch.tensor([box[2:] for box in final_boxes]),
                    "scores": torch.tensor([box[1] for box in final_boxes]),
                    "labels": torch.tensor([box[0] for box in final_boxes]),
                })
            else:
                # 如果没有预测框，创建具有正确形状的空张量
                predictions_for_metric.append({
                    "boxes": torch.empty(0, 4),
                    "scores": torch.empty(0),
                    "labels": torch.empty(0),
                })

            # --- 处理真值 ---
            target_boxes = target_to_boxes(targets[i], IMAGE_SIZE, )

            targets_for_metric.append({
                "boxes" : torch.tensor([box[2:] for box in target_boxes]),
                "labels": torch.tensor([box[0] for box in target_boxes]),
            })


        metric.update(predictions_for_metric, targets_for_metric)

    # 5. 计算最终结果
    print("\n--- COCO mAP Evaluation Results ---")
    results = metric.compute()

    # 切换回训练模式
    model.train()

    return results




# --- 4. 主执行函数 ---
def main():
    """
        主函数，负责组装所有部件并启动训练流程。
    """
    # 初始化YOLOv1模型，并将其移动到指定设备 (CPU或GPU)

    model = YOLOv1().to(DEVICE)

    # 初始化损失函数
    loss_fn = YoloLoss(s=S, b=B, c=C)

    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY)
    # 创建学习率调整函数
    lr_scheduler_func = linear_warmup_decay(WARMUP_EPOCHS, NUM_EPOCHS, FINAL_LR_SCALE)
    # 创建混合精度缩放器
    scaler = GradScaler()
    # 使用 LambdaLR 创建调度器实例
    scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler_func)

    # 准备数据加载器
    train_dataset = VocDataset2YOLO(TRAIN_CSV_FILE,
                                    TRAIN_IMG_DIR,
                                    TRAIN_LABEL_DIR,
                                    get_transforms(True))

    val_dataset = VocDataset2YOLO(VAL_CSV_FILE,
                                  VAL_IMG_DIR,
                                  VAL_LABEL_DIR,
                                  get_transforms(False))



    train_loader = DataLoader(train_dataset,
                             BATCH_SIZE,
                             shuffle=True,
                             drop_last=True,
                             num_workers=8,  # 根据你的CPU核心数调整
                             pin_memory=True,  # 加速数据从CPU到GPU的传输
                             )

    val_loader = DataLoader(val_dataset,
                           BATCH_SIZE,
                           shuffle=False,
                           drop_last=True,
                           num_workers=8,  # 根据你的CPU核心数调整
                           pin_memory=True,  # 加速数据从CPU到GPU的传输
                           )
    """
    val_loader 在打包一个批次的数据时，会执行其默认行为。
    当 val_dataset 的 __getitem__ 方法返回一个 (image, target_dict) 元组时，val_loader 会：
        将所有 image 张量堆叠（stack）成一个大的批处理张量。
        将所有 target_dict 字典收集到一个 Python 列表中。
    因此，从 val_loader 中迭代出的 targets 是一个长度为 N 的列表，
    形如 [{'boxes':..., 'labels':...}, {'boxes':..., 'labels':...}, ...]。
    """

    # 4. 执行主训练循环
    print("--- 开始训练 ---")
    best_map = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1} / {NUM_EPOCHS} ---")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)

        # 在每个周期结束后计算mAP
        current_map = get_map_torchmetrics(val_loader, model)
        print(f"Current validation mAP is: {current_map:.4f}")

        # --- 【核心逻辑】: 只有当mAP提升时才保存模型 ---
        if current_map > best_map:
            best_map = current_map
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=BEST_CHECKPOINT_FILE)
            print(f"New best mAP: {best_map:.4f}! Model saved to {BEST_CHECKPOINT_FILE}")



        # 在每个周期结束后更新学习率
        scheduler.step()


    print("--- 训练完成 ---")
    print(f"训练结束，最佳验证集mAP为: {best_map:.4f}")



# --- 5. 脚本入口 ---
# 确保只有在直接运行这个文件时，main()函数才会被执行
if __name__ == "__main__":
    main()

