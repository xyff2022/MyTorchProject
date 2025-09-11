import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import sys

# --- 动态添加项目根目录到 Python 路径 ---
# 这确保了无论你在哪个目录下运行脚本，都能正确导入项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 导入你项目中的模块 ---
from Model.FasterRCNN.FasterRCNN import FasterRCNN
# 注意：请确保你的数据加载器文件名是 VOCDataset.py，如果不是请修改下面的导入
from dataset.VOCDatasetPlus import VocDataset
from Model.FasterRCNN.Backbone import get_my_resnet50_backbone
from Model.FasterRCNN.RegionProposalNetwork import RegionProposalNetwork
from Model.FasterRCNN.Pooling import RoIPool
from Model.FasterRCNN.DetectorHead import DetectorHead

# --- 1. 配置区域 ---

# --- 训练设置 ---
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # 如果显存不足，请降低此值
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
GRADIENT_CLIP_VAL = 1.0

# --- 数据集路径 (根据你的实际路径修改) ---
TRAIN_DIR = r"E:\py project\torchProject\data\PASCAL_VOC\train"
VAL_DIR = r"E:\py project\torchProject\data\PASCAL_VOC\val"

# --- 模型保存路径 ---
MODEL_SAVE_PATH = "best_model.pth"

# --- PASCAL VOC 数据集的20个类别 ---
PASCAL_VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(PASCAL_VOC_CLASSES)


# --- 2. 数据增强流水线 ---
def get_transforms(train=True):
    """定义训练和验证时使用的数据增强。"""
    if train:
        # 训练时使用丰富的数据增强
        return A.Compose([
            A.Resize(600, 600),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # 验证时只进行必要的尺寸调整和归一化
        return A.Compose([
            A.Resize(600, 600),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# --- 3. 辅助函数 ---
def collate_fn(batch):
    """自定义数据整理函数，以处理不同数量标注的图片。"""
    return tuple(zip(*batch))


def build_model(num_classes):
    """组装 Faster R-CNN 模型。"""
    backbone = get_my_resnet50_backbone()
    rpn = RegionProposalNetwork(in_channels=1024, mid_channels=512, num_anchors=9)
    pooler = RoIPool(output_size=7, spatial_scale=1.0 / 16)
    detector_head = DetectorHead(num_classes=num_classes, pooler=pooler, in_channels=1024)
    model = FasterRCNN(backbone, rpn, detector_head)
    return model


# --- 4. 训练与验证函数 ---
def train_one_epoch(model, optimizer, loader, device):
    model.train()
    loop = tqdm(loader, leave=True, desc="Training")
    total_loss = 0

    for images, targets in loop:
        # *** 关键修复: 将图片列表堆叠成一个批次张量 ***
        # images 是一个元组，包含批次内所有的图片张量
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses):
            print("\n[警告] 检测到 NaN 损失，跳过此批次。")
            continue

        optimizer.zero_grad()
        losses.backward()
        clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
        optimizer.step()

        total_loss += losses.item()
        loop.set_postfix(loss=losses.item())

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
    loop = tqdm(loader, leave=True, desc="Validating")

    with torch.no_grad():
        for images, targets in loop:
            # *** 关键修复: 同样需要堆叠图片 ***
            images = torch.stack(images).to(device)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

            detections = model(images)
            metric.update(detections, targets_device)

    results = metric.compute()
    print("\n--- 验证结果 ---")
    map_50 = results.get('map_50', torch.tensor(-1)).item()
    print(f"mAP@50: {map_50:.4f}")
    return map_50


# --- 5. 主执行函数 ---
def main():
    print(f"--- 正在使用设备: {DEVICE} ---")

    # 构建模型
    model = build_model(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # 准备数据集
    train_dataset = VocDataset(
        image_folder=os.path.join(TRAIN_DIR, "images"),
        label_folder=os.path.join(TRAIN_DIR, "labels"),
        classes_list=PASCAL_VOC_CLASSES,
        transform=get_transforms(train=True)
    )
    val_dataset = VocDataset(
        image_folder=os.path.join(VAL_DIR, "images"),
        label_folder=os.path.join(VAL_DIR, "labels"),
        classes_list=PASCAL_VOC_CLASSES,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
        num_workers=4 if DEVICE == 'cuda' else 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=4 if DEVICE == 'cuda' else 0
    )

    best_map = -1.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch + 1}/{NUM_EPOCHS}] ---")
        avg_train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
        print(f"Epoch {epoch + 1} 训练平均损失: {avg_train_loss:.4f}")

        current_map = validate(model, val_loader, DEVICE)

        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"** 新的最佳模型已保存！mAP@50: {best_map:.4f} **")

    print(f"\n--- 训练完成 ---")
    print(f"最佳验证 mAP@50: {best_map:.4f}")
    print(f"最佳模型权重已保存至: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

