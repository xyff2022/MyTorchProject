import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import sys
import torchvision

# --- 动态添加项目根目录到 Python 路径 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 导入你项目中的模块 ---
from Model.FasterRCNN.FasterRCNN import FasterRCNN
from dataset.VOCDatasetPlus import VocDataset
from Model.FasterRCNN.RegionProposalNetwork import RegionProposalNetwork
from Model.FasterRCNN.Pooling import RoIPool
from Model.FasterRCNN.DetectorHead import DetectorHead

# --- 1. 配置区域 ---

# --- 训练设置 ---
LEARNING_RATE = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
GRADIENT_CLIP_VAL = 1.0

# --- 数据集路径 ---
TRAIN_DIR = r"E:\py project\torchProject\data\PASCAL_VOC\train"
VAL_DIR = r"E:\py project\torchProject\data\PASCAL_VOC\val"

# --- 模型与检查点保存路径 ---
BEST_MODEL_SAVE_PATH = "best_model.pth"
CHECKPOINT_PATH = "latest_checkpoint.pth"  # 用于断点续训

# --- PASCAL VOC 数据集的20个类别 ---
PASCAL_VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                      "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                      "tvmonitor"]
NUM_CLASSES = len(PASCAL_VOC_CLASSES)


# --- 2. 辅助函数 (保持不变) ---
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(600, 600),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(600, 600),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes, pretrained=True):
    weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
    resnet_model = torchvision.models.resnet50(weights=weights)
    backbone = torch.nn.Sequential(*list(resnet_model.children())[:-2])
    backbone.out_channels = 2048
    rpn = RegionProposalNetwork(in_channels=backbone.out_channels, mid_channels=512, num_anchors=9)
    pooler = RoIPool(output_size=7, spatial_scale=1.0 / 16)
    detector_head = DetectorHead(num_classes=num_classes, pooler=pooler, in_channels=backbone.out_channels)
    model = FasterRCNN(backbone, rpn, detector_head)
    return model


# --- 3. 训练与验证函数 (保持不变) ---
def train_one_epoch(model, optimizer, loader, device):
    model.train()
    loop = tqdm(loader, leave=True, desc="Training")
    total_loss = 0
    for images, targets in loop:
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
            images = torch.stack(images).to(device)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            detections = model(images)
            metric.update(detections, targets_device)
    results = metric.compute()
    print("\n--- 验证结果 ---")
    map_50 = results.get('map_50', torch.tensor(-1)).item()
    print(f"mAP@50: {map_50:.4f}")
    return map_50


# --- 4. 主执行函数 (已更新) ---
def main():
    print(f"--- 正在使用设备: {DEVICE} ---")

    model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # --- *** 新增: 加载检查点逻辑 *** ---
    start_epoch = 0
    best_map = -1.0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- 发现检查点文件，正在恢复训练 ---")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        best_map = checkpoint['best_map']
        print(f"已恢复到 epoch {start_epoch}, 当前最佳 mAP@50: {best_map:.4f}")
    else:
        print(f"--- 未发现检查点，将从头开始训练 ---")

    # ... (数据加载器部分不变) ...
    train_dataset = VocDataset(os.path.join(TRAIN_DIR, "images"), os.path.join(TRAIN_DIR, "labels"), PASCAL_VOC_CLASSES,
                               get_transforms(train=True))
    val_dataset = VocDataset(os.path.join(VAL_DIR, "images"), os.path.join(VAL_DIR, "labels"), PASCAL_VOC_CLASSES,
                             get_transforms(train=False))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                              num_workers=4 if DEVICE == 'cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                            num_workers=4 if DEVICE == 'cuda' else 0)

    # --- *** 更新: 训练循环使用 start_epoch *** ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch + 1}/{NUM_EPOCHS}] ---")
        avg_train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
        print(f"Epoch {epoch + 1} 训练平均损失: {avg_train_loss:.4f}")

        current_map = validate(model, val_loader, DEVICE)

        # 在验证之后，更新学习率
        lr_scheduler.step()

        # 保存最佳模型
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"** 新的最佳模型已保存！mAP@50: {best_map:.4f} **")

        # --- *** 新增: 在每个 epoch 结束时保存检查点 *** ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_map': best_map,
        }, CHECKPOINT_PATH)
        print(f"已保存当前 epoch 的检查点至: {CHECKPOINT_PATH}")

    print(f"\n--- 训练完成 ---")
    print(f"最佳验证 mAP@50: {best_map:.4f}")
    print(f"最佳模型权重已保存至: {BEST_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

