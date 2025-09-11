import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import sys

# --- 动态添加项目根目录到 Python 路径，以确保可以正确导入自定义模块 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 导入你项目中的模块 ---
from Model.FasterRCNN.FasterRCNN import FasterRCNN
from dataset.VOCDatasetPlus import VocDataset
from Model.FasterRCNN.Backbone import get_my_resnet50_backbone
from Model.FasterRCNN.RegionProposalNetwork import RegionProposalNetwork
from Model.FasterRCNN.Pooling import RoIPool
from Model.FasterRCNN.DetectorHead import DetectorHead

# --- 1. 配置区域 ---

# --- 评估设置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# --- 数据集路径 ---
EVAL_DIR = r"E:\py project\torchProject\data\PASCAL_VOC\val"

# --- 模型权重路径 ---
MODEL_WEIGHTS_PATH = "best_model.pth"

# --- PASCAL VOC 数据集的20个类别 ---
PASCAL_VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(PASCAL_VOC_CLASSES)


# --- 2. 辅助函数 ---
def get_eval_transforms():
    """定义评估时使用的数据变换。"""
    return A.Compose([
        A.Resize(600, 600),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    """自定义数据整理函数。"""
    return tuple(zip(*batch))


def build_model(num_classes, pretrained=False):  # 在评估时，我们不关心预训练，因为权重会被加载覆盖
    """
    组装 Faster R-CNN 模型，匹配训练时的架构。
    """
    # 即使 pretrained=False，架构也是一致的
    weights = None
    resnet_model = torchvision.models.resnet50(weights=weights)

    backbone = torch.nn.Sequential(*list(resnet_model.children())[:-2])
    backbone.out_channels = 2048

    rpn = RegionProposalNetwork(in_channels=backbone.out_channels, mid_channels=512, num_anchors=9)
    pooler = RoIPool(output_size=7, spatial_scale=1.0 / 16)
    detector_head = DetectorHead(num_classes=num_classes, pooler=pooler, in_channels=backbone.out_channels)

    model = FasterRCNN(backbone, rpn, detector_head)
    return model


def pretty_print_map(results, class_names):
    """
    以美观、结构化的方式打印 mAP 结果。
    """
    print("\n--- 最终评估结果 ---")

    # 总体性能
    print("\n[总体性能指标]")
    print(f"  mAP @ IoU[.50:.95]:    {results.get('map', torch.tensor(-1)).item():.4f}")
    print(f"  mAP @ IoU=0.50:        {results.get('map_50', torch.tensor(-1)).item():.4f}")
    print(f"  mAP @ IoU=0.75:        {results.get('map_75', torch.tensor(-1)).item():.4f}")

    # 按物体尺寸划分的性能
    print("\n[按物体尺寸划分的性能]")
    print(f"  mAP (小型物体):      {results.get('map_small', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (中型物体):      {results.get('map_medium', torch.tensor(-1)).item():.4f}")
    print(f"  mAP (大型物体):      {results.get('map_large', torch.tensor(-1)).item():.4f}")

    # 平均召回率
    print("\n[平均召回率 (MAR)]")
    print(f"  MAR @ 1 detection:     {results.get('mar_1', torch.tensor(-1)).item():.4f}")
    print(f"  MAR @ 10 detections:    {results.get('mar_10', torch.tensor(-1)).item():.4f}")
    print(f"  MAR @ 100 detections:   {results.get('mar_100', torch.tensor(-1)).item():.4f}")

    # 每个类别的 mAP
    map_per_class = results.get('map_per_class', torch.tensor([-1]))
    if map_per_class.numel() > 0 and map_per_class.item() != -1:
        print("\n[每个类别的 mAP @ IoU=0.50]")
        # 确保返回的类别数量与我们的列表匹配
        if map_per_class.numel() == len(class_names):
            for i, class_name in enumerate(class_names):
                print(f"  - {class_name:<15s}: {map_per_class[i].item():.4f}")
        else:
            print("  (返回的类别数量与预定义列表不匹配)")

    print("\n--------------------")
    print("说明: -1.0000 表示该指标无法计算 (例如验证集中没有该尺寸的物体)")


# --- 3. 主执行函数 ---
def main():
    print(f"--- 开始在设备 {DEVICE} 上进行评估 ---")

    # 1. 使用新的 build_model 函数构建模型
    model = build_model(num_classes=NUM_CLASSES).to(DEVICE)

    # 2. 加载训练好的权重
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        print(f"成功加载模型权重: {MODEL_WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"[错误] 找不到模型权重文件: {MODEL_WEIGHTS_PATH}")
        return

    model.eval()

    # 3. 准备评估数据集
    eval_dataset = VocDataset(
        image_folder=os.path.join(EVAL_DIR, "images"),
        label_folder=os.path.join(EVAL_DIR, "labels"),
        classes_list=PASCAL_VOC_CLASSES,
        transform=get_eval_transforms()
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=4 if DEVICE == 'cuda' else 0
    )

    # 4. 初始化 mAP 计算器
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(DEVICE)
    loop = tqdm(eval_loader, desc="正在评估")

    with torch.no_grad():
        for images, targets in loop:
            images_tensor = torch.stack(images).to(DEVICE)
            targets_device = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            detections = model(images_tensor)
            metric.update(detections, targets_device)

    # 5. 计算并打印最终结果
    try:
        results = metric.compute()
        pretty_print_map(results, PASCAL_VOC_CLASSES)
    except Exception as e:
        print(f"\n无法计算 mAP。原因: {e}")


if __name__ == "__main__":
    main()

