import math
import os
import sys
from pathlib import Path

import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.std import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# =============================== 0. PATH CONFIGURATION / 路径配置 ===============================
# 这部分代码确保无论您从哪里运行脚本, 都能正确地导入项目中的其他模块
# =====================================================================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =============================== 导入路径 ===============================
# 导入我们自己编写的模块, 明确指定它们位于 network_files 文件夹下
from network_files.yolov3 import YOLOv3
from network_files.datasets import YOLOv3Dataset
from network_files.utils import compute_ciou_loss
from train_utils.evaluate_map import non_max_suppression,xywh2xyxy,decode_outputs,scale_resize_coords

# =============================== 1. CONFIGURATION / 参数配置 ===============================
# 在这里修改所有训练参数, 无需使用命令行
# =================================================================================


# --- 数据集路径 (健壮版) ---
# 关键改动: 从脚本位置动态构建数据集的绝对路径
# 1. 获取项目的根目录 (YOLO_v3/的上两级)
PROJECT_ROOT = FILE.parents[2]
# 2. 将项目根目录和您的数据集文件夹名拼接
DATA_ROOT = PROJECT_ROOT / "my_yolo_data"

train_img_root = str(DATA_ROOT / "train" / "images")
train_label_root = str(DATA_ROOT / "train" / "labels")

val_img_root = str(DATA_ROOT / "val" / "images")
val_label_root = str(DATA_ROOT / "val" / "labels")

# --- 模型与损失函数超参数 ---
# 关键改动: 将 anchors_all 替换为专门为 PASCAL VOC 数据集优化的参数
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
anchors_all = torch.tensor([
    [12, 16], [19, 36], [40, 28],
    [36, 75], [76, 55], [72, 146],
    [142, 110], [192, 243], [459, 401]
], dtype=torch.float32,device=device)
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
strides = [32, 16, 8]
anchor_t = 4.0

# --- 训练超参数 ---

batch_size = 80

image_size = 512
NUM_EPOCHS=300
num_classes = 20
# 学习率调度器相关
# 学习率调度器相关
FINAL_LR_SCALE = 0.01  # 学习率最终衰减到的比例
WARMUP_EPOCHS = 3      # 新增：预热阶段的轮数

LEARNING_RATE = 1e-3  # 调整为更适合SGD的初始学习率
MOMENTUM = 0.937      # 新增：SGD的动量参数
WEIGHT_DECAY = 5e-4

# nms相关
conf_threshold = 0.1
iou_threshold = 0.6
max_det = 100

# --- 系统参数 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
workers = 4

# --- 权重保存路径 ---
save_dir = ROOT

# --- 结束配置 ---


# --- 新增：学习率调度器函数 ---
def linear_warmup_cosine_decay(warmup_epochs, total_epochs, final_lr_scale):
    """
    生成一个用于 LambdaLR 的函数，该函数实现了线性预热和余弦退火策略。
    """
    def lr_lambda(current_epoch):
        """
        根据当前周期数返回学习率的乘数因子。
        """
        # 预热阶段 (Warm-up Phase)
        if current_epoch < warmup_epochs:
            # 学习率从接近0线性增加到1倍 (即初始学习率)
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # 余弦退火阶段 (Cosine Annealing Phase)
        else:
            # 总的衰减周期数
            decay_epochs = total_epochs - warmup_epochs
            # 当前处于衰减的第几个周期
            current_decay_epoch = current_epoch - warmup_epochs
            # 计算余弦函数
            cosine_decay = 0.5 * (1 + math.cos(math.pi * current_decay_epoch / decay_epochs))
            # 最终的乘数是 final_lr_scale 和 1.0 之间的插值
            return final_lr_scale + (1.0 - final_lr_scale) * cosine_decay

    return lr_lambda



def train_one_epoch(model, train_loader, optimizer, device, scaler):
    """
    执行一个完整的训练轮次 (epoch)。

    Args:
        model (torch.nn.Module): 待训练的模型。
        train_loader (DataLoader): 训练数据的加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        device : 训练设备 (CPU或GPU)。
        scaler:
    Returns:
        float: 该epoch的平均损失。
    """
    model.train()
    loop = tqdm(train_loader, leave=False, desc="Training")
    mean_loss = []

    for images, targets, paths,shapes in loop:
        images = images.to(device)
        targets = targets.to(device)

        with autocast():
            raw_predictions = model(images)

            # 【核心修改】: 重塑模型输出以匹配损失函数要求
            reshaped_predictions = []
            num_anchors = 3  # 每个预测头的anchor数量
            for pred in raw_predictions:
                bs, _, grid_h, grid_w = pred.shape
                # [bs, C, H, W] -> [bs, num_anchors, 5+num_classes, H, W]
                pred = pred.view(bs, num_anchors, num_classes + 5, grid_h, grid_w)
                # [bs, num_anchors, 5+num_classes, H, W] -> [bs, num_anchors, H, W, 5+num_classes]
                pred = pred.permute(0, 1, 3, 4, 2).contiguous()
                reshaped_predictions.append(pred)

            # 调用你自己的损失函数
            loss_dict = compute_ciou_loss(reshaped_predictions,
                                          targets,
                                          anchors_all,
                                          anchor_masks,
                                          strides,
                                          anchor_t,
                                          image_size)

            # 将字典中所有损失项求和
            loss = sum(l for l in loss_dict.values())
            assert isinstance(loss, torch.Tensor)

        mean_loss.append(loss.item())

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.6f}")

    avg_epoch_loss = sum(mean_loss) / len(mean_loss)
    print(f"当前Epoch的平均损失为: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

@torch.no_grad()
def evaluate(model, val_loader, device, metric):
    """
        在验证集上评估模型性能。

        Args:
            model (torch.nn.Module): 待评估的模型。
            val_loader (DataLoader): 验证数据的加载器。
            device: 运行评估的设备 (例如, 'cuda' 或 'cpu')。
            metric (torchmetrics.detection.MeanAveragePrecision): 用于计算mAP的度量对象。
    """
    # 将模型设置为评估模式。这会影响像Dropout和BatchNorm这样的层，
    # 确保它们在推理时使用固定的行为。
    model.eval()
    loop = tqdm(val_loader, leave=False,desc="evaluate")

    # ========================== 主评估循环 ==========================
    for images, targets, paths, shapes in loop:
        # 1. 将图像和标签都移动到指定设备
        images = images.to(device)
        targets = targets.to(device)

        # 2. 模型推理: 获取模型的原始输出 (在GPU上)
        raw_prediction = model(images)

        # 3. 后处理 - 解码 (在GPU上)
        # 【关键修改】: 直接传递GPU上的raw_predictions, 并将anchors_all也移动到GPU
        decode_prediction = decode_outputs(raw_prediction,anchors_all,num_classes,anchor_masks,strides)

        # 4. 后处理 - NMS (在GPU上)
        final_predictions_list = non_max_suppression(decode_prediction,conf_threshold,iou_threshold,max_det)

        # 5. 格式化: 为torchmetrics准备预测和真值
        predictions_for_metric = []
        targets_for_metric = []

        # 遍历批次中的每一张图片
        for i, final_predictions in enumerate(final_predictions_list):
            h,w = shapes[i][0]

            # --- 格式化预测 (在GPU上) ---
            # final_predictions_list有可能为空列表，但后续送入predictions_for_metric时不管是否为空，操作都一样
            if final_predictions.shape[0]>0:
                final_predictions[...,:4] = scale_resize_coords((image_size,image_size),final_predictions[...,:4],(h,w))

            predictions_for_metric.append(
                {
                    "boxes":final_predictions[...,:4],
                    "scores":final_predictions[...,4],
                    "labels":final_predictions[...,5].long()
                }
            )

            # --- 格式化真值 (在GPU上) ---
            targets_for_each_image = targets[targets[:,0]==i]
            targets_labels = targets_for_each_image[:,1].long()
            targets_boxes = targets_for_each_image[:,2:]

            original_targets_boxes = targets_boxes.clone()

            original_targets_boxes[..., 0] = original_targets_boxes[..., 0] * w
            original_targets_boxes[..., 1] = original_targets_boxes[..., 1] * h
            original_targets_boxes[..., 2] = original_targets_boxes[..., 2] * w
            original_targets_boxes[..., 3] = original_targets_boxes[..., 3] * h

            targets_for_metric.append(
                {
                    "boxes":xywh2xyxy(original_targets_boxes),
                    "labels" : targets_labels
                }
            )
        # 【新增】: 将当前批次的结果喂给metric对象进行累积
        metric.update(predictions_for_metric, targets_for_metric)

    # ========================== 循环结束后 ==========================
    # 【新增】: 计算整个验证集的最终mAP结果
    results = metric.compute()

    # 【新增】: 重置metric对象的状态，为下一次评估做准备
    metric.reset()

    # 【新增】: 返回包含所有指标的字典
    return results








# --- 4. 主执行函数 ---
def main():
    """主函数，负责组装所有部件并启动训练流程。"""
    model = YOLOv3(num_classes, anchor_masks).to(device)


    # --- 优化器 ---
    # 【重要】: 只将需要训练的参数传递给优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=LEARNING_RATE, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY, nesterov=True)

    # 创建包含预热和余弦退火的学习率调度器
    lr_scheduler_func = linear_warmup_cosine_decay(WARMUP_EPOCHS, NUM_EPOCHS, FINAL_LR_SCALE)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_func)

    # 混合精度缩放器
    scaler = GradScaler()



    # ------------------- 准备数据 -------------------
    train_dataset = YOLOv3Dataset(train_img_root, train_label_root)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              collate_fn=YOLOv3Dataset.collate_fn
                              )

    val_dataset = YOLOv3Dataset(val_img_root, val_label_root)

    val_loader = DataLoader(val_dataset, batch_size, shuffle=False,
                              collate_fn=YOLOv3Dataset.collate_fn
                              )
    # --- 【新增】初始化mAP计算器和最佳指标 ---
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    best_map50 = 0.0

    # ------------------- 执行主训练循环 -------------------
    print("--- 开始训练 ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1} / {NUM_EPOCHS} ---")

        train_one_epoch(model, train_loader,optimizer, device,scaler, )

        # 在每个周期结束后更新学习率
        scheduler.step()

        # --- 【新增】执行评估 ---
        results = evaluate(model, val_loader, device, metric)
        map50 = results['map_50'].item()

        print(f"--- 验证结果 Epoch {epoch + 1} ---")
        print(f"mAP@.50:.95: {results['map'].item():.4f}")
        print(f"mAP@.50: {map50:.4f}")

        # --- 【新增】保存模型权重 ---
        # 1. 保存最新的 checkpoint, 包含恢复训练所需的所有信息
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map50': best_map50,
        }
        torch.save(checkpoint, save_dir / 'last.pt')
        print(f"最新 checkpoint 已保存至 {save_dir / 'last.pt'}")

        # 2. 如果当前 mAP@.50 是历史最佳，则单独保存一份 best.pt
        if map50 > best_map50  :
            best_map50 = map50
            torch.save(model.state_dict(), save_dir / 'best.pt')
            print(f"*** 发现新的最佳模型 (mAP@.50: {best_map50:.4f})! 已保存至 {save_dir / 'best.pt'} ***")


    print("--- 训练完成 ---")


# --- 5. 脚本入口 ---
if __name__ == "__main__":
    main()










