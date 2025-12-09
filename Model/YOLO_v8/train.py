import os
import sys
import math
from pathlib import Path

import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
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

from models.yolo_v8 import YOLOv8
from models.metrics import non_max_suppression_v8, xywh2xyxy, xyxy2xywhn, xywhn2xyxy
from models.loss_v8 import v8loss
from models.dataloaders import create_dataloader

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
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
strides = [8, 16, 32]

LOG_FILE = "training_v8.txt"

# --- 训练超参数 ---

batch_size = 52

image_size = 640
NUM_EPOCHS = 50
CLOSE_MOSAIC_EPOCHS = 50  # [!!!] 新增: 最后 30 个 epoch 关闭 mosaic
num_classes = 20
# 学习率调度器相关
# 学习率调度器相关
FINAL_LR_SCALE = 0.01  # 学习率最终衰减到的比例
WARMUP_EPOCHS = 3  # 新增：预热阶段的轮数

LEARNING_RATE = 1e-2  # 调整为更适合SGD的初始学习率
MOMENTUM = 0.937  # 新增：SGD的动量参数
WEIGHT_DECAY = 5e-4

# NMS 参数（评估/推理用）
conf_threshold = 0.25
iou_threshold = 0.45
max_det = 300

# --- 系统参数 ---
workers = 8

# --- 权重保存路径 ---
save_dir = ROOT
save_dir.mkdir(parents=True, exist_ok=True)
best_pt_path = save_dir / "best.pt"
last_pt_path = save_dir / "last.pt"
PRETRAINED_WEIGHTS = save_dir / "yolov5s_converted.pt"


# --- 结束配置 ---

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=20, img_size=640):
    """
        在验证集上评估模型性能。
        (此版本已从 v3 逻辑修改为 v5 逻辑)

        Args:
            model (torch.nn.Module): 待评估的模型。
            val_loader (DataLoader): 验证数据的加载器。
            device: 运行评估的设备 (例如, 'cuda' 或 'cpu')。
            num_classes (int): 数据集中的类别数。
            img_size (int): 评估时使用的图像尺寸。
    """
    # [!!!] 核心修改: 在函数内部创建 metric 对象
    # (不再需要 map_calculator 类)
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type="bbox",
        class_metrics=True
    ).to(device)

    # 1. 将模型设置为评估模式
    model.eval()
    loop = tqdm(val_loader, leave=False, desc="Evaluate")

    # ========================== 主评估循环 ==========================
    # 只返回 images 和 targets
    for images, targets in loop:
        # 1. 将图像和标签都移动到指定设备
        images = images.to(device)
        targets = targets.to(device)

        # 模型 eval 时返回 (inference_out, raw_feats)
        infer_out, _ = model(images)  # infer_out: (bs, N, 4+nc)
        # distance2bbox(..., xywh=True)，返回 xywh，需要转 xyxy
        infer_out[..., :4] = xywh2xyxy(infer_out[..., :4])

        # NMS
        nms_outputs = non_max_suppression_v8(
            infer_out, conf_thres=conf_threshold, iou_thres=iou_threshold, max_det=max_det, nc=num_classes
        )

        # 组织 metric 所需格式
        predictions_for_metric = []
        targets_for_metric = []

        # 遍历批次中的每一张图片
        for i in range(images.shape[0]):  # images.shape[0] 就是 batch_size

            # --- 格式化预测 ---
            final_predictions = nms_outputs[i]

            predictions_for_metric.append(
                {
                    "boxes": final_predictions[..., :4],
                    "scores": final_predictions[..., 4],
                    "labels": final_predictions[..., 5].long()
                }
            )

            # --- 格式化真值 ---
            # (这部分逻辑来自 v3，但使用了 v5 的坐标转换函数)
            targets_for_each_image = targets[targets[:, 0] == i]
            targets_labels = targets_for_each_image[:, 1].long()

            # 拿到归一化的 [xn, yn, wn, hn]
            targets_boxes_xywhn = targets_for_each_image[:, 2:]

            # [!!!] 使用 v5 的辅助函数将其转换为 640x640 上的 xyxy
            targets_boxes_xyxy = xywhn2xyxy(
                targets_boxes_xywhn, w=img_size, h=img_size
            )

            targets_for_metric.append(
                {
                    "boxes": targets_boxes_xyxy,
                    "labels": targets_labels
                }
            )

            # 6. 将当前批次的结果喂给 metric 对象进行累积
        metric.update(predictions_for_metric, targets_for_metric)

        # ========================== 循环结束后 ==========================
    # 7. 计算整个验证集的最终mAP结果
    results = metric.compute()

    # 8. 重置metric对象的状态，为下一次评估做准备
    metric.reset()

    # 9. 评估结束后，将模型恢复到训练模式
    model.train()

    # 10. 返回包含所有指标的字典
    return results


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


def main():
    # 1. 初始化模型
    # [!!] 确保 YOLOv5 的构造函数接受 anchors_config 参数
    model = YOLOv8(
        number_classes=num_classes,
    ).to(device)

    # ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
    # load_result = model.load_state_dict(ckpt, strict=False)
    #
    # # 1. 将所有参数先设置为 '不需要梯度' (冻结)
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # # 2. 只解冻 'missing_keys' (即检测头)
    # unfrozen_count = 0
    # if load_result.missing_keys:
    #     for name, param in model.named_parameters():
    #         if name in load_result.missing_keys:
    #             param.requires_grad = True
    #             unfrozen_count += 1
    #             # 打印前 5 个被解冻的层
    #             if unfrozen_count <= 5:
    #                 print(f"  > 解冻 (训练): {name}")
    #     if unfrozen_count > 5:
    #         print(f"  > ... (等 {unfrozen_count - 5} 个层)")
    #
    # if unfrozen_count == 0:
    #     print("[警告] 'missing_keys' 为空。没有层被解冻。请检查权重文件是否正确。")
    # else:
    #     print(f"--- 共解冻 {unfrozen_count} 个参数层进行训练 ---")
    # # --- 冻结逻辑结束 ---
    #
    # if load_result.unexpected_keys:
    #     print(f"  [信息] 忽略的键 (通常是旧模型的检测头):")
    #     print(f"    > {load_result.unexpected_keys[:5]}...")

    # if load_result.missing_keys:
    #     for name, param in model.named_parameters():
    #         if name in load_result.missing_keys:
    #             param.requires_grad = True

    # 3. 初始化优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        pg,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True  # YOLOv5 推荐
    )

    # 2. 初始化损失函数
    # [!!] 确保 ComputeCiou 接受模型作为输入
    loss_fn = v8loss(model).to(device)

    # 4. 初始化学习率调度器
    lf = linear_warmup_cosine_decay(
        WARMUP_EPOCHS,
        NUM_EPOCHS,
        FINAL_LR_SCALE,
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 6. 创建 Dataloaders
    print("正在创建 Dataloaders...")
    train_loader, dataset = create_dataloader(
        train_img_root,
        img_size=image_size,
        batch_size=batch_size,
        stride=max(strides),  # 使用最大步长
        augment=True,
        cache=True,  # 推荐开启 cache 'in RAM'
        workers=workers,
        shuffle=True
    )

    val_loader, _ = create_dataloader(
        val_img_root,
        img_size=image_size,
        batch_size=batch_size,
        stride=max(strides),
        augment=False,
        cache=True,  # 推荐开启 cache 'in RAM'
        workers=workers,
        shuffle=False
    )
    print("Dataloaders 创建完毕。")

    # 7. 训练循环
    best_map50 = 0.0

    for epoch in range(NUM_EPOCHS):

        # [!!!] 新增: 检查是否需要关闭 mosaic (数据增强)
        if epoch == (NUM_EPOCHS - CLOSE_MOSAIC_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS}. 关闭 Mosaic 和数据增强 ---")

            # 销毁旧的 train_loader (有助于释放 cache 内存)
            if 'train_loader' in locals() and train_loader is not None:
                try:
                    del train_loader
                except Exception as e:
                    print(f"警告: 无法删除旧的 train_loader: {e}")

            # 重新创建 train_loader, 关闭数据增强 (augment=False)
            train_loader, dataset = create_dataloader(
                train_img_root,
                img_size=image_size,
                batch_size=batch_size,
                stride=max(strides),
                augment=False,  # [!!!] 关键: 关闭所有增强 (包括 mosaic)
                cache=True,  # 保持 cache=True
                workers=workers,
                shuffle=True  # 保持 shuffle=True
            )
            print("Dataloader 已更新，数据增强已关闭。")

        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_box, epoch_cls, epoch_dfl = 0.0, 0.0, 0.0

        for i, (images, targets) in enumerate(pbar):
            images = images.to(DEVICE)  # [!!] 修正: dataloader 已返回 0.0-1.0 的 float tensor
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            train_outputs = model(images)

            total_loss, loss_items = loss_fn(train_outputs, targets)


            total_loss.backward()
            optimizer.step()

            # 记录分项（loss_items: tensor([box, cls, dfl]))
            if loss_items is not None and loss_items.numel() == 3:
                epoch_box += loss_items[0].item()
                epoch_cls += loss_items[1].item()
                epoch_dfl += loss_items[2].item()

            steps = i + 1
            pbar.set_postfix(
                box=f"{epoch_box / steps:.3f}",
                cls=f"{epoch_cls / steps:.3f}",
                dfl=f"{epoch_dfl / steps:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        # --- Epoch 结束 ---

        # 8. 更新学习率
        scheduler.step()

        #  保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        # 保存 last.pt
        torch.save(checkpoint, last_pt_path)
        print(f"已保存最新模型到: {last_pt_path}")

        if epoch > 0:
            # 9. 运行评估
            print("正在评估...")
            results = evaluate(
                model,
                val_loader,
                device=DEVICE,
                num_classes=num_classes,
                img_size=image_size,

            )

            map50 = results['map_50'].item()
            map = results['map'].item()
            print(f"Epoch {epoch + 1} 评估结果: mAP@.5:.95 = {map:.4f}, mAP@.5 = {map50:.4f}")

            # 保存 best.pt
            if map50 > best_map50:
                best_map50 = map50
                torch.save(checkpoint, best_pt_path)
                print(f"*** 新的最佳模型 (mAP@.5 = {best_map50:.4f})! 已保存到: {best_pt_path} ***")

            try:
                with open(LOG_FILE, 'a') as f:
                    # 'a' 模式 = 追加 (append)
                    f.write(f"--- Epoch {epoch} ---")
                    f.write(f"mAP @ .5:.95: {results['map']:.4f}")
                    f.write(f"mAP @ .5:     {results['map_50']:.4f}")
                    f.write(f"mAP @ .75:    {results['map_75']:.4f}\n")  # 多加-个换行
            except Exception as e:
                print(f"写入日志文件失败: {e}")

    print(f"训练完成。最佳 mAP@.5: {best_map50:.4f} (保存在 {best_pt_path})")


# =============================== 4. 运行入口 ===============================
if __name__ == "__main__":
    main()

