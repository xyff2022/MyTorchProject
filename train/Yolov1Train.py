import torch
from torch import optim
import albumentations as a
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from Model.yolov1 import YOLOv1, YoloLoss
from dataset.YOLODataset import YOLODataset
from dataset.yolo_v1_dataset import YoloV1Dataset

# --- 1. 初始化 ---
# --- 设置超参数和全局配置 ---
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10 # 训练的总轮数
WEIGHT_DECAY = 0 # 权重衰减，一种正则化方法

# YOLOv1 模型和数据的特定参数
S = 7
B = 2
C = 20

# 数据集路径
IMG_DIR = "../data/HelmetDataset-YOLO-Train/images"
LABEL_DIR = "../data/HelmetDataset-YOLO-Train/labels"


# --- 2. 定义数据增强和加载 ---

# 定义用于训练集的数据增强流水线
# (这里可以加入更多您想尝试的增强，如ShiftScaleRotate, ColorJitter等)
train_transform = a.Compose(
    [
        a.Resize(width=448, height=448),
        a.HorizontalFlip(p=0.5), # 50%概率水平翻转
        a.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        a.ToTensorV2(),
    ],
    bbox_params=a.BboxParams(
        format="yolo",
        label_fields=['class_labels']
    )
)

# (可选) 定义用于验证集的数据增强流水线 (通常只做必要处理)
# val_transform = a.Compose(...)

# --- 3. 封装单次训练循环的函数 ---
def train_one_epoch(loader, model, optimizer, loss_fn):
    """
       执行一个完整的训练周期 (epoch)。

       参数:
           loader (DataLoader): 训练数据加载器。
           model (Module): 要训练的模型。
           optimizer (Optimizer): 优化器。
           loss_fn (Module): 损失函数。
    """
    # 使用tqdm来创建一个可视化的进度条
    loop = tqdm(loader, leave = True)
    mean_loss = []

    # 将模型设置为训练模式，这会启用Dropout等层
    model.train()

    for batch_idx, (images, targets) in enumerate(loop):
        # 将数据移动到指定的设备 (GPU或CPU)
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # 1. 前向传播: 将图片送入模型得到预测结果
        prediction = model(images)

        # 2. 计算损失: 比较预测结果和真实标签
        loss = loss_fn(prediction, targets)
        mean_loss.append(loss.item())

        # 3. 反向传播: 清空梯度 -> 计算梯度 -> 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新tqdm进度条的后缀信息，实时显示当前损失
        loop.set_postfix(loss=loss.item())
    # 打印当前epoch的平均损失
    print(f"当前Epoch的平均损失为: {sum(mean_loss) / len(mean_loss):.4f}")

# --- 4. 主执行函数 ---
def main():
    """
        主函数，负责组装所有部件并启动训练流程。
    """
    # 初始化模型、损失函数和优化器
    model = YOLOv1().to(DEVICE)
    loss_fn = YoloLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr= LEARNING_RATE,
        weight_decay= WEIGHT_DECAY
    )

    # 准备数据加载器
    train_dataset = YoloV1Dataset(IMG_DIR, LABEL_DIR, S, C, train_transform)
    train_loder = DataLoader(train_dataset,
                             BATCH_SIZE,
                             shuffle=True,
                             drop_last=True,
                             pin_memory= (DEVICE == "cuda"),
                             num_workers= 2 if DEVICE == "cuda" else 0
                             )

    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch [{epoch + 1} / {NUM_EPOCHS}] ---")

        train_one_epoch(train_loder, model, optimizer,loss_fn)

        # (推荐) 在每个epoch或每隔几个epoch后保存模型
        # if (epoch + 1) % 2 == 0:
        #     torch.save(model.state_dict(), f"yolo_v1_epoch_{epoch + 1}.pth")
        #     print(f"已保存模型: yolo_v1_epoch_{epoch + 1}.pth")

    print("\n--- 训练结束 ---")
    # torch.save(model.state_dict(), "yolo_v1_final.pth")
    # print("已保存最终模型: yolo_v1_final.pth")


# --- 5. 脚本入口 ---
# 确保只有在直接运行这个文件时，main()函数才会被执行
if __name__ == "__main__":
    main()

