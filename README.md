# PyTorch 经典模型学习与实践

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

这是一个用于学习和实践经典深度学习模型的 PyTorch 项目。本项目从零开始，逐步实现各种著名网络架构，并包含了自定义数据集加载、模型训练与测试等模块。

## 🌟 项目特点

- **模块化实现**: 将复杂的网络结构拆分为独立的、可复用的模块，代码结构清晰。
- **经典模型复现**: 包含了对经典计算机视觉模型 **ResNet** 等的完整实现。
- **自定义数据集**: 支持 **VOC** 和 **YOLO** 两种主流标注格式的数据集加载器。
- **版本控制**: 使用 Git 和 GitHub 进行完整的版本控制和远程同步。

## 📂 项目结构

项目遵循关注点分离的原则，将不同功能的代码组织在独立的模块中。

```
torchProject/
├── Model/              # 存放所有模型定义文件 (如 ResNet.py)
├── dataset/            # 存放所有自定义数据集加载器 (如 YOLODataset.py)
├── data/               # (本地) 存放原始数据集 (被 .gitignore 忽略)
├── logs/               # (本地) 存放 TensorBoard 日志 (被 .gitignore 忽略)
├── .gitignore          # Git 忽略文件配置
├── requirements.txt    # 项目依赖包列表
└── README.md           # 本文档
```

| 目录/文件        | 功能说明                                                     |
| :--------------- | :----------------------------------------------------------- |
| `Model/`         | **核心模块**: 存放所有神经网络模型的 Python 定义。           |
| `dataset/`       | **核心模块**: 存放所有自定义数据集加载类 (`Dataset`)。       |
| `data/`          | **本地数据**: 用于存放训练、验证和测试的原始数据。           |
| `logs/`          | **本地日志**: 用于存放训练过程中产生的日志文件 (如 TensorBoard)。 |
| `requirements.txt` | 记录项目运行所需的全部 Python 依赖包。                     |

## 🚀 快速开始

### 1. 环境准备

请确保您的本地已安装 Python (建议 3.10 或更高版本) 和 Git。

**克隆仓库**
```bash
git clone https://github.com/xyff2022/MyTorchProject.git
cd MyTorchProject
```

**创建并激活虚拟环境**
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux/macOS)
# source .venv/bin/activate
```

### 2. 安装依赖

本项目的所有依赖项均在 `requirements.txt` 中。
```bash
pip install -r requirements.txt
```

### 3. 准备数据集

在项目根目录下创建 `data` 文件夹，并将您的数据集存放其中。数据集应遵循特定格式，例如 YOLO 格式：

```
data/
└── your_dataset/
    ├── train/
    │   ├── images/
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── 1.txt
    │       └── ...
    └── val/
        ├── images/
        └── labels/
```

### 4. 测试模型

`Model` 目录下的每个模型文件都可以独立运行，以验证模型结构是否正确。
```bash
python Model/ResNet.py
```
如果一切正常，您将看到 ResNet-34 和 ResNet-50 模型的输入输出尺寸验证信息。

### 5. 开始训练 (示例)

要开始一个完整的训练流程，您可以创建一个 `train.py` 脚本，在其中：

1.  **导入模型**: `from Model.ResNet import resnet34`
2.  **创建数据集**: `from dataset.YOLODataset import YOLODataset`
3.  **实例化对象**: 创建 `Dataset` 和 `DataLoader` 实例。
4.  **定义训练组件**: 定义损失函数和优化器。
5.  **开始训练**: 编写训练和验证循环。

祝您使用愉快！
