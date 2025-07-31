# PyTorch 经典模型学习与实践

这是一个用于学习和实践经典深度学习模型的 PyTorch 项目。本项目从零开始，逐步实现各种著名网络架构，并包含了自定义数据集加载、模型训练与测试等模块。

## 🌟 项目特点

* **模块化实现**: 将复杂的网络结构拆分为独立的、可复用的模块，代码结构清晰。  
* **经典模型复现**: 包含了对经典计算机视觉模型 **ResNet** 等的完整实现。  
* **自定义数据集支持**: 实现了对标准 ImageFolder 结构以及自定义 **YOLO 标注格式**和**VOC数据集** 的数据集加载器。  
* **版本控制**: 使用 Git 和 GitHub进行完整的版本控制和远程同步。

## 📂 项目结构

torchProject/  
├── .gitignore          \# Git忽略文件配置  
├── Model/              \# 存放所有模型定义文件  
│   ├── Model01.py  
│   └── ResNet.py  
├── utils/              \# (可选) 存放辅助工具和函数  
├── dataset/            \# 存放所有自定义数据集加载器  
│   ├── vocDataset.py  
│   └── YOLODataset.py  
├── data/               \# (本地) 存放原始数据集文件 (被.gitignore忽略)  
├── logs/               \# (本地) 存放TensorBoard日志 (被.gitignore忽略)  
├── requirements.txt    \# 项目所需的Python依赖包列表  
└── README.md           \# 本说明文档

## 🚀 快速开始

### 1\. 环境准备

首先，请确保您的本地电脑或服务器上已经安装了 Git 和 Python 3.12。

**克隆仓库**

git clone https://github.com/xyff2022/MyTorchProject.git  
cd MyTorchProject

**创建并激活虚拟环境**

\# 创建虚拟环境  
python \-m venv .venv

\# 激活虚拟环境 (Windows)  
.venv\\Scripts\\activate

\# 激活虚拟环境 (Linux/macOS)  
\# source .venv/bin/activate

### 2\. 安装依赖

本项目的所有依赖都记录在 requirements.txt 文件中。运行以下命令进行安装：

pip install \-r requirements.txt

### 3\. 准备数据集

本项目不包含数据集文件。请根据您的需求，按照以下结构准备您的数据集：

torchProject/  
└── data/  
    ├── train/  
    │   ├── class\_a/  
    │   │   ├── image1.jpg  
    │   │   └── ...  
    │   └── class\_b/  
    │       └── ...  
    └── val/  
        ├── class\_a/  
        │   └── ...  
        └── class\_b/  
            └── ...

### 4\. 运行模型测试

所有模型代码均放置在 Model 文件夹中。

您可以直接运行 Model 文件夹中的模型文件（如ResNet.py）来测试模型结构是否正确搭建：

python Model/ResNet.py

如果一切正常，您将看到 ResNet-34 和 ResNet-50 模型的输入输出尺寸验证通过的信息。

### 5\. 开始训练 (示例)

要开始一个完整的训练流程，您需要创建一个训练脚本（例如 train.py），在其中：

1. 导入您的模型（如 from Model.ResNet import resnet34）  
2. 创建您的 Dataset 和 DataLoader 实例。  
3. 定义损失函数和优化器。  
4. 编写训练和验证循环。