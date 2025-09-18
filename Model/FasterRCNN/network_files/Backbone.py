import torch

from Model.ResNet.network_files.ResNet_Pro import ResNet50, BottleneckResidualBlock


class ResNet50Backbone(ResNet50):
    def __init__(self, block, n_blocks, in_channels, n_channels):
        super().__init__(block, n_blocks, in_channels, n_channels)

        # 定义一个输出通道数的属性，方便后续模块（如RPN）获取
        self.out_channels = 1024

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)# <--- 在 layer3 结束后停止

        return x # <--- 直接返回 layer3 的输出

def get_my_resnet50_backbone():
    """工厂函数，方便地创建我们的主干网络"""
    resnet50_blocks = (3, 4, 6, 3)
    resnet50_channels = (64, 128, 256, 512)
    in_channels = 3

    Model = ResNet50Backbone(
        BottleneckResidualBlock,
        resnet50_blocks,
        in_channels,
        resnet50_channels
    )
    return Model


# --- 测试代码 ---
if __name__ == '__main__':
    backbone = get_my_resnet50_backbone()
    print("Backbone Model Structure is inherited from your ResNet50.")

    dummy_image = torch.randn(1, 3, 800, 800)
    print(f"\nInput image shape: {dummy_image.shape}")

    # 将图片输入主干网络，得到特征图
    feature_map = backbone(dummy_image)

    # 打印输出特征图的形状
    print(f"Output feature map shape: {feature_map.shape}")

    stride = int(dummy_image.shape[2] / feature_map.shape[2])
    print(f"Calculated stride: {stride}")

    out_channels = feature_map.shape[1]
    print(f"Output feature map channels: {out_channels}")

    assert stride == 16, "Stride should be 16!"
    assert out_channels == 1024, "Output channels should be 1024!"
    # 也可以直接访问我们定义的属性
    assert backbone.out_channels == 1024

    print("\nYour custom ResNet-50 Backbone test passed successfully!")

