# 复现论文
# ResNet strikes back: An improved training procedure in timm

def get_padding(kernel_size: int,stride: int, dilation: int = 1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


