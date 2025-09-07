import torch
from torch import nn
from torchvision.ops import roi_pool, roi_align

"""
torchvision.ops.roi_pool(input: Tensor, 
                         boxes: Union[Tensor, list[torch.Tensor]], 
                         output_size: None, 
                         spatial_scale: float = 1.0
                         ) → Tensor

Parameters:
    input (Tensor[N, C, H, W]) – The input tensor, i.e. a batch with N elements. Each element contains C feature maps of dimensions H x W.

    boxes (Tensor[K, 5] or List[Tensor[L, 4]]) – the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from. 

    output_size (int or Tuple[int, int]) – the size of the output after the cropping is performed, as (height, width)

    spatial_scale (float) – a scaling factor that maps the box coordinates to the input coordinates. Default: 1.0

Returns:
    The pooled RoIs.

Return type:
    Tensor[K, C, output_size[0], output_size[1]]
"""

class RoIPool(nn.Module):
    def __init__(self, output_size, spatial_scale = 1):
        """
            初始化 RoI Pooling 层。

            Args:
                output_size (tuple or int): 池化后输出的尺寸, (height, width)。
                                                如果是一个整数，则高宽相等。例如 7 or (7, 7)。
                spatial_scale (float): 空间尺度因子，用于将 RoI 坐标从输入图像尺度缩放到特征图尺度。
                                         这个值等于 1.0 / backbone_stride。
        """
        super().__init__()

        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, proposals):
        """
            对 RoI 进行池化操作。

               Args:
                   input (Tensor[N, C, H, W]) : 主干网络输出的特征图，形状为 [B, C, H, W]。
                   proposals (Tensor[K, 5] or List[Tensor[L, 4]]): ProposalCreator 输出的候选区域，
                                        格式为 (batch_index, xmin, ymin, xmax, ymax)。
                Returns:
                    torch.Tensor: 池化后的特征，Tensor[K, C, output_size[0], output_size[1]]。
        """
        return roi_pool(input, proposals, self.output_size, self.spatial_scale)



"""
torchvision.ops.roi_align(input: Tensor, 
                          boxes: Union[Tensor, list[torch.Tensor]], 
                          output_size: None, 
                          spatial_scale: float = 1.0, 
                          sampling_ratio: int = - 1, 
                          aligned: bool = False
                          ) → Tensor                  
                          
Parameters:
    input (Tensor[N, C, H, W]) – The input tensor, i.e. a batch with N elements. Each element contains C feature maps of dimensions H x W. If the tensor is quantized, we expect a batch size of N == 1.

    boxes (Tensor[K, 5] or List[Tensor[L, 4]]) – the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from. 
    
    output_size (int or Tuple[int, int]) – the size of the output (in bins or pixels) after the pooling is performed, as (height, width).

    spatial_scale (float) – a scaling factor that maps the box coordinates to the input coordinates. Default: 1.0

    sampling_ratio (int) – number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio sampling points per bin are used. If <= 0, then an adaptive number of grid points are used (computed as ceil(roi_width / output_width), and likewise for height). Default: -1

    aligned (bool) – If False, use the legacy implementation. If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two neighboring pixel indices. This version is used in Detectron2

Returns:
    The pooled RoIs.

Return type:
    Tensor[K, C, output_size[0], output_size[1]]
                          
"""


class RoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale =1, sampling_ratio = -1, aligned =False):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, proposals):
        return roi_align(input, proposals,
                         self.output_size,
                         self.spatial_scale,
                         self.sampling_ratio,
                         self.aligned)



if __name__ == '__main__':
    """
        一个完整的测试函数，用于演示和验证 RoIPool 和 RoIAlign 模块。
        """
    print("--- 正在测试 Pooling.py中的 RoIPool 和 RoIAlign ---")

    # --- 1. 准备通用输入数据 ---
    # 定义通用参数
    output_size = (7, 7)
    stride = 16
    spatial_scale = 1.0 / stride

    # 创建假的特征图，batch_size=2
    feature_map = torch.randn(2, 256, 50, 50)

    # 创建假的候选区域 (RoIs)
    # 3个RoI, 其中2个属于图片0 (batch_index=0), 1个属于图片1 (batch_index=1)
    rois = torch.tensor([
        [0, 21.5, 43.1, 126.8, 121.3],  # RoI 1 on image 0
        [0, 100.2, 150.9, 300.1, 400.6],  # RoI 2 on image 0
        [1, 50.0, 50.0, 250.0, 250.0]  # RoI 3 on image 1
    ], dtype=torch.float32)

    print(f"\n输入特征图形状: {feature_map.shape}")
    print(f"输入 RoIs 形状: {rois.shape}")

    # --- 2. 测试 RoIPool ---
    print("\n--- 测试 RoIPool ---")
    try:
        # 实例化 RoIPool
        roi_pooler = RoIPool(output_size, spatial_scale)

        # 在一个更大的模型中，你可以这样使用类型提示：
        # pooler: Union[RoIPool, RoIAlign] = roi_pooler

        # 进行前向传播
        output_pool = roi_pooler(feature_map, rois)

        print("RoIPool forward 方法成功运行！")
        print(f"RoIPool 输出特征形状: {output_pool.shape}")

        # 验证输出形状
        expected_shape = (rois.shape[0], feature_map.shape[1], output_size[0], output_size[1])
        assert output_pool.shape == expected_shape, "RoIPool 输出形状不匹配!"
        print("RoIPool 测试通过！")

    except Exception as e:
        print(f"RoIPool 测试失败: {e}")

    # --- 3. 测试 RoIAlign ---
    print("\n--- 测试 RoIAlign ---")
    try:
        # 实例化 RoIAlign
        roi_aligner = RoIAlign(output_size, spatial_scale, sampling_ratio=2)

        # 进行前向传播
        output_align = roi_aligner(feature_map, rois)

        print("RoIAlign forward 方法成功运行！")
        print(f"RoIAlign 输出特征形状: {output_align.shape}")

        # 验证输出形状
        expected_shape = (rois.shape[0], feature_map.shape[1], output_size[0], output_size[1])
        assert output_align.shape == expected_shape, "RoIAlign 输出形状不匹配!"
        print("RoIAlign 测试通过！")

    except Exception as e:
        print(f"RoIAlign 测试失败: {e}")