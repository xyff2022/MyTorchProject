import torch
import torch.nn
from torch.nn import Module, Conv2d, Sequential, MaxPool2d, Upsample

from .dark_net_53 import BasicConv, darknet53


class YOLOv3(Module):
    def __init__(self,
                 # anchors,
                 num_classes,
                 # input_shape,
                 anchor_mask=[[6,7,8], [3,4,5], [0,1,2]]):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_mask = anchor_mask

        # ---------------------------------------------------#
        #   Backbone: Darknet-53
        #   输入: [B, 3, 512, 512]
        #   输出: out3(64x64x256), out4(32x32x512), out5(16x16x1024)
        # ---------------------------------------------------#
        self.backbone = darknet53()
        self.sequential1 = Sequential(
            BasicConv(1024, 512, 1),
            BasicConv(512, 1024, 3, 1),
            BasicConv(1024, 512, 1)
        )
        self.sequential2 = Sequential(
            BasicConv(2048, 512, 1),
            BasicConv(512, 1024, 3, 1),
            BasicConv(1024, 512, 1)
        )


        # ------------------------------------------------------------------------#
        #   SPP 模块 + 第一个 Convolutional Set
        #   处理来自 backbone 的最深层特征 out5
        # ------------------------------------------------------------------------#

        # SPP
        self.spp_pool1 = MaxPool2d(5, 1, 2)
        self.spp_pool2 = MaxPool2d(9, 1, 4)
        self.spp_pool3 = MaxPool2d(13, 1, 6)

        # ------------------------------------------------------------------------#
        #   第一个 YOLO Head (预测大物体)
        # ------------------------------------------------------------------------#
        self.head1 = Sequential(
            BasicConv(512, 1024, 3),
            Conv2d(1024, len(anchor_mask[0]) * (num_classes + 1 + 4), 1)
        )

        # ------------------------------------------------------------------------#
        #   上采样与特征融合 (out5 + out4)
        # ------------------------------------------------------------------------#
        self.upsample1 = Sequential(
            BasicConv(512, 256, 1),
            Upsample(scale_factor= 2, mode= "nearest")
        )

        # 第一个 Convolutional Set (处理融合后的特征)
        self.convolutional_set1 = Sequential(
            BasicConv(768, 256, 1),
            BasicConv(256, 512, 3),
            BasicConv(512, 256, 1),
            BasicConv(256, 512, 3),
            BasicConv(512, 256, 1)
        )

        # ------------------------------------------------------------------------#
        #   第二个 YOLO Head (预测中物体)
        # ------------------------------------------------------------------------#
        self.head2 = Sequential(
            BasicConv(256, 512, 3),
            Conv2d(512, len(anchor_mask[1]) * (num_classes + 1 + 4), 1)
        )

        # ------------------------------------------------------------------------#
        #   上采样与特征融合 (out4 + out3)
        # ------------------------------------------------------------------------#
        self.upsample2 = Sequential(
            BasicConv(256, 128, 1),
            Upsample(scale_factor=2, mode="nearest")
        )

        # 第二个 Convolutional Set (处理融合后的特征)
        self.convolutional_set2 = Sequential(
            BasicConv(384, 128, 1),
            BasicConv(128, 256, 3),
            BasicConv(256, 128, 1),
            BasicConv(128, 256, 3),
            BasicConv(256, 128, 1)
        )
        # ------------------------------------------------------------------------#
        #   第三个 YOLO Head (预测小物体)
        # ------------------------------------------------------------------------#
        self.head3 = Sequential(
            BasicConv(128, 256, 3),
            Conv2d(256, len(anchor_mask[2]) * (num_classes + 1 + 4), 1)
        )


    def forward(self, x):
        out3, out4, out5 = self.backbone(x)

        spp_out5 = self.sequential1(out5)
        spp_pool1 = self.spp_pool1(spp_out5)
        spp_pool2 = self.spp_pool2(spp_out5)
        spp_pool3 = self.spp_pool3(spp_out5)

        spp = torch.cat([spp_out5,spp_pool1,spp_pool2,spp_pool3], dim = 1)
        route1 = self.sequential2(spp)

        out_l = self.head1(route1)

        before_first_concat = self.upsample1(route1)
        first_concat = torch.cat([before_first_concat, out4], dim = 1)
        route2 = self.convolutional_set1(first_concat)

        out_m = self.head2(route2)

        before_second_concat = self.upsample2(route2)
        second_concat = torch.cat([before_second_concat, out3], dim = 1)
        route3 = self.convolutional_set2(second_concat)

        out_s = self.head3(route3)


        return out_l, out_m, out_s




