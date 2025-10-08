import torch
import torch.nn
from torch.nn import Module, Conv2d, Sequential, MaxPool2d, Upsample



class DecodeBox(Module):
    def __init__(self, anchors, num_classes, image_shape):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.image_shape = image_shape
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #   [12,16],  [19,36],  [40,28],  [36,75],  [76,55],  [72,146],  [142,110],  [192,243],  [459,401]
        #-----------------------------------------------------------#


    def forward(self, input):
        """

        :param input: 模型的任何一个输入,out_s或out_m或out_l，形状 (batch_size, 3 * (5 + num_classes), H, W)
        :return: 一个形状为(B, N, 4 + 1 + num_classes)的张量。对于批次中的每一张图，它都包含N个预测结果，
        每个结果都是一个长度为5 + 类别数的向量
        """
        # -----------------------------------------------#
        #   input的shape为 (batch_size, 3 * (5 + num_classes), H, W)
        #   H, W 即为 16, 32, 64
        # -----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # -----------------------------------------------#
        #   stride_h = stride_w = 512 / H
        # -----------------------------------------------#
        stride_h = self.image_shape / input_height
        stride_w = self.image_shape / input_width
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in self.anchors]



        #-----------------------------------------------#
        #   输入的input一共有三种，他们的shape分别是
        #   (batch_size, 3 * (5+num_classes), 16, 16)
        #   (batch_size, 3 * (5+num_classes), 32, 32)
        #   (batch_size, 3 * (5+num_classes), 64, 64)
        #   将其变形为 (batch_size, 3, H, W, 5 + num_classes)
        #-----------------------------------------------#
        prediction = input.view(batch_size,
                                 len(self.anchors),
                                 self.num_classes + 5,
                                 input_height,
                                 input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        prediction_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #   (batch_size, 3, H, W)
        #----------------------------------------------------------#
        # 详细描述grid_x的生成过程:
        # 1. torch.linspace(0, input_width - 1, input_width):
        #    函数参数: start=0, end=input_width-1, steps=input_width
        #    生成一个一维张量，内容为 [0., 1., 2., ..., input_width-1]。
        # 2. .repeat(input_height, 1):
        #    函数参数: (input_height, 1)
        #    将上述一维张量在第0维(行)重复input_height次，在第1维(列)重复1次，
        #    生成一个二维张量，形状为(input_height, input_width)，内容为每一行都是[0,1,2...]。
        # 3. .repeat(batch_size * len(self.anchors), 1, 1):
        #    将二维张量在新的第0维重复 batch_size * 3 次。
        # 4. .view(x.shape):
        #    将张量形状重塑为与x完全一致的 (batch_size, 3, input_height, input_width)。
        # 5. .type(FloatTensor):
        #    将张量的数据类型转换为指定的浮点数类型(CPU或GPU)。
        # 最终生成grid_x: 一个四维张量，其任意一个元素 grid_x[b, a, h, w] 的值都等于其最后一维的索引w。
        grid_x = torch.linspace(0 ,input_width -1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(self.anchors), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(self.anchors), 1, 1).view(y.shape).type(FloatTensor)
        # 最终生成grid_y: 一个四维张量，其任意一个元素 grid_y[b, a, h, w] 的值都等于其倒数第二维的索引h。

        # ----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   (batch_size, 3, H, W)
        # ----------------------------------------------------------#
        # 详细描述anchor_w和anchor_h的生成过程:
        # 1. FloatTensor(scaled_anchors):
        #    将scaled_anchors列表(例如[[w1,h1],[w2,h2],[w3,h3]])转换为一个形状为(3, 2)的张量。
        # 2. .index_select(1, LongTensor([0])):
        #    函数参数: dim=1 (沿列选择), index=LongTensor([0]) (选择第0列)。
        #    从(3, 2)的张量中选出第0列。
        #    最终生成anchor_w: 一个形状为(3, 1)的张量，内容为[[w1],[w2],[w3]]。
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        # 3. .index_select(1, LongTensor([1])):
        #    函数参数: dim=1, index=LongTensor([1]) (选择第1列)。
        #    最终生成anchor_h: 一个形状为(3, 1)的张量，内容为[[h1],[h2],[h3]]。
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        # 4. anchor_w.repeat(batch_size, 1):
        #    函数参数: (batch_size, 1)。
        #    将(3,1)的anchor_w张量在第0维重复batch_size次，生成形状为(batch_size * 3, 1)的张量。
        # 5. .repeat(1, 1, input_height * input_width):
        #    函数参数: (1, 1, H*W)。
        #    将(batch_size*3, 1)的二维张量提升为三维，并在新的第2维重复 H*W 次，
        #    生成形状为(batch_size*3, 1, H*W)的张量。
        # 6. .view(w.shape):
        #    将张量形状重塑为与网络预测值w完全一致的(batch_size, 3, H, W)。
        # 最终生成anchor_w: 一个四维张量，其在[b, a, h, w]位置的值，是第a个先验框的宽度。anchor_h同理。
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   利用预测结果对先验框进行解码
        #   首先调整先验框的中心，从左上角到中心
        #   然后得到最终预测框的中心和宽高
        #----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                            conf.view(batch_size, -1, 1),
                            prediction_cls.view(batch_size, -1, self.num_classes)), -1)

        # 一个形状为(B, N, 4 + 1 + num_classes)的张量。对于批次中的每一张图，它都包含N个预测结果，每个结果都是一个长度为5 + 类别数
        # 的向量，内容是[归一化x, 归一化y, 归一化w, 归一化h, 置信度, 类别1的概率, 类别2的概率, ...]。
        return output

