import torch
from torch import optim
from torch.nn import Module, Sequential, Conv2d
from torch.nn import  MaxPool2d, LeakyReLU, Flatten, Linear, Dropout, MSELoss
from Model.YOLO_v1.utils.Intersection_over_Union import intersection_over_union


class YOLOv1(Module):
    def __init__(self):
        super().__init__()
        self.YOLO = Sequential(
            Conv2d(3, 64, 7, 2, 3),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(64, 192, 3, 1, 1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(192, 128, 1),
            LeakyReLU(0.1),
            Conv2d(128, 256, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 256, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 512, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(1024, 512, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 512, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 2, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Flatten(),
            Linear(50176, 4096),
            LeakyReLU(0.1),
            Dropout(0.5),
            Linear(4096, 1470),
        )

    def forward(self, x):
        return self.YOLO(x)


"""
我们将创建一个名为 YoloLoss 的类，它继承自 PyTorch 的 nn.Module。这个类的核心是它的 forward 方法，该方法接收模型的预测值（predictions）和真实标签（target），并计算出它们之间的总损失。

整个计算过程可以分解为以下五大步骤：

步骤 0：初始化和准备

    在 YoloLoss 类的 __init__ 方法中，我们需要做几件事：
    
    存储超参数，如网格尺寸S、每个单元格预测的边界框数B、类别数C，以及两个重要的权重 lambda_coord 和 lambda_no_object。
    
    初始化一个均方差损失函数 nn.MSELoss。根据YOLOv1论文，他们使用的是各项误差的平方和，而不是均值。因此，我们在初始化时需要设置 reduction="sum"。

步骤 1：重塑（Reshape）预测张量

神经网络的输出通常是一个扁平的一维向量（例如，对于一个批次，形状是 (N, 1470)）。为了方便地按网格单元进行操作，forward 方法的第一步就是将这个扁平的张量重塑成具有空间结构的形状：(N, S, S, C + B*5)，即 (N, 7, 7, 30)。

步骤 2：区分“有物体”和“无物体”的区域

    我们需要知道在每个网格单元中，哪个预测框应该为物体负责，以及哪些单元格根本没有物体。
    确定“负责”的预测框：
        对于每个包含物体的网格单元，模型会预测 B 个（比如2个）边界框。我们不能让两个框都去拟合这一个物体。
        我们需要用之前写好的 intersection_over_union 函数，分别计算这2个预测框与真实框的IoU值。
        IoU值更高的那个预测框，就被指定为“负责人”，它将参与坐标损失和有物体时的置信度损失计算。另一个预测框则不参与。
    创建“存在物体”的掩码（Mask）：
        我们需要一个标记，来告诉我们 7x7 个网格单元中，哪些是真实存在物体的。
        这个信息直接从 target 张量中获取。按照约定，target 张量中代表置信度的那个位置，如果值为1，就表示该单元格有物体。我们将这个信息提取出来，形成一个形状为 (N, S, S, 1) 的掩码，我们称之为 exist。

步骤 3：分部计算损失

    现在我们有了“负责人”和“掩码”这两个关键工具，就可以逐一计算公式中的五个部分了。
    定位损失 (Localization Loss)：
        坐标损失 (x, y)：只在 exist 为1的单元格计算。取出“负责”的那个预测框的 x, y 和真实框的 x, y，计算它们的和方差。
        宽高损失 (w, h)：同样只在 exist 为1的单元格计算。取出“负责”的预测框的 w, h 和真实框的 w, h。重要：在计算损失前，要对所有的宽和高取平方根，以实现论文中对小物体更敏感的设计。
        将坐标损失和宽高损失相加，再乘以权重。
    
    置信度损失 (Confidence Loss)：
        有物体时 (Object Loss)：只在 exist 为1的单元格计算。取出“负责”的那个预测框的置信度，与真实置信度（在这里，真实框的置信度就是1）计算和方差。
        无物体时 (No-Object Loss)：在 exist 为0的单元格计算。对于这些单元格，两个预测框的置信度都应该被惩罚，让它们趋近于0。计算这两个预测置信度与0之间的和方差。将这部分损失乘以权重。
        
    分类损失 (Classification Loss)：
        只在 exist 为1的单元格计算。
        取出该单元格预测的类别概率向量（长度为 C），与真实的one-hot类别向量计算和方差。

步骤 4：整合总损失

    最后一步，将上面计算出的所有损失部分（定位损失、有物体置信度损失、无物体置信度损失、分类损失）全部加起来，得到一个最终的标量（单个数值）。这个值就是整个批次的总损失，PyTorch将用它来进行反向传播。
"""


class YoloLoss(Module):
    def __init__(self, s = 7, b=2, c=20, lambda_coord=5, lambda_no_object=0.5):
        """
            初始化YOLOv1损失函数.

            Args:
                s (int): 网格尺寸 (通常为7).
                b(int): 每个网格单元预测的边界框数量 (通常为2).
                c (int): 类别数 (PASCAL VOC为20).
                lambda_coord (float): 定位损失的权重.
                lambda_no_object (float): 不包含物体的置信度损失的权重.
        """
        super().__init__()
        self.YoloMse = MSELoss(reduction= "sum")
        self.S = s
        self.B = b
        self.C = c
        self.lambda_coord = lambda_coord
        self.lambda_no_object = lambda_no_object


    def forward(self, prediction, target):
        """
            计算损失.

            Args:
                prediction (torch.Tensor): 模型的输出张量, 形状为 (N, S*S*(C+B*5)).
                target (torch.Tensor): 真实的标签张量, 形状为 (N, S, S, C+5+5).
                    target每个单元格只包含一个真实框的信息(这是v1的假设),因此c+5就是c,conf,x,y,w,h,0,0,0,0,0

            Returns:
                total_loss:损失和
        """

        # --- 步骤 1: 重塑预测张量 ---
        # 将 (N, 1470) -> (N, 7, 7, 30)
        prediction = prediction.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # predictions (N, 1470) -> (N, 7, 7, 30)

        # --- 步骤 2: 确定“负责”检测的预测框 ---
        # prediction张量的格式: [C1...C20, conf1, x1,y1,w1,h1, conf2,x2,y2,w2,h2] (共30)
        # target张量的格式:   [C1...C20, conf,  x,y,w,h] (共25)
        b1_iou = intersection_over_union(prediction[..., 21:25], target[..., 21:25])
        b2_iou = intersection_over_union(prediction[..., 26:30], target[..., 21:25])
        # 把这2个预测框想象成你的工具箱里有两把扳手；面对一个待检测的物体,网络会同时用这两个预测框去“尝试”
        # 但在训练时,只有与真实物体交并比(IoU)更高的那个“优胜”预测框才会被用来计算坐标损失并进行学习.
        # 这种“优胜劣汰”的反馈机制会迫使这两个预测框在训练中逐渐演化成各自的“专家”,提高模型整体的预测精度.
        iou = torch.cat([b1_iou.unsqueeze(0), b2_iou.unsqueeze(0)], dim=0)
        # .unsqueeze(dim)会在张量的指定维度(dim)上增加一个大小为1的新维度
        # torch.cat(tensors, dim)会将一个张量列表里的所有张量,沿着指定的维度(dim)进行拼接(Concatenate).
        # 拼接后的最终结果 iou 张量的形状就变成了 (2, N, 7, 7, 1)

        iou_max, best_idx = torch.max(iou, dim=0)
        # 取出最大值及其最大值位置
        # torch.max(input, dim)input: 我们要操作的张量.dim: 这是最重要的参数,指定沿着哪个维度进行比较.
        # dim=0,让 torch.max 沿着 iou 的第一个维度(也就是那个长度为2的维度)进行操作.
        # torch.max(iou, dim=0) 返回两个张量的元组(Tuple),赋值给iou_max和best_idx,形状均为(N, S, S, 1)
        # 如B = torch.tensor             ([[[1, 5, 2],[4, 3, 6]],
        #   形状(2,2,3)                    [[7, 2, 8],[3, 9, 1]]])
        # B_max, best_idx = torch.max(B, dim=0)
        # B_max =                    tensor([[7, 5, 8],[4, 9, 6]]) 形状(2,3)
        # best_idx =                 tensor([[1, 0, 1],[0, 1, 0]]) 形状(2,3)

        # --- 步骤 3: 创建“存在物体”的掩码 (I_obj_i) ---
        # target中第21个元素(索引20)是置信度,1表示有物体,0表示没有
        # exists_box 的形状为 (N, S, S, 1)
        exist = target[..., 20].unsqueeze(3)
        # “切片”索引box_label[..., 0:1]返回(N, S, S, 1),“整数”索引target[..., 20]返回(N, S, S)(降维)
        # (N, S, S, 25)可以理解为S*S个区域,每个区域存储25个值
        # “整数”索引target[..., 20]后,取出了最后一个维度存储的25个值中的第20个(即置信度),返回的就是这个值的信息
        # target[..., 20]提取出来的这个 (N, S, S) 张量,其内容全部由 1 和 0 组成()
        # .unsqueeze(3)在第三个维度后加一个新维度,使其形状变为(N, S, S, 1)

        # --- 步骤 4.1: 计算定位损失 (Box Coordinate Loss) ---
        # 只在有物体的单元格,对“负责”的那个预测框计算损失
        # 选择出“负责”的预测框
        box_pred = (exist *
                    (best_idx * prediction[..., 26:30] +
                     (1 - best_idx) * prediction[..., 21:25]))
        # exist为真正存在标记的位置,best_idx的值为1则表示iou中1号位置有预测值,为0则表示iou中0号位置有预测值
        # 形状为(N, S, S, 4),在有物体且是负责人的位置,它保留了预测的[x, y, w, h].在所有其他位置,它的值都是 [0, 0, 0, 0].

        box_target = (exist * target[..., 21:25])
        # 掩码exist乘以实际位置,得到形状为(N, S, S, 4)的张量,在需要答案的位置存在真实的[x, y, w, h]

        # 我们需要分别对box_predictions和box_targets这两个张量中的宽和高部分（即第2和第3个索引位置）进行开方操作.
        # 对预测值开方时,为了防止预测出的宽高为负数而导致数学错误（负数不能开方）,我们要先取绝对值再开方,最后再把原始的符号乘回去.
        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6))
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        # 定位损失
        box_coordinate_loss = self.YoloMse(
            torch.flatten(box_pred, end_dim = -2),
            torch.flatten(box_target, end_dim = -2),
        )
        # [x, y, w, h]被一起送入函数计算,同时计算中心点误差和宽高误差
        # torch.flatten()函数提供了两个可选参数start_dim和end_dim.
        # start_dim: 从哪个维度开始展平（默认为0，即第一个维度）.end_dim: 在哪个维度结束展平（默认为 - 1，即最后一个维度）。
        # (2, 7, 7, 4) -->  (98, 4)
        # flatten非必须

        # --- 步骤 4.2: 计算置信度损失 (Confidence Loss) ---
        # 4.2.1: 有物体时 (Object Loss)
        # 取出“负责”的预测框的置信度
        pre_confidence = (best_idx * prediction[..., 25:26] + (1 - best_idx) * prediction[..., 20:21])
        object_loss = self.YoloMse(
            torch.flatten(exist * pre_confidence),
            torch.flatten(exist * target[..., 20:21])
        )

        # 4.2.2: 无物体时 (No Object Loss)
        # 在不存在物体的单元格，两个预测框的置信度都应该被惩罚
        b1_no_object_loss = self.YoloMse(
            torch.flatten((1 - exist) * prediction[..., 20:21], start_dim = 1),
            torch.flatten((1 - exist) * target[..., 20:21], start_dim = 1)
        )
        b2_no_object_loss = self.YoloMse(
            torch.flatten((1 - exist) * prediction[..., 25:26], start_dim =1),
            torch.flatten((1 - exist) * target[..., 20:21], start_dim =1)
        )
        no_object_loss = b1_no_object_loss + b2_no_object_loss

        # --- 步骤 4.3: 计算分类损失 (Classification Loss) ---
        # 只在有物体的单元格计算分类损失
        classification_loss = self.YoloMse(
            torch.flatten(exist * prediction[..., 0:20], end_dim = -2),
            torch.flatten(exist * target[..., 0:20], end_dim = -2)
        )

        # --- 步骤 5: 整合所有损失 ---
        # 应用权重系数
        total_loss = (self.lambda_coord * box_coordinate_loss +
                      object_loss +
                      self.lambda_no_object * no_object_loss +
                      classification_loss)

        return total_loss


if __name__ == '__main__':
    # 1. 设置超参数和设备
    LEARNING_RATE = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16  # 举例
    NUM_EPOCHS = 100
    # ... 其他参数
    # 2. 初始化模型、损失函数和优化器
    # 实例化模型，并将其移动到GPU（如果可用）
    yolo = YOLOv1().to(DEVICE)

    # 实例化损失函数
    loss_fn = YoloLoss()

    # 实例化优化器，将模型的参数传递给它
    optimizer = optim.Adam(
        yolo.parameters(), lr=LEARNING_RATE,
    )

    print("组件组装完毕！")

    print("--- 正在测试 YoloV1 前向传播 ---")
    try:
        # 创建一个虚拟的输入张量
        # 形状: (批次大小, 通道数, 高, 宽)
        dummy_input = torch.randn(4, 3, 448, 448)

        # 4. 将输入送入模型进行一次前向传播
        output = yolo(dummy_input)
        output = output.view(-1, 7, 7, 30)

        # 5. 打印输出的形状以进行验证
        print(f"\n输入尺寸: {dummy_input.shape}")
        print(f"输出尺寸: {output.shape}")

        # 根据YoloV1的结构，最终输出的特征向量应该是(4, 7, 7, 30)维
        # 所以期望的输出尺寸是 (4, 7, 7, 30)
        assert output.shape == (4, 7, 7, 30), "输出尺寸不匹配！"
        print("\n前向传播验证通过！")

    except Exception as e:
        print(f"\n前向传播测试失败，错误信息: {e}")
        print("请检查您的 模型 实现是否有误。")


    def test_loss_function_ideal_case():
        """
        单元测试函数，专门用于验证YoloLoss函数在理想情况下的表现。
        理想情况：构造一个完美的prediction，使其与target完全匹配。
        预期结果：计算出的总损失应为0。
        """
        print("\n--- 开始进行损失函数单元测试（理想情况） ---")

        # 1. 初始化损失函数
        loss_fn = YoloLoss(s=7, b=2, c=20)

        # 2. 创建一个基准 target 张量
        # 我们假设批次大小为1 (N=1)，在 (3, 3) 的网格单元有一个物体。
        # target 格式: [类别(20), 坐标(4), 置信度(1)]
        target = torch.zeros(1, 7, 7, 25)

        # 设置类别 (假设是第2类，索引为1)
        target[0, 3, 3, 1] = 1
        # 设置坐标 (x,y,w,h)
        target[0, 3, 3, 20:24] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        # 设置置信度 (有物体，为1)
        target[0, 3, 3, 24] = 1

        print("基准 target 已创建。")

        # 3. 根据 target 构建一个完美的 prediction 张量
        # prediction 格式: [类别(20), 坐标1(4), 置信度1(1), 坐标2(4), 置信度2(1)]
        prediction = torch.zeros(1, 7, 7, 30)

        # a. 对于有物体的单元格 (3, 3):
        #   - 复制类别信息
        prediction[0, 3, 3, 0:20] = target[0, 3, 3, 0:20]
        #   - 让第一个预测框 (box1) 完全等于 target 的 box
        prediction[0, 3, 3, 20:24] = target[0, 3, 3, 20:24]
        #   - 让第一个预测框的置信度 (conf1) 等于 target 的置信度 (即1)
        #     由于IoU(box1, target_box)会是1，而IoU(box2, target_box)是0，
        #     所以box1会被选为“负责”的预测框。
        prediction[0, 3, 3, 24] = target[0, 3, 3, 24]

        # b. 对于所有没有物体的单元格:
        #   - prediction 张量已经用0初始化，所以所有位置的两个置信度(conf1, conf2)都为0。
        #   - 这正好满足了“无物体时置信度损失为0”的条件。

        print("完美匹配的 prediction 已根据 target 创建。")

        # 4. 计算损失
        # 将 prediction 张量从 (N,S,S,30) 展平为 (N, S*S*30) 以模拟模型输出
        prediction_flat = torch.flatten(prediction, start_dim=1)

        total_loss = loss_fn(prediction_flat, target)

        # 5. 验证结果
        print(f"\n计算出的总损失值: {total_loss.item()}")

        # 使用断言来自动检查结果是否正确
        # 由于浮点数计算可能存在微小误差，我们检查损失是否小于一个很小的值
        assert total_loss.item() < 1e-6, "测试失败！理想情况下的损失不为0！"

        print("测试通过！在理想情况下，损失函数表现符合预期，返回值为0。")
        print("--- 损失函数单元测试结束 ---")

    def sanity_check(model, loss_fn, optimizer):
        print("--- 开始进行单元测试 ---")

        # 将模型设置为训练模式
        model.train()

        # 1. 创建虚拟输入数据
        # 创建一个批次大小为4的虚拟图片输入
        # 形状: (N, C, H, W) -> (4, 3, 448, 448)
        dummy_images = torch.randn(4, 3, 448, 448).to(DEVICE)

        # 2. 创建虚拟标签数据 (这是最关键的一步)
        # 形状: (N, S, S, C+5) -> (4, 7, 7, 25)
        dummy_target = torch.zeros(4, 7, 7, 25).to(DEVICE)

        # 我们在第一张图片的 (3, 3) 网格单元放置一个物体
        # 类别是第2类 (索引为1)
        dummy_target[0, 3, 3, 1] = 1
        dummy_target[0, 3, 3, 20] = 1
        dummy_target[0, 3, 3, 21:25] = torch.tensor(
            [0.5, 0.5, 0.2, 0.2]  # 归一化的 [x, y, w, h]
        )
        # 其余图片是背景，所以target张量保持全零

        print("虚拟数据创建完毕。")

        # 3. 模拟几次训练迭代
        for i in range(10):  # 迭代10次观察损失变化
            # 前向传播
            predictions = model(dummy_images)

            # 计算损失
            loss = loss_fn(predictions, dummy_target)

            # 反向传播和优化
            optimizer.zero_grad()   # 清空上一轮的梯度
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新权重

            print(f"迭代 {i + 1}: 损失值 = {loss.item()}")

        print("--- 单元测试结束 ---")
        print("请检查损失值是否呈下降趋势。")

    print("--- 正在运行 YoloV1 单元测试 ---")

    try:
        # sanity_check(yolo, loss_fn, optimizer)
        # 在最后调用新的验证函数
        test_loss_function_ideal_case()

    except Exception as e:
        print(f"\nYoloV1 单元测试失败，错误信息: {e}")
        print("请检查您的 YoloV1 单元测试是否有误。")




