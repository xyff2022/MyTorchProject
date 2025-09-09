import torch

def encode_boxes(anchors, target):
    """
    将边界框编码为 RPN 所需的偏移量格式。
    这是 decode_boxes 的逆运算。

        Args:
            anchors (torch.Tensor): 所有正样本，形状 [N, 4]。
            target (torch.Tensor): 每个正样本对应的真实物体框或目标框，形状 [N, 4]。

        Returns:
            torch.Tensor: 编码后的偏移量 (tx, ty, tw, th)。
    """
    # 确保张量在同一设备上
    anchors = anchors.to(target.device)

    # --- 1. 将 anchors 转换为 (center_x, center_y, w, h) 格式 --
    anchors_w = anchors[..., 2] - anchors[..., 0]
    anchors_h = anchors[..., 3] - anchors[..., 1]
    anchors_center_x = anchors[..., 0] + 0.5 * anchors_w
    anchors_center_y = anchors[..., 1] + 0.5 * anchors_h

    # --- 2. 将 boxes (真实框) 转换为 (center_x, center_y, w, h) 格式 ---
    target_w = target[..., 2] - target[..., 0]
    target_h = target[..., 3] - target[..., 1]
    target_center_x = target[..., 0] + 0.5 * target_w
    target_center_y = target[..., 1] + 0.5 * target_h

    # --- 3. 应用编码公式 ---
    # 为防止除零，加上一个极小值 epsilon
    eps = torch.finfo(anchors_w.dtype).eps
    anchors_w = torch.clamp(anchors_w, min = eps)
    anchors_h = torch.clamp(anchors_h, min = eps)

    dx = (target_center_x - anchors_center_x) / anchors_w
    dy = (target_center_y - anchors_center_y) / anchors_h
    dw = torch.log(target_w / anchors_w)
    dh = torch.log(target_h / anchors_h)

    return torch.stack((dx, dy, dw, dh), dim = 1)





def calculate_iou(pre_anchors, target):
    """
    计算两组边界框之间的 IoU 矩阵。

        Args:
            pre_anchors (torch.Tensor): 预测的边界框，形状为 [N, 4]。
            target (torch.Tensor): 真实的边界框，形状为 [M, 4]。

        Returns:
            torch.Tensor: IoU 矩阵，形状为 [N, M]。
    """
    # 扩展维度以利用广播机制
    # pre_anchors: [N, 4] -> [N, 1, 4]
    # target: [M, 4] -> [1, M, 4]
    pre_anchors = pre_anchors.unsqueeze(1)
    target = target.unsqueeze(0)

    # 计算交集的坐标
    intersection_min = torch.max(pre_anchors[..., :2], target[..., :2])  #->[N,M,2(xmin,ymin)]
    intersection_max = torch.min(pre_anchors[..., 2:], target[..., 2:])  #->[N,M,2(xmax,ymax)]

    # 计算交集的宽和高，如果 non-positive 则设为0  ->[N,M,2]
    intersection_wh = (intersection_max - intersection_min).clamp(min = 0)

    # 计算交集面积  ->[N,M]
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    # 计算各自的面积
    pre_anchors_area = ((pre_anchors[..., 2] - pre_anchors[..., 0]) *
                        (pre_anchors[..., 3] - pre_anchors[..., 1]))   #-> [N, 1]
    target_area = ((target[..., 2] - target[..., 0]) *
                   (target[..., 3] - target[..., 1]))   #-> [1, M]

    # 计算并集面积
    union = pre_anchors_area + target_area - intersection

    # 计算 IoU，避免除以零
    iou = intersection / (union + 1e-9)  #-> [N, M]

    return iou


class AnchorTargetCreator:
    def __init__(self,
                 n_samples = 256,
                 pos_iou_thresh = 0.7,
                 neg_iou_thresh = 0.3,
                 pos_ratio = 0.5):
        """
                初始化锚点目标生成器。

                Args:
                    n_samples (int): 每个 mini-batch 中用于计算 RPN 损失的锚点总数。
                    pos_iou_thresh (float): IoU 大于此阈值的锚点被视为正样本（前景）。
                    neg_iou_thresh (float): IoU 小于此阈值的锚点被视为负样本（背景）。
                    pos_ratio (float): 在一个 mini-batch 中，正样本所占的比例。
        """

        self.n_samples = n_samples
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, anchors, target, image_size):
        """
                为锚点分配目标（标签和回归偏移量）。

                Args:
                    anchors (torch.Tensor): 所有锚点，形状 [num_anchors, 4]。
                    target (torch.Tensor): 真实物体框，形状 [num_target, 4]。
                    image_size (tuple): 图像尺寸 (height, width)。
                Returns:
                    final_labels(torch.Tensor): 锚点标签，-> [num_anchors]。
                    final_loc_targets(torch.Tensor): 回归偏移量，-> [num_anchors,4](dx,dy,dw,dh)。

        """
        # --- 步骤 7.3：计算 IoU ---
        # 1. 过滤掉超出图像边界的锚点 (可选但推荐)
        H, W =image_size
        inside = torch.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= W) &
            (anchors[:, 3] <= H)
        )[0]
        valid_anchors = anchors[inside]

        # 2. 初始化所有有效锚点的标签为 -1
        labels = torch.full((len(valid_anchors),), -1, dtype=torch.long)
        # len()返回第一个维度的长度

        if target.numel() == 0:
            labels.fill_(0)
        else:
            # 3. 计算有效锚点与所有真实物体框的 IoU 矩阵
            #    iou 形状: [valid_anchors, target]
            iou = calculate_iou(valid_anchors, target)

            # --- 步骤 7.4：为锚点分配标签 ---

            # 1. 对每个有效锚点，找出它与哪个真实框的 IoU 最大
            #    max_iou_index: 每个有效锚点与和它对应的最佳真实框的索引，形状 [valid_anchors]
            #    max_iou: 每个有效锚点对应的最大 IoU 值，形状 [valid_anchors]

            # max_iou_index = iou.argmax(dim = 1)
            # .argmax(): 这个函数的作用是返回最大值的索引(index)。它不关心最大值本身是多少，只关心它在哪个位置。
            # dim = 1: 这个参数是关键。它告诉argmax沿着维度1（也就是列的方向）进行操作。
            # 对于一个二维矩阵，dim = 1意味着对每一行进行操作，找出该行中最大值的列索引。
            # dim = 0，对行操作不保留行，dim = 1，对列操作不保留列

            # max_iou = iou.max(dim=1)[0]
            # .max(): 会同时返回最大值本身(values)和最大值的索引(indices)。
            # dim = 1意味着对每一行进行操作。当它作用于我们的iou矩阵时，会返回一个元组(values, indices)。
            # values: 包含了每一行的最大IoU值。
            # indices: 包含了每一行最大IoU值的列索引（这个结果和.argmax(dim=1)的结果完全一样）。
            max_iou, max_iou_index = iou.max(dim=1)
            # 形状均为[valid_anchors]

            # 2. 分配负样本 (标签 0)
            #    将与所有真实框的最大 IoU 都小于 neg_iou_thresh 的锚点设为背景
            labels[max_iou < self.neg_iou_thresh] = 0

            # 3. 分配正样本 (标签 1) - 规则 1
            #    对每个真实框，找出与它 IoU 最高的锚点，确保每个真实物体都有锚点匹配
            # 留意全是背景的图片
            labels[iou.argmax(dim=0)] = 1

            # 4. 分配正样本 (标签 1) - 规则 2
            #    将与任何一个真实框的 IoU 大于 pos_iou_thresh 的锚点设为前景
            labels[max_iou > self.pos_iou_thresh] = 1

        # --- 步骤 7.5：对正负样本进行采样 ---
        # 1. 计算我们希望采样的正样本数量上限
        pos_iou_number = int(self.n_samples * self.pos_ratio)

        # 2. 找出当前所有正样本的索引
        # torch.where(condition)查找并返回所有满足条件的元素的“坐标”或“索引”。
        # condition: 必须是一个布尔（Boolean）张量，里面由True和False组成。
        # 返回值: 它返回一个元组。元组的长度等于输入张量的维度,每个元素都是一个张量，对应每个维度上True元素的索引。
        # 如果x是一个张量，x == y输出一个大小和x一样的布尔张量，里面的每一个元素都是和y比教后的布尔结果
        pos_index = torch.where(labels ==1)[0]

        # 3. 如果正样本数量超过了上限，就随机“禁用”一些（将标签设回-1）
        # torch.randperm(n),生成一个包含从0到n-1的序列号的tensor
        # torch.randperm(len(pos_index))[:(len(pos_index) - pos_iou_number)]
        # 取出len(pos_index)个索引的前(len(pos_index) - pos_iou_number)个
        if len(pos_index) > pos_iou_number:
            drop_index = pos_index[torch.randperm(len(pos_index))[:(len(pos_index) - pos_iou_number)]]
            labels[drop_index] = -1

        # 4. 计算我们希望采样的负样本数量
        # torch.sum主要作用是计算一个张量（tensor）中所有元素的和。
        # torch.sum()的结果是一个标量张量（一个装有单个数字的“盒子”）。
        # 假设pos_iou_number为1（最小为1，值为最大的iou），那neg_iou_number就是127
        neg_iou_number = self.n_samples - int(torch.sum(labels ==1))

        # 5. 找出当前所有负样本的索引
        neg_index = torch.where(labels ==0)[0]

        # 6. 如果负样本数量超过了上限，就随机“禁用”一些
        if len(neg_index) > neg_iou_number:
            drop_index = neg_index[torch.randperm(len(neg_index))[:(len(neg_index) - neg_iou_number)]]
            labels[drop_index] = -1



        # --- 步骤 7.6：为正样本生成回归目标 ---
        # 1. 初始化一个全零的最终回归目标张量标签和回归偏移量
        final_labels = torch.full((len(anchors),), -1, dtype= torch.long)
        final_loc_targets = torch.zeros_like(anchors)

        # 2. 将计算出的有效锚点标签填充到最终标签张量中
        # valid_anchors = anchors[inside] inside 是连接valid_anchors和anchors的索引
        # 10个anchors,inside = [0,2,3,4,5]，labels = [-1,0,1,0,1]
        # valid_anchors = anchors[inside],valid_anchors形状[5,4],
        # max_iou_index[1,2,3,4,5],表明这几个valid_anchors所对应的真实物体编号，
        final_labels[inside] = labels
        # 意思是把labels这个表的5个值赋值给final_labels的这[0,2,3,4,5]这几个位置

        # 3. 找出最终被保留为正样本的锚点
        # finally_pos_index是一个一维张量，其长度为正样本的数量
        finally_pos_index = torch.where(labels ==1)[0]

        if len(finally_pos_index) > 0 and target.numel()> 0:
            # 4. 获取这些正样本锚点
            finally_pos = valid_anchors[finally_pos_index]
            # finally_pos_index -> [N_pos],N_pos为最终的正样本数量，valid_anchors -> [valid_anchors]
            # finally_pos -> [N_pos,4]

            # 5. 找出这些正样本锚点各自对应的真实物体框
            # max_iou_index 存储了每个锚点最匹配的真实框的索引
            # max_iou_index[finally_pos_index]取出每个正样本匹配的真实物体编号
            # pos_anchors[N_pos,4],是每个正样本锚点所匹配的真实物体框，即标准答案
            pos_anchors = target[max_iou_index[finally_pos_index]]

            # 6. 调用 encode_boxes 计算回归目标
            pos_loc_target = encode_boxes(finally_pos, pos_anchors)

            # 7. 将计算结果填充到 final_loc_targets 张量中
            # valid_anchors = anchors[inside] inside 是连接valid_anchors和anchors的索引
            # 10个anchors,inside = [0,2,3,4,5]，labels = [-1,0,1,0,1]
            # valid_anchors = anchors[inside],valid_anchors形状[5,4],
            # max_iou_index[1,2,3,4,5],表明这几个valid_anchors所对应的真实物体编号
            # finally_pos_index=[2,4],original_loc_targets=[3,5]
            original_loc_targets = inside[finally_pos_index]
            final_loc_targets[original_loc_targets] = pos_loc_target


        return final_labels,final_loc_targets


# --- 全面的单元测试套件 ---
if __name__ == '__main__':
    def test_normal_case(creator):
        print("--- 1. 测试正常情况 ---")
        image_size = (120, 120)
        dummy_bboxes = torch.tensor([[20, 20, 80, 80], [90, 90, 110, 110]], dtype=torch.float32)
        dummy_anchors = torch.tensor([
            [22, 22, 78, 78],  # 高IoU -> 正样本 (1)
            [0, 0, 15, 15],  # 低IoU -> 负样本 (0)
            [45, 45, 85, 85],  # 中等IoU -> 忽略 (-1)
            [110, 110, 130, 130],  # 边界外 -> 忽略 (-1)
        ], dtype=torch.float32)

        labels, locs = creator(dummy_anchors, dummy_bboxes, image_size)

        assert labels[0] == 1, "高IoU锚点应为正样本"
        assert labels[1] == 0, "低IoU锚点应为负样本"
        assert labels[2] == -1, "中等IoU锚点应被忽略"
        assert labels[3] == -1, "界外锚点应被忽略"
        assert torch.any(locs[0] != 0), "正样本应有回归目标"
        assert torch.all(locs[1:] == 0), "非正样本不应有回归目标"
        print("正常情况测试通过！")


    def test_no_gt_case(creator):
        print("\n--- 2. 测试无物体图片 (纯背景) ---")
        image_size = (100, 100)
        dummy_bboxes = torch.empty((0, 4), dtype=torch.float32)
        dummy_anchors = torch.tensor([[10, 10, 30, 30], [50, 50, 80, 80]], dtype=torch.float32)

        labels, locs = creator(dummy_anchors, dummy_bboxes, image_size)

        assert torch.all(labels == 0), "无物体时，所有有效锚点都应为负样本"
        assert torch.all(locs == 0), "无物体时，不应有任何回归目标"
        print("无物体图片测试通过！")


    def test_positive_sampling(creator):
        print("\n--- 3. 测试正样本过多时的下采样 ---")
        image_size = (100, 100)
        dummy_bboxes = torch.tensor([[10, 10, 90, 90]], dtype=torch.float32)
        # 创建大量与GT框高度重叠的锚点
        num_high_iou_anchors = 200
        high_iou_anchors = torch.tensor([[11, 11, 89, 89]], dtype=torch.float32).repeat(num_high_iou_anchors, 1)

        labels, _ = creator(high_iou_anchors, dummy_bboxes, image_size)

        num_pos_sampled = torch.sum(labels == 1)
        expected_num_pos = int(creator.n_samples * creator.pos_ratio)
        print(f"期望正样本数: {expected_num_pos}, 实际采样数: {num_pos_sampled}")
        assert num_pos_sampled == expected_num_pos, "正样本下采样数量不正确"
        print("正样本下采样测试通过！")


    def test_negative_sampling(creator):
        print("\n--- 4. 测试负样本过多时的下采样 ---")
        image_size = (500, 500)
        # 一个在角落的小物体框
        dummy_bboxes = torch.tensor([[480, 480, 495, 495]], dtype=torch.float32)
        # --- 修复：创建大量、确定在界内、且确定是负样本的锚点 ---
        # 在左上角创建一个网格的锚点，确保它们都离右下角的GT框很远
        num_neg_anchors = 400
        x = torch.linspace(10, 200, 20)
        y = torch.linspace(10, 200, 20)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        low_iou_anchors_tl = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        low_iou_anchors = torch.cat([low_iou_anchors_tl, low_iou_anchors_tl + 20], dim=1)

        labels, _ = creator(low_iou_anchors, dummy_bboxes, image_size)

        num_neg_sampled = torch.sum(labels == 0)
        num_pos_sampled = torch.sum(labels == 1)
        expected_num_neg = creator.n_samples - num_pos_sampled
        print(f"期望负样本数: {expected_num_neg}, 实际采样数: {num_neg_sampled}")
        assert num_neg_sampled == expected_num_neg, "负样本下采样数量不正确"
        print("负样本下采样测试通过！")


    def main():
        print("--- AnchorTargetCreator 全面功能测试 ---")
        # 使用默认参数实例化
        creator = AnchorTargetCreator()

        test_normal_case(creator)
        test_no_gt_case(creator)
        test_positive_sampling(creator)
        test_negative_sampling(creator)

        print("\n\n--- 所有测试用例通过！AnchorTargetCreator 功能健壮。 ---")


    main()
