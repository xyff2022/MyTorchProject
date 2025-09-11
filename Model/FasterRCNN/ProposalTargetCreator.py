import torch

from Model.FasterRCNN.AnchorTargetCreator import calculate_iou, encode_boxes


class ProposalTargetCreator:
    def __init__(self,
                 n_samples=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0):
        """
                初始化提议目标生成器。

                Args:
                    n_samples (int): 每个 mini-batch 中用于训练检测头的 RoI (候选区域) 总数。
                    pos_ratio (float): 在一个 mini-batch 中，正样本 (前景) 所占的比例。
                    pos_iou_thresh (float): IoU 大于此阈值的 RoI 被视为正样本。
                    neg_iou_thresh_hi (float):,neg_iou_thresh_lo (float)
                        : neg_iou_thresh_lo=<IoU < neg_iou_thresh_hi 范围内的 RoI 被视为负样本。
        """
        self.n_samples = n_samples
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, proposals, bboxes, labels):
        """
                在训练阶段为候选区域分配目标（多分类标签和回归偏移量）。
                为了保证模型在训练时的高效性，ProposalTargetCreator会精选n_samples个正负样本均衡的“习题集”。
                它会刻意保证这128个习题里有32个是正样本（各种物体），96个是负样本（典型的背景）。
                通过这种方式，检测头每一轮学习都能接触到高质量、有代表性的正负样本，使得训练过程又快又稳定。
                bboxes和labels是截取的target[n,1+4],其中n是标注个数,1是类别,4是候选框位置
                proposals是ProposalCreator 输出的候选区域，我们在训练时采用其中最高质量的，而预测时这采用全部

                Args:
                    proposals (torch.Tensor): ProposalCreator 输出的候选区域，形状 [num_proposals, 4]。
                    bboxes (torch.Tensor): 真实物体框，形状 [num_bboxes, 4]。
                    labels (torch.Tensor): 真实物体框对应的类别标签，形状 [num_bboxes]。
                                           (注意：背景标签为0，前景标签从1开始)。

                Returns:
                    tuple: 包含三个张量:
                        - roi (torch.Tensor): 采样后的 RoI，形状 [n_samples, 4]。
                        - roi_locs (torch.Tensor): 采样后 RoI 的回归目标，形状 [n_samples, 4]。
                        - roi_labels (torch.Tensor): 采样后 RoI 的类别标签，形状 [n_samples]。
        """

        # *** 新增代码开始: 处理没有真实物体框的情况 ***
        if bboxes.numel() == 0:
            # 如果没有真实物体，所有提议都应被视为负样本（背景）
            # 我们从中随机采样 self.n_samples 个作为训练样本
            num_proposals = proposals.shape[0]
            if num_proposals > self.n_samples:
                indices = torch.randperm(num_proposals, device=proposals.device)[:self.n_samples]
                sampled_rois = proposals[indices]
            else:
                sampled_rois = proposals

            # 为这些负样本创建目标
            # 标签全部为0（背景），回归目标全部为0
            gt_roi_labels = torch.zeros(sampled_rois.shape[0], dtype=torch.long, device=proposals.device)
            gt_roi_locs = torch.zeros_like(sampled_rois)
            return sampled_rois, gt_roi_locs, gt_roi_labels
        # *** 新增代码结束 ***

        # --- 步骤 1: 合并候选区域与真实物体框 ---
        # 真实物体框是 "完美" 的候选区域，将它们加入训练池
        candidate = torch.cat([proposals, bboxes], dim = 0)

        # --- 步骤 2: 计算 IoU 并分配初步标签 ---
        # iou 形状: [num_candidate, num_bboxes]
        iou = calculate_iou(candidate, bboxes)

        # 找出每个 candidate 对应的最大 IoU 值和 max_iou_index 索引
        # max_iou: [num_candidate], max_iou_index: [num_candidate]
        max_iou, max_iou_index = iou.max(dim = 1)
        # max_iou_index可以理解为一个有num_candidate个元素的数组，其中的值代表这个与这个元素iou值最大的元素位置

        # 根据最大 IoU 分配类别标签
        # 关键: 我们将真实标签 +1，因为标签 0 要留给背景类
        candidate_labels = labels[max_iou_index] + 1
        # labels[max_iou_index]的意思是生成一个和max_iou_index一样的张量，所有物体标签从1开始，将0留给了“背景”类别

        # --- 步骤 3: 分配负样本 (背景) ---
        # 将 IoU 在 [neg_iou_thresh_lo, neg_iou_thresh_hi) 范围内的 roi 设为负样本 (背景, 标签为0)
        neg_index = torch.where((max_iou>=self.neg_iou_thresh_lo)&(max_iou<self.neg_iou_thresh_hi))[0]
        candidate_labels[neg_index] = 0

        # --- 步骤 4: 确认正样本 ---
        # 将 IoU 大于 pos_iou_thresh 的 roi 最终确认为正样本
        pos_index = torch.where(max_iou >= self.pos_iou_thresh)[0]
        # 这一步会重新确认标签，避免一些高IoU区域被错误地划为负样本
        candidate_labels[pos_index] = labels[max_iou_index[pos_index]] + 1
        # 这一步参考candidate_labels = labels[max_iou_index] + 1

        # --- 步骤 5: 对正负样本进行采样 ---
        # 1. 计算希望采样的正样本数量上限
        pos_number = int(self.n_samples * self.pos_ratio)

        # 2. 找出当前所有正样本的索引 (标签 > 0)
        final_pos_index = torch.where(candidate_labels > 0)[0]

        # 3. 如果正样本数量超过上限，随机丢弃
        if len(final_pos_index) > pos_number:
            drop_index = final_pos_index[torch.randperm(len(final_pos_index))[:len(final_pos_index) - pos_number]]
            # 注意: 这里不是将标签设为-1，而是直接不把它们选入最终的样本
            final_pos_index = final_pos_index[torch.isin(final_pos_index, drop_index, invert = True)]

        # 4. 计算希望采样的负样本数量
        neg_number = int(self.n_samples - len(final_pos_index))

        # 5. 找出当前所有负样本的索引 (标签 == 0)
        final_neg_index = torch.where(candidate_labels == 0)[0]

        # 6. 如果负样本数量超过需求，随机丢弃
        if len(final_neg_index) > neg_number:
            drop_index = final_neg_index[torch.randperm(len(final_neg_index))[:len(final_neg_index) - neg_number]]
            # 注意: 这里不是将标签设为-1，而是直接不把它们选入最终的样本
            final_neg_index = final_neg_index[torch.isin(final_neg_index, drop_index, invert = True)]

        # 7. 合并最终被选中的正负样本索引
        index = torch.cat([final_pos_index,final_neg_index], dim=0)
        # 长度为128的一维向量，存储candidate的正负样本索引

        # --- 步骤 6: 生成回归目标并返回最终结果 ---
        # 1. 根据最终索引，获取采样后的 RoI 及其标签
        roi = candidate[index]   # -> [n_samples或index,4]
        roi_label = candidate_labels[index]

        # 2. 初始化回归目标张量
        roi_locs = torch.zeros_like(roi)

        # 3. 找出采样后的样本中，哪些是正样本
        pos_roi = torch.where(roi_label > 0)

        # 4. 获取正样本 RoI 匹配的真实物体框
        #    首先获取正样本在原始 candidate 列表中的索引
        pos_candidate_index = index[pos_roi]
        #    然后用这些原始索引去 max_iou_index 中查找匹配的 bboxes 索引
        bboxes_index = max_iou_index[pos_candidate_index]
        #    最后获取这些 bbox 的坐标
        bboxes_locs = bboxes[bboxes_index]

        # 5. 获取正样本 RoI 的坐标
        pos_roi_locs = roi[pos_roi]

        # 6. 为正样本计算回归目标
        roi_locs[pos_roi] = encode_boxes(pos_roi_locs, bboxes_locs)


        return roi, roi_locs, roi_label




# --- 用于测试本模块完整功能的脚本 ---
if __name__ == '__main__':
    print("--- ProposalTargetCreator 完整功能测试 ---")

    # 1. 实例化目标生成器
    proposal_target_creator = ProposalTargetCreator()
    print(f"测试参数: n_samples={proposal_target_creator.n_samples}, pos_ratio={proposal_target_creator.pos_ratio}, pos_iou_thresh={proposal_target_creator.pos_iou_thresh}")

    # 2. 创建假的输入数据
    # 假设 RPN 输出了 500 个候选区域
    dummy_proposals = torch.randint(0, 100, (500, 4)).float()
    dummy_proposals[:, 2:] += dummy_proposals[:, :2] + 20 # 保证 xmax > xmin

    # 假设图片中有2个真实物体: 一个猫 (label=0), 一个狗 (label=1)
    dummy_bboxes = torch.tensor([
        [10, 10, 50, 50],   # 猫
        [60, 60, 90, 90],   # 狗
    ], dtype=torch.float32)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long)

    # 为了精确验证，我们手动加入一个与"猫"高度重叠的 proposal
    high_iou_proposal = torch.tensor([[12, 12, 48, 48]], dtype=torch.float32)
    dummy_proposals = torch.cat([high_iou_proposal, dummy_proposals], dim=0)

    # 3. 调用 __call__ 方法
    sample_rois, gt_roi_locs, gt_roi_labels = proposal_target_creator(
        dummy_proposals, dummy_bboxes, dummy_labels
    )

    # 4. 验证输出的形状
    print("\n--- 1. 形状验证 ---")
    assert sample_rois.shape == (proposal_target_creator.n_samples, 4), f"采样RoI形状错误: 期望 ({proposal_target_creator.n_samples}, 4), 得到 {sample_rois.shape}"
    assert gt_roi_locs.shape == (proposal_target_creator.n_samples, 4), f"回归目标形状错误: 期望 ({proposal_target_creator.n_samples}, 4), 得到 {gt_roi_locs.shape}"
    assert gt_roi_labels.shape == (proposal_target_creator.n_samples,), f"标签形状错误: 期望 ({proposal_target_creator.n_samples},), 得到 {gt_roi_labels.shape}"
    print("形状验证通过！")

    # 5. 验证采样数量
    print("\n--- 2. 采样数量验证 ---")
    num_pos = torch.sum(gt_roi_labels > 0)
    num_neg = torch.sum(gt_roi_labels == 0)
    print(f"采样后，正样本数量: {num_pos}")
    print(f"采样后，负样本数量: {num_neg}")
    assert num_pos <= int(proposal_target_creator.n_samples * proposal_target_creator.pos_ratio)
    assert (num_pos + num_neg) <= proposal_target_creator.n_samples
    print("采样数量验证通过！")

    # 6. 验证标签分配
    print("\n--- 3. 标签分配验证 ---")
    # 我们加入的真实物体框本身，应该被采样为正样本
    # bboxes[0] (猫) 的标签应该是 0+1=1
    # bboxes[1] (狗) 的标签应该是 1+1=2
    assert 1 in gt_roi_labels, "猫 (标签1) 未被正确采样为正样本"
    assert 2 in gt_roi_labels, "狗 (标签2) 未被正确采样为正样本"
    print("真实物体框 (bboxes) 被正确采样并分配了标签！")

    # 7. 验证回归目标
    print("\n--- 4. 回归目标验证 ---")
    assert torch.all(gt_roi_locs[gt_roi_labels == 0] == 0), "负样本的回归目标必须为零"
    print("负样本的回归目标全部为零, 正确！")
    assert torch.any(gt_roi_locs[gt_roi_labels > 0] != 0), "正样本的回归目标不应全部为零"
    print("正样本的回归目标已正确计算！")

    print("\n\n--- 所有测试通过！ProposalTargetCreator 功能完整且正确。 ---")


