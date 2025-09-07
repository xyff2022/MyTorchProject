import torch
from torch.nn.functional import cross_entropy

# ------------------------------------------------------------------
#                           RPN 损失
# ------------------------------------------------------------------

def rpn_cls_loss(pred_scores, labels):
    """
        计算 RPN 的分类损失 (前景/背景)。

        Args:
            pred_scores (torch.Tensor): RPN 输出的原始分类分数，形状 [N, 2]。
                                        (第0列是背景分，第1列是前景分)。
            labels (torch.Tensor): AnchorTargetCreator 输出的标签，形状 [N]。
                                      (1=前景, 0=背景, -1=忽略)。

        Returns:
            torch.Tensor: 计算出的交叉熵损失 (一个标量)。
    """
    # --- 步骤 1: 筛选有效样本 ---
    # 我们只对标签为 0 (背景) 或 1 (前景) 的锚点计算损失。
    valid_labels = torch.where(labels != -1)[0]

    # 如果没有有效样本，则分类损失为0
    if valid_labels.numel() == 0:
        return torch.tensor(0.0, device = pred_scores.device)

    # --- 步骤 2: 计算交叉熵损失 ---
    # F.cross_entropy 接收原始分数(logits)和类别索引作为输入。
    # 它会自动在内部应用 Softmax。
    return cross_entropy(pred_scores[valid_labels], labels[valid_labels])

def rpn_loc_loss(pred_locs, final_loc_targets, final_labels, sigma=1.0):
    """
        计算 RPN 的回归损失。

    """
    return smooth_l1_loss(pred_locs, final_loc_targets, final_labels)

# ------------------------------------------------------------------
#                         检测头 (ROI Head) 损失
# ------------------------------------------------------------------

def roi_cls_loss(pred_scores, labels):
    """
      计算检测头的分类损失 (多类别)。

      Args:
          pred_scores (torch.Tensor): 检测头输出的原始分类分数，形状 [N, C+1]。
                                      (N 是输入数据的数量, C 是物体类别数)。
          labels (torch.Tensor): ProposalTargetCreator 输出的标签，形状 [N]。
                                    (0=背景, 1=猫, 2=狗, ...)。

      Returns:
          torch.Tensor: 计算出的交叉熵损失 (一个标量)。
      """
    # --- 步骤 1: (可选) 检查输入是否为空 ---
    # ProposalTargetCreator 应该总是返回固定数量的样本，但做一个检查更稳健
    if labels.numel() == 0:
        return torch.tensor(0.0, device = pred_scores.device)

    # --- 步骤 2: 计算交叉熵损失 ---
    # gt_labels 已经很干净，没有需要忽略的 -1，可以直接使用
    return cross_entropy(pred_scores, labels)

def roi_loc_loss(pred_locs, roi_locs, roi_labels, sigma=1.0):
    """
        计算检测头的回归损失 (类别相关)。

        Args:
            pred_locs (torch.Tensor): 检测头输出的原始回归参数，形状 [N, C*4]。
                                  (N 是采样数量, C 是物体类别数)。
            roi_locs (torch.Tensor): ProposalTargetCreator 输出的回归目标，形状 [N, 4]。
            roi_labels (torch.Tensor): ProposalTargetCreator 输出的标签，形状 [N]。
                                  (0=背景, 1=猫, 2=狗, ...)。
            sigma (float): Smooth L1 Loss 的阈值参数。

        Returns:
            torch.Tensor: 计算出的平滑 L1 损失 (一个标量)。
    """
    # --- 步骤 1: 筛选正样本 ---
    # 回归损失只对正样本有意义(0是背景)且检查不为0
    valid_label = torch.where(roi_labels > 0)[0]
    if valid_label.numel() == 0:
        return torch.tensor(0.0, device = pred_locs.device)

    # --- 步骤 2: 挑选正样本的预测 ---
    # pos_labels 的值是类别索引，例如 [1, 2, 1] (猫, 狗, 猫)
    pos_labels = roi_labels[valid_label]

    # pos_pred_locs 的形状是 [num_pos, C*4]
    pos_pred_locs = pred_locs[valid_label]
    # 以pred_locs[128,84],roi_locs[128,4],roi_labels[128]为例
    # 我们要取出这128个中的第一个物体时，上面三个参数为pred_locs[0,4:8],roi_locs[0,4],roi_labels[0]
    # 1. 准备“行号”：即所有正样本的索引 [0, 1, ..., num_pos-1]
    pos_number = valid_label.numel()
    row_index = torch.arange(pos_number, device = pred_locs.device).view(-1,1)

    # 2. 准备“列号”：为每一行计算出需要提取的4个列的索引
    #    a. 计算起始列号
    start_col_index = (pos_labels - 1).long() * 4
    # 假设roi_labels=[2, 1, 2]，start_col_index = torch.tensor([4, 0, 4])
    #    b. 生成完整的4个列号
    col_index = start_col_index.view(-1, 1) + torch.arange(4, device=pred_locs.device)
    # .view(-1, 1)重塑start_col_index成一个二维的列向量[[4],[0],[4]](3*1)
    # torch.arange(4, device=pred_locs.device)创建一个简单的一维行向量 [0, 1, 2, 3] (1*4)
    # 加法操作 - 广播 3*1->3*4   [[4, 4, 4, 4],    1*4_>3*4   [[0, 1, 2, 3],       [[4, 5, 6, 7],
    #                           [0, 0, 0, 0],       +       [0, 1, 2, 3],  =      [0, 1, 2, 3],
    #                           [4, 4, 4, 4]]               [0, 1, 2, 3]]         [4, 5, 6, 7]]


    # 3. 使用高级索引提取数据
    final_pred_locs = pos_pred_locs[row_index, col_index]

    # --- 步骤 3: 调用 smooth_l1_loss ---
    return smooth_l1_loss(final_pred_locs, roi_locs[valid_label], pos_labels)







def smooth_l1_loss(pred_locs, bboxes, labels, sigma=1.0):
    """
    计算平滑 L1 损失 (Smooth L1 Loss)。

    Args:
        pred_locs (torch.Tensor): 预测的回归参数 (tx, ty, tw, th)，形状 [N, 4]。
        bboxes (torch.Tensor): 真实的回归参数，形状 [N, 4]。
        labels (torch.Tensor): 两个target生成函数生成的标签，形状 [N]。用于只计算正样本的损失。
        sigma (float): Smooth L1 Loss 的阈值参数。

    Returns:
        torch.Tensor: 计算出的平滑 L1 损失 (一个标量)。
    """

    # 请确保 sigma^2 不为零，避免除零错误
    sigma2 = sigma ** 2

    # --- 步骤 1: 筛选正样本 ---
    # 我们只关心那些被判定为前景 (positive) 的样本的回归准确度。
    # 我们写的是总smooth_l1_loss，rpn中前景标识为1，proposal中前景是从1到N的类别(0表示背景)
    sample = torch.where(labels > 0)[0]

    # 如果没有正样本，则回归损失为0
    # numel()返回张量中所有元素的总数（将所有维度的尺寸相乘）
    if sample.numel() == 0:
        loss = torch.tensor(0.0, device = pred_locs.device)

    # --- 步骤 2: 计算误差 ---
    difference = (pred_locs[sample] - bboxes[sample]).abs()

    # --- 步骤 3: 应用公式 ---
    smooth_l1 = torch.where(
        difference > 1/sigma2,
        difference -0.5 * sigma2,
        0.5 * sigma2 * difference * difference
    )

    # --- 步骤 4: 求和与归一化 ---
    # 损失值需要除以参与计算的样本数量，以获得平均损失。
    # 这里的 .numel() 是获取张量中元素的总数。
    loss = smooth_l1.sum() / labels.numel()

    return loss



# --- 用于测试本模块所有损失功能的脚本 ---
if __name__ == '__main__':
    print("--- 损失函数模块完整功能测试 ---")

    # --- 1. RPN 损失测试 ---
    print("\n--- 1. RPN损失函数测试 ---")
    # 假设有5个锚点，RPN的输出
    dummy_rpn_scores = torch.randn(5, 2)
    dummy_rpn_locs = torch.randn(5, 4)
    # 假设AnchorTargetCreator的输出
    dummy_gt_rpn_labels = torch.tensor([1, 0, -1, 1, 0])
    dummy_gt_rpn_locs = torch.randn(5, 4)

    # a. 测试 rpn_cls_loss
    rpn_loss_c = rpn_cls_loss(dummy_rpn_scores, dummy_gt_rpn_labels)
    # 手动计算以验证 (只计算标签不为-1的)
    valid_scores = dummy_rpn_scores[[0, 1, 3, 4]]
    valid_labels = dummy_gt_rpn_labels[[0, 1, 3, 4]]
    expected_rpn_loss_c = cross_entropy(valid_scores, valid_labels)
    assert torch.allclose(rpn_loss_c, expected_rpn_loss_c), "RPN 分类损失计算错误"
    print(f"RPN 分类损失: {rpn_loss_c.item():.4f} (验证通过)")

    # b. 测试 rpn_loc_loss
    rpn_loss_l = rpn_loc_loss(dummy_rpn_locs, dummy_gt_rpn_locs, dummy_gt_rpn_labels)
    # 手动计算以验证 (只计算标签为1的)
    pos_pred = dummy_rpn_locs[[0, 3]]
    pos_gt = dummy_gt_rpn_locs[[0, 3]]
    pos_labels = dummy_gt_rpn_labels[[0, 3]]
    expected_rpn_loss_l = smooth_l1_loss(pos_pred, pos_gt, pos_labels) * len(pos_labels) / len(dummy_gt_rpn_labels) # 重新调整归一化因子
    assert rpn_loss_l > 0, "RPN 回归损失应为正"
    print(f"RPN 回归损失: {rpn_loss_l.item():.4f} (验证通过)")


    # --- 2. 检测头损失测试 ---
    print("\n--- 2. 检测头(ROI Head)损失函数测试 ---")
    # 假设有5个采样后的RoI, 3个物体类别 (C=3)
    num_rois = 5
    num_classes = 3
    # 检测头的输出
    dummy_roi_scores = torch.randn(num_rois, num_classes + 1) # C+1, 包括背景
    dummy_roi_locs = torch.randn(num_rois, num_classes * 4) # C*4, 类别相关
    # ProposalTargetCreator的输出
    dummy_gt_roi_labels = torch.tensor([1, 0, 2, 3, 0]) # 0=背景, 1=猫, 2=狗, 3=车
    dummy_gt_roi_locs = torch.randn(num_rois, 4)

    # a. 测试 roi_cls_loss
    roi_loss_c = roi_cls_loss(dummy_roi_scores, dummy_gt_roi_labels)
    expected_roi_loss_c = cross_entropy(dummy_roi_scores, dummy_gt_roi_labels)
    assert torch.allclose(roi_loss_c, expected_roi_loss_c), "检测头分类损失计算错误"
    print(f"检测头分类损失: {roi_loss_c.item():.4f} (验证通过)")

    # b. 测试 roi_loc_loss
    roi_loss_l = roi_loc_loss(dummy_roi_locs, dummy_gt_roi_locs, dummy_gt_roi_labels)
    assert roi_loss_l > 0, "检测头回归损失应为正"
    # 验证负样本的回归损失是否为0
    dummy_gt_roi_labels_only_neg = torch.tensor([0, 0, 0, 0, 0])
    roi_loss_l_neg = roi_loc_loss(dummy_roi_locs, dummy_gt_roi_locs, dummy_gt_roi_labels_only_neg)
    assert roi_loss_l_neg == 0, "负样本的回归损失必须为0"
    print(f"检测头回归损失: {roi_loss_l.item():.4f} (验证通过)")
    print("负样本的回归损失为0, 正确！")

    print("\n\n--- 所有损失函数测试通过！---")
