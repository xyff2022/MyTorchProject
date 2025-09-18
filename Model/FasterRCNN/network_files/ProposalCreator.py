import torch
from torchvision.ops import nms


def decode_boxes(locs, anchors):
    """
    将 RPN 的偏移量解码到锚点上，得到预测框。
    Args:
        locs (torch.Tensor): RPN 输出的位置偏移，形状为 [N, 4]。
        anchors (torch.Tensor): 锚点坐标，形状为 [N, 4]，格式为 (xmin, ymin, xmax, ymax)。
    Returns:
        torch.Tensor: 解码后的预测框坐标。
    """
    # 计算锚点的宽度、高度和中心点坐标
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_center_x = anchors[:, 0] + 1/2 * anchor_w
    anchor_center_y = anchors[:, 1] + 1/2 * anchor_h

    # 提取网络预测的4个偏移量
    dx, dy, dw, dh = locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3]
    # 1. 计算中心点坐标
    pre_x = dx * anchor_w + anchor_center_x
    pre_y = dy * anchor_h + anchor_center_y
    # 2. 计算宽度和高度
    pre_w = anchor_w * torch.exp(dw)
    pre_h = anchor_h * torch.exp(dh)
    # 将计算后的中心点和宽高转换回 (xmin, ymin, xmax, ymax) 格式
    pre_box = torch.zeros_like(anchors)
    pre_box[:, 0] = pre_x - 1/2 * pre_w
    pre_box[:, 1] = pre_y - 1 / 2 * pre_h
    pre_box[:, 2] = pre_x + 1 / 2 * pre_w
    pre_box[:, 3] = pre_x + 1 / 2 * pre_h

    return pre_box




class ProposalCreator:
    def __init__(self,
                 mode,  # 'training' 或 'validation'
                 nms_iou_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_validation_pre_nms=6000,
                 n_validation_post_nms=300,
                 min_size=16):
        """
                初始化提议生成器。

                Args:
                    mode (str): 'training' 或 'validation' 模式。不同模式下筛选的提议数量不同。
                    nms_iou_thresh (float): NMS 的 IoU 阈值。
                    n_train_pre_nms (int): 训练时，NMS之前保留的框的数量。
                    n_train_post_nms (int): 训练时，NMS之后保留的框的数量。
                    n_validation_pre_nms (int): 验证时，NMS之前保留的框的数量。
                    n_validation_post_nms (int): 验证时，NMS之后保留的框的数量。
                    min_size (int): 允许的边界框最小尺寸。
        """

        self.mode = mode
        self.nms_iou_thresh = nms_iou_thresh
        self.min_size = min_size
        # 根据模式设置不同的提议数量
        if self.mode == 'training':
            self.n_pre_nms = n_train_pre_nms
            self.n_post_nms = n_train_post_nms
        else:
            self.n_pre_nms = n_validation_pre_nms
            self.n_post_nms = n_validation_post_nms

    def __call__(self, rpn_locs, rpn_scores, anchors, img_size):
        """
        将RPN输出转换为提议。
        Args:
            rpn_locs (torch.Tensor): RPN 输出的位置偏移，形状 [N, 4]。
            rpn_scores (torch.Tensor): RPN 输出的前景分，形状 [N, 1]。
            anchors (torch.Tensor): 锚点坐标，形状 [N, 4]。
            img_size (tuple): (height, width) of the image.
        """

        # 1. 计算回归后的边界框
        # 在最开始的图像上看看有哪些提议越界了
        pre_boxes = decode_boxes(rpn_locs, anchors)


        # 2. 裁剪边界框到图像范围内
        H, W =img_size
        pre_boxes[:, 0].clamp_(min = 0, max = W)
        pre_boxes[:, 1].clamp_(min = 0, max = H)
        pre_boxes[:, 2].clamp_(min = 0, max = W)
        pre_boxes[:, 3].clamp_(min = 0, max = H)

        # 3. 过滤掉尺寸过小的框
        min_w = pre_boxes[:, 2] - pre_boxes[:, 0]
        min_h = pre_boxes[:, 3] - pre_boxes[:, 1]
        keep_small = torch.where((min_w >= self.min_size) & (min_h >= self.min_size))[0]
        boxes = pre_boxes[keep_small, :]
        scores = rpn_scores[keep_small]

        # 4a. 初步海选：按分数排序，保留前 n_pre_nms 个
        order = scores.ravel().argsort(descending = True)
        if self.n_pre_nms > 0:
            order = order[ : self.n_pre_nms]
        boxes = boxes[order, :]
        scores = scores[order]

        # 4b. 正式比赛：执行NMS(torchvision.ops.nms)
        keep_nms = nms(boxes, scores, self.nms_iou_thresh)

        # 4c. 最终决赛：保留前 n_post_nms 个
        if self.n_post_nms > 0:
            keep_nms = keep_nms[ : self.n_post_nms]
        proposals = boxes[keep_nms]

        return proposals


# proposal_creator.py 文件末尾
if __name__ == '__main__':
    # --- 测试 ProposalCreator ---
    print("--- 正在测试 ProposalCreator ---")

    # 假设我们在验证模式
    proposal_creator = ProposalCreator(mode='validation')

    # 创建假的 RPN 输出和锚点
    num_anchors = 22500
    rpn_locs = torch.randn(num_anchors, 4) * 0.1  # 较小的随机偏移

    # 创建假的前景分，让分数有高有低
    scores_raw = torch.randn(num_anchors, 2)
    rpn_scores = torch.softmax(scores_raw, dim=1)[:, 1]  # 只取前景分

    # 创建假的锚点
    # 为了简化，我们只创建一些随机框作为锚点
    # 实际中应使用 AnchorGenerator
    anchors = torch.randint(0, 600, (num_anchors, 4)).float()
    anchors[:, 2] += anchors[:, 0]  # 确保 xmax > xmin
    anchors[:, 3] += anchors[:, 1]  # 确保 ymax > ymin

    # 图像尺寸
    img_size = (800, 800)

    # 调用提议生成器
    proposals = proposal_creator(rpn_locs, rpn_scores, anchors, img_size)

    print(f"输入锚点数量: {num_anchors}")
    print(f"NMS 之前保留数量 (n_pre_nms): {proposal_creator.n_pre_nms}")
    print(f"NMS 之后保留数量 (n_post_nms): {proposal_creator.n_post_nms}")
    print(f"最终生成的提议数量: {proposals.shape[0]}")
    print(f"提议的形状: {proposals.shape}")

    # 验证输出的提议数量最多不超过 n_post_nms
    assert proposals.shape[0] <= proposal_creator.n_post_nms
    assert proposals.shape[1] == 4

    print("\nProposalCreator 测试通过！")




