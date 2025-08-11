from typing import Tuple, Any

from torch import nn


def normalize_to_tuple(value :Any, n :int) -> Tuple:
    """
    这是一个辅助函数，用于将输入值规范化为一个指定长度的元组。
    它复现了 timm.layers.to_ntuple 的核心功能。

    Args:
        value: 输入值。可以是一个单独的元素，也可以是一个列表或元组。
        n: 期望的输出元组长度。

    Returns:
        一个长度为 n 的元组。
    """
    if isinstance(value , (tuple, list)):
        assert len(value) == n ,f"输入的序列长度必须为 {n}, 但实际为 {len(value)}"
        return tuple(value)
    return (value,) * n


if __name__ == '__main__':
    # --- 占位符 ---
    class BasicBlock(nn.Module): pass


    class Bottleneck(nn.Module): pass


    # 假设网络的 stage 数量为 4
    num_stages = 4

    # --- 场景1: 传入一个单一的 Block 类 ---
    # 这是最常见的情况，整个网络使用同一种残差块
    block_input_single = Bottleneck

    # 使用我们的函数进行规范化
    block_fns_single = normalize_to_tuple(block_input_single, num_stages)

    print("--- 场景1: 传入单个元素 ---")
    print(f"输入: {block_input_single}")
    print(f"输出: {block_fns_single}")
    print(f"输出类型: {type(block_fns_single)}")
    print(f"输出长度: {len(block_fns_single)}")
    assert len(block_fns_single) == num_stages

    print("\n" + "=" * 30 + "\n")

    # --- 场景2: 传入一个包含不同 Block 类的元组 ---
    # 这用于构建混合模型
    block_input_mixed = (BasicBlock, BasicBlock, Bottleneck, Bottleneck)

    # 使用我们的函数进行规范化
    block_fns_mixed = normalize_to_tuple(block_input_mixed, num_stages)

    print("--- 场景2: 传入元组 ---")
    print(f"输入: {block_input_mixed}")
    print(f"输出: {block_fns_mixed}")
    print(f"输出类型: {type(block_fns_mixed)}")
    print(f"输出长度: {len(block_fns_mixed)}")
    assert len(block_fns_mixed) == num_stages

