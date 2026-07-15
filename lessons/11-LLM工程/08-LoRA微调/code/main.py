# LoRA 微调：低秩适配演示

import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# 第 1 步：LoRA 层
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA 层——冻结原始权重，只训练低秩增量。"""
    def __init__(self, original_linear, rank=16, alpha=16):
        super().__init__()
        self.original = original_linear
        self.original.requires_grad_(False)  # 冻结
        d_in, d_out = original_linear.in_features, original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.original(x) + self.scaling * (x @ self.lora_A @ self.lora_B.T)


# ============================================================================
# 第 2 步：模拟微调
# ============================================================================

def demo_lora():
    """演示 LoRA 参数量对比。"""
    d = 768  # 隐藏维度
    linear = nn.Linear(d, d)

    # 全量微调参数
    full_params = d * d  # 589,824

    # LoRA 参数
    r = 16
    lora_params = 2 * d * r  # 24,576

    print(f"全量微调参数: {full_params:,}")
    print(f"LoRA 参数 (r={r}): {lora_params:,}")
    print(f"LoRA 占比: {lora_params/full_params:.2%}")

    # QLoRA 内存估算
    print(f"\n显存估算:")
    print(f"  全量微调: ~{(d*d*4 + d*d*8)/1e9:.2f}GB (权重+优化器)")
    print(f"  QLoRA INT4: ~{(d*d*0.5 + lora_params*4)/1e9:.2f}GB (量化+LoRA)")


if __name__ == "__main__":
    print("LoRA 微调演示\n")
    demo_lora()
