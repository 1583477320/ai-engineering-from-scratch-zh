# LoRA 从零实现
# 演示低秩自适应的微调和推理流程

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第 1 步：LoRA 核心实现
# ============================================================================

class LoRALinear(nn.Module):
    """
    带 LoRA 的线性层。
    原始层权重 W 冻结，额外添加低秩分解 A×B 用于微调。
    推理时可合并权重以加速。
    """

    def __init__(self, in_features, out_features, rank=16, alpha=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 原始线性层（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False,  # 冻结
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features),
            requires_grad=bias,
        ) if bias else None

        # LoRA 低秩矩阵
        # A: 降维 (in_features -> rank)，用高斯初始化
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # B: 升维 (rank -> out_features)，初始化为零
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 缩放因子
        self.scaling = alpha / rank

    def forward(self, x, merge_weight=False):
        """
        前向传播。
        Args:
            x: 输入 (..., in_features)
            merge_weight: 是否使用合并权重（推理模式）
        Returns:
            输出 (..., out_features)
        """
        base_output = F.linear(x, self.weight, self.bias)

        if merge_weight:
            # 推理模式：使用合并后的权重（更快）
            merged_weight = self.weight + self.scaling * self.lora_B @ self.lora_A
            return F.linear(x, merged_weight, self.bias)
        else:
            # 训练模式：分开计算（节省显存）
            lora_output = self.scaling * (self.lora_B @ self.lora_A @ x.t()).t()
            return base_output + lora_output

    def merge_and_remove_galore(self):
        """
        合并 LoRA 权重到原始权重，删除 LoRA 矩阵。
        用于推理部署——减少显存占用，加速前向传播。
        Returns:
            合并后的权重矩阵
        """
        merged_weight = self.weight + self.scaling * self.lora_B @ self.lora_A
        # 释放 LoRA 矩阵
        del self.lora_A, self.lora_B
        self.weight = nn.Parameter(merged_weight, requires_grad=False)
        return self.weight


class LoRASequential(nn.Sequential):
    """
    支持 LoRA 的 Sequential 容器。
    自动遍历子模块，为 Linear 层添加 LoRA。
    """

    def __init__(self, *args, rank=16, alpha=1.0, target_modules=None):
        super().__init__(*args)
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or {"0"}
        self._inject_lora()

    def _inject_lora(self):
        """为所有线性层注入 LoRA。"""
        for idx, module in self.named_children():
            if isinstance(module, nn.Linear) and idx in self.target_modules:
                # 替换为带 LoRA 的版本
                lora_linear = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    bias=module.bias is not None,
                )
                # 复制原始权重
                with torch.no_grad():
                    lora_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        lora_linear.bias.copy_(module.bias)
                setattr(self, idx, lora_linear)


# ============================================================================
# 第 2 步：完整的带 LoRA 的网络
# ============================================================================

class SimpleNetworkWithLoRA(nn.Module):
    """
    一个简单的三层网络，其中两层注入了 LoRA。
    演示如何在真实网络中使用 LoRA。
    """

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, rank=16):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # 为 fc1 和 fc2 注入 LoRA
        self.lora_fc1 = LoRALinear(input_dim, hidden_dim, rank=rank, alpha=rank)
        self.lora_fc2 = LoRALinear(hidden_dim, hidden_dim, rank=rank, alpha=rank)

        # 复制原始权重
        with torch.no_grad():
            self.lora_fc1.weight.copy_(self.fc1.weight)
            self.lora_fc2.weight.copy_(self.fc2.weight)

        # 冻结原始层
        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(False)

    def forward(self, x, use_lora=True, merge=False):
        """
        Args:
            x: 输入
            use_lora: 是否使用 LoRA
            merge: 推理时是否合并权重
        """
        h = self.relu(self.lora_fc1(x, merge_weight=merge) if use_lora else self.fc1(x))
        h = self.relu(self.lora_fc2(h, merge_weight=merge) if use_lora else self.fc2(h))
        return self.fc3(h)


# ============================================================================
# 第 3 步：LoRA 微调训练
# ============================================================================

def train_with_lora(model, train_loader, num_epochs=5, lr=1e-4):
    """
    使用 LoRA 微调模型。
    只有 LoRA 矩阵参与梯度更新。
    """
    # 只优化 LoRA 参数
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([
                {"params": module.lora_A, "lr": lr},
                {"params": module.lora_B, "lr": lr},
            ])

    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.view(x.size(0), -1)  # 展平
            logits = model(x, use_lora=True)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Acc: {accuracy:.4f}")

    return model


# ============================================================================
# 第 4 步：参数统计——对比全量微调和 LoRA
# ============================================================================

def compare_parameters(full_model, lora_model):
    """
    对比全量微调和 LoRA 的参数数量。
    """
    # 全量参数
    total_full = sum(p.numel() for p in full_model.parameters())

    # LoRA 可训练参数
    total_lora = 0
    for name, module in lora_model.named_modules():
        if isinstance(module, LoRALinear):
            total_lora += module.lora_A.numel() + module.lora_B.numel()

    ratio = total_lora / total_full * 100

    print(f"全量模型参数: {total_full:,}")
    print(f"LoRA 可训练参数: {total_lora:,}")
    print(f"LoRA 参数占比: {ratio:.2f}%")

    return total_full, total_lora, ratio


# ============================================================================
# 主程序：演示
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 创建模型
    input_dim = 784  # MNIST 28x28
    hidden_dim = 256
    output_dim = 10
    rank = 16

    full_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )

    lora_model = SimpleNetworkWithLoRA(input_dim, hidden_dim, output_dim, rank=rank)

    # 参数对比
    print("=" * 50)
    print("参数对比")
    print("=" * 50)
    total_full, total_lora, ratio = compare_parameters(full_model, lora_model)

    # 测试前向传播
    print("\n" + "=" * 50)
    print("前向传播测试")
    print("=" * 50)
    batch_size = 32
    dummy_input = torch.randn(batch_size, input_dim).to(device)

    # 训练模式
    lora_model.train()
    output_train = lora_model(dummy_input, use_lora=True, merge=False)
    print(f"训练模式输出形状: {output_train.shape}")

    # 推理模式（合并权重）
    lora_model.eval()
    with torch.no_grad():
        output_infer = lora_model(dummy_input, use_lora=True, merge=True)
    print(f"推理模式输出形状: {output_infer.shape}")

    # 验证合并前后的输出一致性
    diff = (output_train - output_infer).abs().max().item()
    print(f"合并前后最大差异: {diff:.2e}")

    print("\n完成！")
