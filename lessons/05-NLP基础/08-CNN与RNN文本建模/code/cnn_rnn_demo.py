# cnn_rnn_demo.py — 理解 CNN/RNN 在文本上的核心机制
# 依赖：torch>=2.0
# 安装：pip install torch
# 对应课程：阶段 05 · 08（CNN 与 RNN 文本建模）

import math
from typing import List


# ============================================================
# 1. 梯度消失演示——为什么普通 RNN 学不到长距离依赖
# ============================================================

def vanishing_gradient_sim(seq_len: int,
                           recurrent_weight: float = 0.9) -> float:
    """模拟普通 RNN 的梯度衰减。

    h_t = f(W·x_t + U·h_{t-1})
    反向传播时，梯度乘了 U 的 seq_len 次方。
    U < 1 → 梯度指数衰减 → 远端信号归零 → 学不到长距离依赖。
    U > 1 → 梯度爆炸 → 训练崩溃。
    LSTM 用门控和"梯度高速公路"（cell state）解决了这个问题。
    """
    return math.pow(recurrent_weight, seq_len)


def demo_vanishing():
    print("=== 梯度消失演示 ===")
    print("普通 RNN: h_t = tanh(W·x_t + U·h_{t-1})")
    print("梯度反传时乘了 U 的 seq_len 次方 →")
    for length in [10, 50, 100, 200]:
        grad = vanishing_gradient_sim(length, recurrent_weight=0.9)
        bars = "█" * max(1, int(-math.log10(max(grad, 1e-30))))
        print(f"  seq_len={length:3d}: "
              f"0.9^{length:3d} = {grad:.2e}  {bars}")
    print("  结论：到第 100 步，梯度只剩 0.0027%——远端信号完全丢失。")


# ============================================================
# 2. TextCNN 概念演示——1D 卷积 + 全局最大池化
# ============================================================

def conv1d_over_embeddings(embeddings: List[List[float]],
                           filter_matrix: List[List[float]],
                           bias: float = 0.0) -> List[float]:
    """在词嵌入序列上模拟 1D 卷积。

    filter_matrix 的 shape = (filter_width, embed_dim)。
    卷积核在序列上滑动，每次与连续的 filter_width 个词嵌入做逐元素乘积。
    输出一个"特征图"——每个位置一个激活值。
    """
    filter_width = len(filter_matrix)
    embed_dim = len(embeddings[0])
    out = []
    for i in range(len(embeddings) - filter_width + 1):
        total = bias
        for k in range(filter_width):
            for d in range(embed_dim):
                total += embeddings[i + k][d] * filter_matrix[k][d]
        out.append(max(total, 0.0))  # ReLU
    return out


def demo_textcnn():
    print("\n=== TextCNN 概念演示 ===")
    # 5 个词元，每个 3 维嵌入
    embeddings = [
        [1.0, 0.2, 0.5],   # 词 0: 可能是 "not"
        [0.8, 0.9, 0.1],   # 词 1: 可能是 "good"
        [0.3, 0.4, 0.7],   # 词 2
        [0.6, 0.5, 0.5],   # 词 3
        [0.1, 0.8, 0.2],   # 词 4
    ]

    # 宽度=2 的卷积核（检测 bigram 模式，如 "not good"）
    filter_w2 = [[0.5, 0.0, 0.5], [0.2, 0.3, 0.1]]
    # 宽度=3 的卷积核（检测 trigram 模式）
    filter_w3 = [[0.3, 0.3, 0.3], [0.2, 0.4, 0.1], [0.5, 0.2, 0.1]]

    act_w2 = conv1d_over_embeddings(embeddings, filter_w2, bias=-0.5)
    act_w3 = conv1d_over_embeddings(embeddings, filter_w3, bias=-0.4)

    print(f"  输入：5 个词元 × 3 维嵌入")
    print(f"  宽度=2 卷积激活: {[round(x, 2) for x in act_w2]}")
    print(f"  宽度=3 卷积激活: {[round(x, 2) for x in act_w3]}")

    # 全局最大池化：每个卷积核只保留最强的激活位置
    pooled_w2 = max(act_w2)
    pooled_w3 = max(act_w3)
    print(f"  全局最大池化:    w2_max={pooled_w2:.2f}, w3_max={pooled_w3:.2f}")
    print(f"  理解：'not good' 在任意位置触发 → 最大池化捕获了此信号 ")
    print(f"       无论这对词出现在句首还是句末，池化值相同")
    print(f"       这就是 TextCNN 的位置不变性——也是它无法建模序列顺序的原因")


# ============================================================
# 3. PyTorch 版 TextCNN + LSTM（框架级实现）
# ============================================================

def demo_pytorch():
    print("\n=== PyTorch TextCNN + BiLSTM 架构（代码结构展示）===")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class TextCNN(nn.Module):
            """Kim (2014) 的文本卷积网络。

            多个宽度的卷积核在嵌入序列上滑动 → 全局最大池化 →
            拼接 → 全连接分类。训练完全并行——没有序列依赖。
            """
            def __init__(self, vocab_size, embed_dim, n_classes,
                         filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                # 每个宽度对应一组卷积核
                self.convs = nn.ModuleList([
                    nn.Conv1d(embed_dim, n_filters, kernel_size=k)
                    for k in filter_widths
                ])
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

            def forward(self, token_ids):
                # (batch, seq_len, embed_dim) → (batch, embed_dim, seq_len)
                x = self.embed(token_ids).transpose(1, 2)
                pooled = []
                for conv in self.convs:
                    c = F.relu(conv(x))
                    # 全局最大池化 → (batch, n_filters)
                    p = F.max_pool1d(c, c.size(2)).squeeze(2)
                    pooled.append(p)
                h = torch.cat(pooled, dim=1)  # (batch, n_filters * n_widths)
                return self.fc(self.dropout(h))

        class BiLSTMClassifier(nn.Module):
            """双向 LSTM + 最大池化 → 分类。

            LSTM 从左到右和从右到左各跑一遍 → 每个词元的表示
            同时包含了上文和下文信息。最大池化选取整个序列中最强
            的激活位置——比取"最后一个隐藏状态"更稳定。
            """
            def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                         bidirectional=True, dropout=0.3):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.lstm = nn.LSTM(embed_dim, hidden_dim,
                                    batch_first=True,
                                    bidirectional=bidirectional)
                factor = 2 if bidirectional else 1
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim * factor, n_classes)

            def forward(self, token_ids):
                x = self.embed(token_ids)
                out, _ = self.lstm(x)        # (batch, seq_len, hidden*factor)
                # 全局最大池化 → 选序列中每维最强的信号
                pooled = out.max(dim=1).values
                return self.fc(self.dropout(pooled))

        # 玩具数据验证形状
        batch, seq_len, vocab = 2, 5, 20
        x = torch.randint(0, vocab, (batch, seq_len))

        cnn = TextCNN(vocab, 8, 2)
        lstm = BiLSTMClassifier(vocab, 8, 16, 2)
        print(f"  TextCNN 输入: {x.shape} → 输出: {cnn(x).shape}")
        print(f"  BiLSTM  输入: {x.shape} → 输出: {lstm(x).shape}")
        print(f"  TextCNN 参数量: {sum(p.numel() for p in cnn.parameters()):,}")
        print(f"  BiLSTM  参数量: {sum(p.numel() for p in lstm.parameters()):,}")

    except ImportError:
        print("  PyTorch 未安装，跳过。安装：pip install torch")


# ============================================================
# 演示主程序
# ============================================================

def main():
    demo_vanishing()
    demo_textcnn()
    demo_pytorch()

    print("\n=== 总结：CNN vs RNN vs Transformer 的选择 ===")
    print("┌────────────┬──────────┬──────────┬─────────────┐")
    print("│            │ TextCNN  │ BiLSTM   │ Transformer │")
    print("├────────────┼──────────┼──────────┼─────────────┤")
    print("│ 并行训练   │    ✅    │    ❌    │     ✅      │")
    print("│ 长距离依赖 │    ❌    │    ⚠️    │     ✅      │")
    print("│ 推理速度   │  极快    │   快     │    慢       │")
    print("│ 模型大小   │  小(几MB)│  中      │   大(百MB+) │")
    print("│ 边缘部署   │  最佳    │  可用    │   困难      │")
    print("│ 流式处理   │    ❌    │    ✅    │     ❌      │")
    print("└────────────┴──────────┴──────────┴─────────────┘")
    print("规则：边缘设备→TextCNN。流式输入→LSTM。其他→Transformer。")

if __name__ == "__main__":
    main()
