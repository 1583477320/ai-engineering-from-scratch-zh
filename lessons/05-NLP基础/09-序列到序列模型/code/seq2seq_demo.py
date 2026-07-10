# seq2seq_demo.py — Seq2Seq 瓶颈演示 + 编码器-解码器结构 + 教师强制
# 依赖：torch>=2.0
# 安装：pip install torch
# 对应课程：阶段 05 · 09（序列到序列模型）

import math
import random
from typing import List, Tuple


# ============================================================
# 1. 瓶颈模拟——固定大小上下文向量的信息上限
# ============================================================

def simulate_copy_accuracy(seq_len: int,
                           context_dim: int = 8,
                           seed: int = 0) -> Tuple[float, float]:
    """模拟 Seq2Seq 在复制任务上的表现。

    编码器将输入序列压成一个固定大小（context_dim 维）的上下文向量。
    解码器需要从这个向量中恢复全部输入信息。
    序列越长，信息损失越大——这就是注意力要解决的核心问题。
    """
    rng = random.Random(seed)
    vocab = list("abcdefghij")
    vocab_size = len(vocab)

    # 随机嵌入
    embed = [[rng.gauss(0, 0.3) for _ in range(context_dim)]
             for _ in range(vocab_size)]

    def encode(sequence: List[str]) -> List[float]:
        """编码器：将序列压缩为固定维度的上下文向量。"""
        c = [0.0] * context_dim
        for token in sequence:
            idx = vocab.index(token)
            for d in range(context_dim):
                c[d] = c[d] * 0.85 + embed[idx][d]  # 指数衰减旧的，加入新的
        return c

    def decode_score(ctx: List[float], target: List[str]) -> float:
        """解码器：上下文向量与目标嵌入的相似度。"""
        total = 0.0
        for token in target:
            idx = vocab.index(token)
            score = sum(ctx[d] * embed[idx][d] for d in range(context_dim))
            total += max(0.0, math.tanh(score))
        return total / max(1, len(target))

    # 测试：正样本 vs 噪声
    trials = 200
    hits = 0
    for _ in range(trials):
        seq = [rng.choice(vocab) for _ in range(seq_len)]
        ctx = encode(seq)
        target_score = decode_score(ctx, seq)
        noise = [rng.choice(vocab) for _ in range(seq_len)]
        noise_score = decode_score(ctx, noise)
        if target_score > noise_score:
            hits += 1

    accuracy = hits / trials
    # 信息压缩比：context_dim 个浮点数需要无损表示 seq_len 个离散选择
    info_bits_in = seq_len * math.log2(len(vocab))
    info_bits_bottleneck = context_dim * 32  # 32-bit float
    compression_ratio = info_bits_in / info_bits_bottleneck if info_bits_bottleneck else float("inf")

    return accuracy, compression_ratio


def demo_bottleneck():
    print("=" * 60)
    print("Seq2Seq 瓶颈演示——复制任务的准确率随序列长度衰减")
    print("=" * 60)
    print(f"{'序列长度':<10} {'复制准确率':<12} {'信息压缩比':<12} {'状态'}")
    print("-" * 55)
    for length in [5, 10, 20, 40, 80]:
        acc, ratio = simulate_copy_accuracy(length)
        status = "OK" if acc > 0.8 else "告警" if acc > 0.5 else "崩溃"
        bar = "█" * int(acc * 10)
        print(f"{length:<10} {acc:.1%}  {bar:<10} {ratio:.1f}x{'':<6} {status}")
    print("\n结论：一个固定大小的上下文向量无法无损存储任意长度的输入。")
    print("编码器逐步丢失了序列前部的信息（指数衰减机制）。")
    print("注意力机制修复了这一点——解码器可以回看编码器的每个位置。")


# ============================================================
# 2. 编码器-解码器架构（PyTorch 版本）
# ============================================================

def demo_architecture():
    print("\n" + "=" * 60)
    print("编码器-解码器架构")
    print("=" * 60)
    try:
        import torch
        import torch.nn as nn

        class Encoder(nn.Module):
            """编码器：读源语言 → 输出上下文向量。"""
            def __init__(self, vocab_size, embed_dim, hidden_dim):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

            def forward(self, src):
                e = self.embed(src)              # (B, S, E)
                outputs, hidden = self.gru(e)    # outputs: (B,S,H), hidden: (1,B,H)
                return outputs, hidden

        class Decoder(nn.Module):
            """解码器：从上下文向量生成目标语言。"""
            def __init__(self, vocab_size, embed_dim, hidden_dim):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, vocab_size)

            def forward(self, token, hidden):
                e = self.embed(token)            # (B, 1, E)
                out, hidden = self.gru(e, hidden) # (B, 1, H)
                logits = self.fc(out)            # (B, 1, V)
                return logits, hidden

        # 形状验证
        B, S_src, S_tgt = 2, 5, 4
        V_src, V_tgt, E, H = 20, 20, 8, 16
        bos_id = 1

        encoder = Encoder(V_src, E, H)
        decoder = Decoder(V_tgt, E, H)

        src = torch.randint(0, V_src, (B, S_src))
        _, hidden = encoder(src)
        print(f"编码器: {src.shape} → hidden {hidden.shape}")
        print(f"  上下文向量大小: {hidden.numel()} 个浮点数")
        print(f"  这 {hidden.numel()} 个数必须概括整个输入序列的全部信息。")

        # 逐词元解码演示
        inp = torch.full((B, 1), bos_id, dtype=torch.long)
        print(f"\n解码器（逐词元生成）:")
        for t in range(3):
            logits, hidden = decoder(inp, hidden)
            pred = logits.argmax(dim=-1)
            print(f"  第 {t+1} 步: 输入=BOS → 预测 token ID = {pred.squeeze().tolist()}")
            inp = pred  # 用预测当作下一步输入（推理模式）

        # 教师强制 vs 自回归的差异
        print(f"\n关键概念:")
        print(f"  教师强制（训练）: 解码器的输入 = 上一步的真实目标词元")
        print(f"  自回归（推理）:   解码器的输入 = 上一步自己预测的词元")
        print(f"  暴露偏差:         训练从未让其从错误中恢复 → 推理时一步错步步错")

    except ImportError:
        print("PyTorch 未安装，跳过。安装：pip install torch")


# ============================================================
# 演示主程序
# ============================================================

def main():
    demo_bottleneck()
    demo_architecture()

    print("\n" + "=" * 60)
    print("从 Seq2Seq 到 Transformer 的演化逻辑")
    print("=" * 60)
    print("""
    Seq2Seq (2014)
    │  编码器: RNN 读源序列 → 最后一个隐藏状态
    │  解码器: RNN 从上下文向量生成目标序列
    │  问题: 上下文向量是固定大小的——长序列信息丢失
    │
    ├─→ Attention (2015)
    │     解码器的每一步可以回看编码器的 ALL 隐藏状态
    │     不再被一个固定向量限制——任意长度的输入皆可
    │
    └─→ Transformer (2017)
          完全抛弃 RNN——全注意力 + 位置编码
          并行训练、O(1) 路径长度的 token 间通信
    """)


if __name__ == "__main__":
    main()
