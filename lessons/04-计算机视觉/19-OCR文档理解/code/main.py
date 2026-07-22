# main.py — 从零实现 OCR 核心：CTC 损失 + TinyCRNN + 贪婪解码
# 依赖：torch>=2.0, numpy>=1.24
# 对应课程：第 4 阶段 · 19（OCR 文档理解）

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


# === 词表定义 ===
VOCAB = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
BLANK_IDX = 0


# === CTC 损失封装 ===
def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = BLANK_IDX,
) -> torch.Tensor:
    """
    计算 CTC 损失。

    Args:
        log_probs: (T, N, C) 对数概率
        targets: (N, S) 目标字符索引（不含空白符）
        input_lengths: (N,) 每个样本的有效时间步数
        target_lengths: (N,) 每个样本的目标长度
        blank: 空白符索引

    Returns:
        标量损失值
    """
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )


# === 贪婪解码器 ===
def greedy_ctc_decode(log_probs: torch.Tensor, blank: int = BLANK_IDX) -> list[list[int]]:
    """
    CTC 贪婪解码：取每步最大概率，合并重复，移除空白。

    Args:
        log_probs: (T, N, C) 对数概率张量
        blank: 空白符索引

    Returns:
        解码后的字符索引序列列表
    """
    # 取每步最大概率的索引，交换批次和时间维 → (N, T)
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()

    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            # 跳过空白符，合并连续重复
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out


# === Beam Search 解码器 ===
def beam_ctc_decode(
    log_probs: torch.Tensor,
    vocab: list[str],
    beam_width: int = 5,
    blank: int = BLANK_IDX,
) -> list[str]:
    """
    CTC Beam Search 解码器。

    在贪婪解码的基础上维护一个候选前缀队列，每一步扩展 beam_width 个最可能的
    前缀。当模型置信度较低（如模糊图像、手写体）时，Beam Search 比贪婪解码
    能获得更低的 CER。

    Args:
        log_probs: (T, N, C) 对数概率张量
        vocab: 词表（不含空白符）
        beam_width: 束搜索宽度
        blank: 空白符索引

    Returns:
        解码后的字符串列表
    """

    def _logsumexp(a: float, b: float) -> float:
        """数值稳定的 log-sum-exp。"""
        if a == float("-inf"):
            return b
        if b == float("-inf"):
            return a
        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    results: list[str] = []
    lp = log_probs.cpu().numpy()  # (T, N, C)
    T, N, C = lp.shape

    for n in range(N):
        # beams: {前缀元组: (以空白结尾的对数概率, 不以空白结尾的对数概率)}
        beams: dict[tuple, tuple[float, float]] = {((),): (0.0, float("-inf"))}

        for t in range(T):
            new_beams: dict[tuple, tuple[float, float]] = {}
            logits_t = lp[t, n]  # (C,)

            for prefix, (p_blank, p_nonblank) in beams.items():
                for c in range(C):
                    prob = logits_t[c]
                    if c == blank:
                        # 状态 1: 添加空白 → 仍以空白结尾
                        p_b_new = _logsumexp(p_blank + prob, p_nonblank + prob)
                        # 状态 2: 添加空白 → 转为非空白结尾（当前字符为空白，无效）
                        p_nb_new = p_nonblank + prob

                        existing = new_beams.get(prefix, (float("-inf"), float("-inf")))
                        new_beams[prefix] = (
                            _logsumexp(existing[0], p_b_new),
                            existing[1],
                        )
                        # 非空白部分保持不变（blank 路径不影响 p_nb）
                        pass
                    else:
                        char = vocab[c - 1]  # vocab 不含空白符，所以减 1
                        if char == prefix[-1] if prefix else "":
                            # 相邻相同字符：需要通过空白分隔
                            # Case A: 延长现有前缀（从 p_nonblank 路径添加）
                            updated_p_nb = _logsumexp(
                                new_beams.get(prefix, (float("-inf"), float("-inf")))[1],
                                p_nonblank + prob,
                            )
                            existing = new_beams.get(prefix, (float("-inf"), float("-inf")))
                            new_beams[prefix] = (existing[0], updated_p_nb)

                            # Case B: 扩展为新前缀（从 p_blank 路径添加）
                            new_prefix = prefix + (char,)
                            p_extend = _logsumexp(p_blank, p_nonblank) + prob
                            updated = new_beams.get(new_prefix, (float("-inf"), float("-inf")))
                            new_beams[new_prefix] = (updated[0], _logsumexp(updated[1], p_extend))
                        else:
                            # 不同字符：直接追加
                            new_prefix = prefix + (char,)
                            p_extend = _logsumexp(p_blank, p_nonblank) + prob
                            updated = new_beams.get(new_prefix, (float("-inf"), float("-inf")))
                            new_beams[new_prefix] = (updated[0], _logsumexp(updated[1], p_extend))

            # 保留 top-K 前缀
            sorted_beams = sorted(
                new_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )
            beams = dict(sorted_beams[:beam_width])

        # 选择最优前缀
        best_prefix = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        results.append("".join(best_prefix))

    return results


# === TinyCRNN 文字识别网络 ===
class TinyCRNN(nn.Module):
    """
    最小的卷积-循环文字识别网络。

    CNN 负责提取空间特征并将高度压缩到 1，
    BiLSTM 负责利用左右上下文帮助字符识别，
    全连接层 + CTC 损失负责输出字符序列。

    输入: (N, 1, H, W) — 灰度图像，固定高度，可变宽度
    输出: (W', N, C) — 每个时间步的词元分布（含空白符）
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        hidden_size: int = 128,
        feat_size: int = 32,
    ):
        super().__init__()

        # --- CNN 特征提取器 ---
        # MaxPool2d((2, 1)) 只压缩高度不压缩宽度，确保最终高度为 1
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat_size, 3, 1, 1),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feat_size, feat_size * 2, 3, 1, 1),
            nn.BatchNorm2d(feat_size * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feat_size * 2, feat_size * 4, 3, 1, 1),
            nn.BatchNorm2d(feat_size * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(feat_size * 4, feat_size * 4, 3, 1, 1),
            nn.BatchNorm2d(feat_size * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )

        # --- BiLSTM 序列建模 ---
        self.rnn = nn.LSTM(
            feat_size * 4,
            hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        # --- 输出层 ---
        self.head = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入图像 (N, 1, H, W)

        Returns:
            对数概率 (T, N, C)，CTC 格式
        """
        features = self.cnn(x)       # (N, C', H', W')
        features = features.mean(dim=2)  # (N, C', W') → 高度压缩到 1
        features = features.transpose(1, 2)  # (N, W', C') → 序列格式

        h, _ = self.rnn(features)   # (N, W', 2*hidden)
        logits = self.head(h)        # (N, W', C)

        return logits.transpose(0, 1).contiguous()  # (W', N, C)


# === 合成数据生成器 ===
def synthetic_line(
    text: str, height: int = 32, char_width: int = 16
) -> np.ndarray:
    """
    在白色背景上生成黑色文字行图像。
    字母数字字符用黑色（0.0），其他字符用灰色（0.5）。

    Args:
        text: 要生成的文本
        height: 图像高度（像素）
        char_width: 每个字符占用的宽度（像素）

    Returns:
        灰度图像 ndarray，形状 (height, width)，值域 [0, 1]
    """
    width = char_width * max(1, len(text))
    img = np.ones((height, width), dtype=np.float32)

    for i, char in enumerate(text):
        x_start = i * char_width
        shade = 0.0 if char.isalnum() else 0.5
        img[6 : height - 6, x_start + 2 : x_start + char_width - 2] = shade

    return img


def build_batch(
    strings: List[str], max_len: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将文本字符串批次化为 (图像, 目标索引, 目标长度)。

    Args:
        strings: 文本字符串列表
        max_len: 最大字符数（填充用）

    Returns:
        images: (N, 1, H, W)
        targets: 展平的目标索引
        target_lengths: 每个样本的目标长度
    """
    height = 32
    max_len = max_len or max(len(s) for s in strings)
    width = 16 * max_len

    images = np.ones((len(strings), 1, height, width), dtype=np.float32)
    targets: list[int] = []
    target_lengths: list[int] = []

    for i, s in enumerate(strings):
        line_img = synthetic_line(s)
        images[i, 0, :, : line_img.shape[1]] = line_img

        char_ids = [VOCAB.index(c) if c in VOCAB else BLANK_IDX for c in s]
        targets.extend(char_ids)
        target_lengths.append(len(char_ids))

    return (
        torch.from_numpy(images),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def decode_to_str(ids: List[int]) -> str:
    """将 CTC 输出索引序列解码为可读字符串。"""
    return "".join(VOCAB[i] for i in ids)


def compute_cer(predicted: str, reference: str) -> float:
    """
    计算字符错误率（CER）。
    CER = Levenshtein 编辑距离 / 参考文本长度
    """
    m, n = len(reference), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == predicted[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # 删除
                dp[i][j - 1] + 1,          # 插入
                dp[i - 1][j - 1] + cost,   # 替换
            )

    edit_dist = dp[m][n]
    return edit_dist / m if m > 0 else 0.0


# === 主程序 ===
def main():
    """演示 TinyCRNN + CTC 的合成 OCR 训练和评估流程。"""
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCRNN(vocab_size=len(VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"=== TinyCRNN OCR 训练演示 ===")
    print(f"模型参数总量: {total_params:,}")
    print(f"运行设备: {device}")
    print()

    # --- 训练循环 ---
    train_vocab = [f"abc{d}" for d in range(10)] + [f"xy{d}{d+1}" for d in range(10)]
    num_steps = 200

    for step in range(num_steps):
        idx = np.random.choice(len(train_vocab), 8)
        strings = [train_vocab[i] for i in idx]
        imgs, targets, target_lens = build_batch(strings, max_len=5)
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lens = target_lens.to(device)

        log_probs = model(imgs)       # (T, N, C)
        input_lens = torch.full(
            (imgs.size(0),), log_probs.size(0), dtype=torch.long
        )

        loss = ctc_loss(
            log_probs, targets, input_lens, target_lens, blank=BLANK_IDX
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            print(f"Step {step:3d}  Loss: {loss.item():.3f}")

    print()

    # --- 贪婪解码评估 ---
    print("--- 贪婪解码测试结果 ---")
    test_strings = ["abc7", "xy45", "abc2", "xyz9"]
    model.eval()
    imgs, _, _ = build_batch(test_strings, max_len=5)
    imgs = imgs.to(device)

    with torch.no_grad():
        log_probs = model(imgs)

    greedy_preds = greedy_ctc_decode(log_probs)
    cer_scores = []

    for target, pred_ids in zip(test_strings, greedy_preds):
        pred_str = decode_to_str(pred_ids)
        cer = compute_cer(pred_str, target)
        cer_scores.append(cer)
        match = "正确" if target == pred_str else "偏差"
        print(f"  真实: {target!r:>10s}  预测: {pred_str!r:>10s}  CER: {cer:.1%}  [{match}]")

    avg_cer = np.mean(cer_scores)
    print(f"  平均 CER: {avg_cer:.1%}")
    print()

    # --- Beam Search 解码对比 ---
    print("--- Beam Search (width=5) 对比 ---")
    beam_preds = beam_ctc_decode(log_probs, VOCAB[1:], beam_width=5, blank=BLANK_IDX)

    beam_cers = []
    for target, pred_str in zip(test_strings, beam_preds):
        cer = compute_cer(pred_str, target)
        beam_cers.append(cer)
        match = "正确" if target == pred_str else "偏差"
        print(f"  真实: {target!r:>10s}  预测: {pred_str!r:>10s}  CER: {cer:.1%}  [{match}]")

    avg_beam_cer = np.mean(beam_cers)
    print(f"  平均 CER: {avg_beam_cer:.1%}")
    print()

    # --- 两种解码器的 CER 对比 ---
    print(f"=== 解码策略对比 ===")
    print(f"  贪婪解码平均 CER:  {avg_cer:.1%}")
    print(f"  Beam Search CER:   {avg_beam_cer:.1%}")
    improvement = (avg_cer - avg_beam_cer) / max(avg_cer, 1e-8) * 100
    print(f"  改善幅度: {improvement:+.1f}%")
    print()

    if avg_beam_cer < avg_cer:
        print("  结论: Beam Search 在此批次上优于贪婪解码")
    elif avg_beam_cer > avg_cer:
        print("  结论: 贪婪解码在此批次上更好（合成数据过于简单，差距可忽略）")
    else:
        print("  结论: 两者表现相同（数据太简单不需要搜索）")


if __name__ == "__main__":
    main()
