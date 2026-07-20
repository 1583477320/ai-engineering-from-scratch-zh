# 综合项目41——完整评测管道（Full Eval Pipeline）

> 训练是可以用损失曲线监控的部分。评测是你必须设计的部分。本课程构建统一的评测管道，对任何训练好的语言模型运行四种异构评测，聚合为每任务报告，并内置本地 mock LLM 评判器。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第19章第30-37节
**预计时间：** 90分钟

---

## 学习目标

- 计算带掩码词元计数的 hold-out 困惑度
- 在短形式事实提示上运行精确匹配评测
- 计算预测和参考字符串间的词元 F1
- 构建本地 mock LLM 评判器（1-5 分）
- 将四种评测聚合为加权报告

---

## 1. 问题

单一指标从不描述一个语言模型。困惑度说模型拟合语言分布多好，但不说它是否回答问题。精确匹配说模型是否产生 gold 字符串，但惩罚正确改写。词元 F1 原谅改写但被错误内容的词重叠欺骗。

例如：模型回答"巴黎是法国的首都"时：

| 指标 | 得分 | 原因 |
|------|------|------|
| 困惑度 | 反映分布拟合度 | 正确的语言模型概率分配 |
| 精确匹配 | 0 | 不是 "Paris" 的精确匹配 |
| 词元 F1 | 0.5 | "Paris" 在预测中出现，但有额外词元 |
| 评判器 | 5/5 | 回答正确且自然 |

你需要的管道是四种指标都有。

---

## 2. 核心概念

### 2.1 困惑度（正确计数）

困惑度 = $\exp(\text{平均每个词元的负对数似然})$。两个陷阱：

1. 均值必须在**实际词元位置**上——排除 padding 词元
2. 模型在位置 $i$ 预测位置 $i+1$ 的词元（自回归模型的 shifted 目标）

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i | x_1, \ldots, x_{i-1})\right)$$

### 2.2 精确匹配

对预测和参考进行归一化：小写化、去空白、折叠双空格、去除尾标点。归一化使精确匹配在实践中可用——"Paris。"和 "Paris" 应该视为相同。

### 2.3 词元 F1

将预测和参考分别分词为词元集合，计算多集交集：

```
精确率 = 交集大小 / 预测词元数
召回率 = 交集大小 / 参考词元数
F1 = 2 × 精确率 × 召回率 / (精确率 + 召回率)
```

### 2.4 Mock LLM 评判器

5 分制评分标准：

| 分数 | 条件 |
|------|------|
| 5 | 精确匹配 |
| 4 | 词元 F1 ≥ 0.8 |
| 3 | 词元 F1 ∈ [0.5, 0.8) |
| 2 | 词元 F1 ∈ [0.2, 0.5) |
| 1 | 词元 F1 < 0.2 |

在生产中，该评分器替换为 Claude 或 GPT-4 作为评判模型。

### 2.5 聚合

加权平均归一化分数。默认权重：0.2 困惑度、0.3 精确匹配、0.3 词元 F1、0.2 评判。权重可按任务类型调整。

---

## 3. 从零实现

```python
"""完整评测管道——困惑度+精确匹配+词元 F1+评判器。"""
import json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


class InstructionTokenizer:
    """简单的字节级词元化器。"""
    INST = 256; RESP = 257; PAD = 258; VOCAB = 260; IGNORE = -100


class CausalSelfAttn(nn.Module):
    """因果自注意力——每个位置只能关注它之前的位置。"""
    def __init__(self, D, H, L):
        super().__init__()
        self.H = H; self.dh = D // H
        self.qkv = nn.Linear(D, D * 3, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(L, L, dtype=torch.bool)))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        return self.out((F.softmax(att, -1) @ v).transpose(1, 2).contiguous().view(B, T, D))


class Block(nn.Module):
    """Transformer 块：注意力 + 前馈网络。"""
    def __init__(self, D, H, L):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = CausalSelfAttn(D, H, L)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, D * 4)
        self.fc2 = nn.Linear(D * 4, D)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)))) + x


class TinyGPT(nn.Module):
    """小型 GPT 模型，用于演示评测管道。"""
    def __init__(self, V, D, H, depth, L):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(L, D)
        self.blocks = nn.ModuleList([Block(D, H, L) for _ in range(depth)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.L = L

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


# ── 指标实现 ──

def perplexity_eval(model, ids, mask):
    """带掩码的困惑度评估。排除 padding 词元。"""
    with torch.no_grad():
        logits = model(ids)
        shifted_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shifted_logits, targets, ignore_index=258, reduction="none")
        mask_flat = mask[:, 1:].reshape(-1).float()
        total = (loss * mask_flat).sum()
        tokens = mask_flat.sum().clamp(min=1)
    return {"perplexity": float(math.exp(total / tokens.item())), "tokens": int(tokens.item())}


def normalise(text):
    """文本归一化——小写、去空白、去尾标点。"""
    s = text.lower().strip()
    s = " ".join(s.split())
    if s and s[-1] in ".!?":
        s = s[:-1]
    return s


def exact_match_eval(pairs):
    """精确匹配评估。"""
    hits = sum(1 for p, r in pairs if normalise(p) == normalise(r))
    return {"exact_match": hits / max(len(pairs), 1), "total": len(pairs), "hits": hits}


def token_f1(pred, ref):
    """词元级 F1 分数。"""
    p = set(normalise(pred).split())
    r = set(normalise(ref).split())
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    inter = len(p & r)
    prec = inter / max(len(p), 1)
    rec = inter / max(len(r), 1)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0


def token_f1_eval(pairs):
    """批量词元 F1 评估。"""
    scores = [token_f1(p, r) for p, r in pairs]
    return {"token_f1": float(np.mean(scores)), "per_example": scores}


def mock_judge(inst, pred, ref):
    """Mock LLM 评判器——基于词元 F1 的 5 分制评分。"""
    if normalise(pred) == normalise(ref):
        return 5, "精确匹配"
    f1 = token_f1(pred, ref)
    if f1 >= 0.8:
        return 4, "高度重叠"
    if f1 >= 0.5:
        return 3, "中等重叠"
    if f1 >= 0.2:
        return 2, "低度重叠"
    return 1, "极少匹配"


def judge_eval(pairs):
    """批量评判评估。"""
    scores = [mock_judge(i, p, r)[0] for i, p, r in pairs]
    return {"judge_score": float(np.mean(scores)) / 5, "per_example": scores}


def aggregate(results, weights):
    """将多种指标聚合为加权总分。"""
    norm = lambda k, v: min(1, 1 / (1 + math.log(max(v, 1e-10)))) if k == "perplexity" else v
    total = sum(weights[k] * norm(k, results[k]) for k in weights)
    return {"aggregate": total, "details": {k: norm(k, v) for k, v in results.items()}}


# ── 主程序 ──

def main():
    torch.manual_seed(0)
    tok = InstructionTokenizer()
    model = TinyGPT(260, 96, 4, 2, 96)
    # 简单训练
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    for _ in range(10):
        ids = torch.randint(0, 260, (4, 64))
        loss = F.cross_entropy(model(ids)[:, :-1, :].reshape(-1, 260), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    # 评测
    ids = torch.randint(0, 260, (2, 32))
    mask = torch.ones(2, 32)
    ppl = perplexity_eval(model, ids, mask)
    print(f"困惑度: {ppl['perplexity']:.2f} (词元数: {ppl['tokens']})")

    em = exact_match_eval([("Paris", "Paris."), ("Madrid", "Madrid")])
    print(f"精确匹配: {em['exact_match']:.2f} ({em['hits']}/{em['total']})")

    f1 = token_f1_eval([("Paris is capital", "Paris"), ("Tokyo is capital", "Tokyo")])
    print(f"词元 F1: {f1['token_f1']:.2f}")

    jd = judge_eval([("Capital?", "Paris", "Paris."), ("Capital?", "Paris is in France", "Tokyo")])
    print(f"评判: {jd['judge_score']:.2f}")

    agg = aggregate(
        {"perplexity": ppl["perplexity"], "exact_match": em["exact_match"],
         "token_f1": f1["token_f1"], "judge_score": jd["judge_score"]},
        {"perplexity": 0.2, "exact_match": 0.3, "token_f1": 0.3, "judge_score": 0.2},
    )
    print(f"聚合: {agg['aggregate']:.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

---

## 4. 工业工具

### 4.1 HuggingFace Evaluate

HuggingFace 提供了标准化的评估工具：

```python
import evaluate

# ROUGE
rouge = evaluate.load("rouge")
result = rouge.compute(predictions=[pred], references=[ref])

# 精确匹配
exact_match = evaluate.load("exact_match")
result = exact_match.compute(predictions=[pred], references=[ref])
```

### 4.2 评测框架选择

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| LM Evaluation Harness | 最全面的 LLM 评测 | 学术评测 |
| HuggingFace Evaluate | 标准化指标库 | 快速原型 |
| lm-eval-hallucination | 专注幻觉检测 | 安全评估 |
| 自定义管道 | 完全可控 | 生产环境 |

### 4.3 评判器（Judge-as-a-Judge）

在现代 LLM 评测中，使用 Claude 或 GPT-4 作为评判器已成为标准做法：

```python
judge_prompt = f"""评分以下回答（1-5分）：
问题：{question}
参考答案：{reference}
模型回答：{prediction}

评分标准：5=精确匹配，4=正确但不精确，3=部分正确，2=勉强相关，1=错误
评分："""
```

---

## 5. 工程最佳实践

### 5.1 评测数据隔离

训练集和评测集必须严格隔离。hold-out 评测的 token 数量必须与训练数据完全分开——否则困惑度评估会被数据污染。

### 5.2 基线对比

始终建立基线对比。对困惑度，基线是随机初始化模型的困惑度（约等于词表大小）。对精确匹配，基线是简单规则（如选择最频繁答案）的表现。

### 5.3 中文场景特别建议

- **分词对词元 F1 的影响**：中文词元 F1 使用空格切分（`split()`）与英文等价，但对中文文本会产生单字符词元，需要改用中文分词器。更好的方式是使用字符级的集合交集。
- **评判器的中文能力**：mock 评判器在中文上效果有限。生产中应替换为中文大模型评判器（如 GPT-4）。

---

## 6. 常见错误

### 错误 1：困惑度评估未排除 padding

**现象：** 困惑度远低于真实值（看起来很好，但实际不好）。

**原因：** padding 词元的交叉熵损失被当作正常损失计算，导致平均负对数似然被人为降低。

**修复：** 使用 `ignore_index=PAD_TOKEN_ID` 的交叉熵或手动 mask。

### 错误 2：精确匹配过度惩罚改写

**现象：** 正确的改写式回答被评为 0 分。

**原因：** 精确匹配要求字面相等，不接受语义等价的改写。

**修复：** 结合使用精确匹配和词元 F1——精确匹配衡量"完全正确"的比例，词元 F1 衡量"内容相关性"。

### 错误 3：评判器对长回答有偏见

**现象：** 较长的回答总是得分较高。

**原因：** mock 评判器基于词元重叠——更长的回答包含更多词元，天然有更高的重叠率。

**修复：** 在 mock 评判器中使用 F1（平衡精确率和召回率）而非单纯的召回率。

---

## 7. 面试考点

### Q1：为什么大语言模型的评测需要多种指标？（难度：⭐⭐）

**参考答案：** 每种指标衡量模型能力的不同方面：困惑度衡量模型对语言分布的拟合度（内部质量），精确匹配衡量事实正确性（外部准确性），词元 F1 衡量内容相关性（容忍改写），评判器衡量整体回答质量（综合能力）。没有任何单一指标能全面描述模型——一个困惑度很低的模型可能无法回答问题，一个精确匹配很好的模型可能回答得很啰嗦。

### Q2：如何设计一个公平的 hold-out 评测？（难度：⭐⭐⭐）

**参考答案：** 三个原则：(1) 评测数据必须与训练/验证数据完全不重叠（包括同一文章的不同段落）；(2) 评测数据必须覆盖目标分布的各种情况（简单/困难、长/短、不同领域）；(3) 评测指标必须可复现——固定随机种子、固定提示词模板、固定评测脚本版本。缺少任何一项，评测结果都不可信。

---

## 🔑 关键术语

| 术语 | 含义 |
|------|------|
| 困惑度 | $\exp(\text{平均负对数似然})$——衡量模型对下一个词元预测的不确定性 |
| 精确匹配 | 预测与参考在归一化后完全一致的比率 |
| 词元 F1 | 预测与参考在词元级别的精确率和召回率的调和平均 |
| Mock 评判器 | 使用规则（如词元重叠度）模拟 LLM 评判的本地评分器 |
| 加权聚合 | 多种指标按权重组合为单一总分 |

---

## 📚 小结

完整的评测管道使你能客观、可复现地评估模型能力。你实现了困惑度、精确匹配、词元 F1 和 mock 评判器四种异构评测，并将它们聚合为加权报告。这些指标是大语言模型评测的基石。

下一节将构建大语料库下载器——训练流水线的第一步。

---

## ✏️ 练习

1. 【理解】为什么困惑度的"正确计数"需要排除 padding 词元？如果不排除，会产生什么偏差？

2. 【实现】将 mock 评判器替换为基于 `rouge_l` 的评判器——使用最长公共子序列的 F1 分数作为评判依据。

3. 【实验】对同一模型在不同随机种子下运行评测管道 5 次。评测结果的方差有多大？哪个指标最稳定？

4. 【思考】在多轮对话评测中，如何设计评测指标来评估"上下文一致性"和"指令遵循度"？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 完整评测管道 | `code/main.py` | 困惑度+精确匹配+词元 F1+评判器 |
| 评测报告数据 | `outputs/report.json` | 聚合后的评测结果 |

---

## 📖 参考资料

1. [论文] Papineni et al. "BLEU: a Method for Automatic Evaluation of Machine Translation". ACL 2002. https://aclanthology.org/P02-1040/
2. [论文] Lin. "ROUGE: A Package for Automatic Evaluation of Summaries". ACL 2004. https://aclanthology.org/W04-1013/
3. [官方文档] HuggingFace Evaluate. https://huggingface.co/docs/evaluate/
4. [GitHub] EleutherAI lm-evaluation-harness. https://github.com/EleutherAI/lm-evaluation-harness
