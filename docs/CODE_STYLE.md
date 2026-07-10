# 代码风格 (Code Style)

> **规则：** 所有代码示例必须遵循本指南。代码的第一读者是人，第二读者才是机器。

---

## 1. 总体原则

### 1.1 代码的三条铁律

1. **必须可以运行** — 读者复制粘贴后不报错
2. **必须易于理解** — 变量名清晰、逻辑平铺、注释精准
3. **必须最小化** — 不写与知识点无关的代码

### 1.2 代码的定位

本项目中的代码是**教学工具**，不是生产级代码。权衡标准：

| | 教学代码优先 | 生产代码优先 |
|---|---|---|
| 依赖数量 | 少（标准库 + numpy + torch） | 可以有几十个 |
| 错误处理 | 基本断言即可 | 完善的 try/except |
| 抽象层次 | 展开，平铺直叙 | 封装，复用 |
| 性能 | 够用就行 | 极致优化 |
| 配置管理 | 硬编码 | 配置文件/环境变量 |

**当教学清晰度和生产规范冲突时，选教学清晰度。**

---

## 2. 语言与依赖

### 2.1 主要语言

- **Python 3.10+** 为默认语言
- 其他语言（TypeScript、Rust）仅在有明确理由时使用

### 2.2 依赖管理

每课在代码文件顶部以注释形式标注依赖：

```python
# 依赖：torch>=2.0, transformers>=4.30, numpy
# 安装：pip install torch transformers numpy
```

规则：
- 优先标准库，其次 PyTorch 生态，避免引入冷门第三方库
- 每个文件最多导入 5 个不同库
- 必须标注最低版本（如果对版本有要求）

### 2.3 标准依赖清单

按课程阶段，常用依赖如下：

| 阶段 | 推荐依赖 |
|---|---|
| 基础课 | `numpy`、`scipy`、`matplotlib` |
| 深度学习 | `torch`、`torchvision` |
| NLP/Transformer | `transformers`、`datasets`、`tokenizers` |
| LLM 工程 | `tiktoken`、`sentencepiece`、`langchain`、`vllm` |
| 其他 | `pandas`、`scikit-learn` |

---

## 3. 命名规范

### 3.1 Python 命名

| 类型 | 风格 | 示例 |
|---|---|---|
| 变量 | `snake_case` | `num_layers`、`learning_rate`、`token_ids` |
| 函数 | `snake_case` | `scaled_dot_product_attention()`、`train_one_epoch()` |
| 类 | `PascalCase` | `SelfAttention`、`BPETokenizer`、`TransformerBlock` |
| 常量 | `UPPER_SNAKE_CASE` | `MAX_SEQ_LENGTH`、`PAD_TOKEN_ID` |
| 模块/文件 | `snake_case` | `bpe_tokenizer.py`、`attention_viz.py` |
| 私有成员 | `_leading_underscore` | `_get_pairs()`、`_merge_pair()` |

### 3.2 变量名要求

**变量名即注释。** 看名字就知道它是干什么的。

```python
# ❌ 模糊命名
x = model(data)
y = x[0]
z = y.sum()

# ✓ 清晰命名
logits = model(input_ids)
first_token_logits = logits[0]
total_prob = first_token_logits.sum()

# ❌ 单字母（除非是数学公式的直接映射）
a = torch.matmul(q, k.transpose(-2, -1))

# ✓ 数学公式中，单字母对应公式符号是可接受的
# Attention(Q, K, V) = softmax(Q @ K^T / √dk) @ V
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
```

**特殊允许的单字母变量：**
- 数学公式中直接对应的符号：`Q`、`K`、`V`、`X`、`Y`
- 循环索引：`i`、`j`、`k`（仅在循环体中不超过 3 行时）
- 维度占位符：`n`、`d`、`h`（如 `n_tokens`、`d_model`、`n_heads`）
- 临时变量/类型变量：`x`、`y`、`T`（仅在泛型上下文中）

### 3.3 中文场景变量名

涉及中文数据时，变量名可以包含拼音，但要确保含义明确：

```python
# ✓ 可接受
chinese_texts = ["我爱北京天安门", "人工智能很有趣"]
pinyin_tokens = tokenizer.encode(chinese_texts)

# ❌ 不推荐：英文变量名与中文内容语义割裂
sentences = ["我爱北京天安门"]  # 这不是 "sentences"，这是中文
```

---

## 4. 注释规范

### 4.1 注释语言

**注释使用中文。** 代码本身（变量名、函数名、类名）使用英文。

```python
# ✓ 正确
def compute_attention_scores(query, key):
    """计算缩放点积注意力分数。"""
    d_k = query.size(-1)
    # 缩放因子 √d_k 防止点积过大导致 softmax 梯度消失
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    return scores
```

### 4.2 注释层级

```python
# === 块注释：用于分隔代码块 ===
# 这类注释独占一行，前后留空行，用于标记"下面开始一个新的逻辑块"


def train_step(model, batch):
    """函数文档字符串：描述做什么、参数含义、返回值。"""
    
    # 行注释：解释"为什么"这样做，不是"是什么"
    # 除以 √d_k 是为了防止点积过大
    scores = Q @ K.T / math.sqrt(d_k)
    
    x = x + 1  # 这种显而易见的操作不需要注释
```

**注释黄金法则：** 解释"为什么"（Why），不解释"是什么"（What）。

```python
# ❌ 无意义注释（重复代码本身）
x = x + 1  # 将 x 加 1

# ✓ 有用注释（解释原因）
x = x + 1  # tokens 从 1 开始编号，0 留给 padding
```

### 4.3 函数文档字符串

对核心函数写文档字符串。格式使用 Google 风格：

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """计算缩放点积注意力。
    
    这是 Transformer 的核心操作。每个查询与所有键计算相似度，
    softmax 归一化后与值矩阵加权求和。
    
    Args:
        Q: 查询矩阵，形状 (batch, seq_len, d_k)
        K: 键矩阵，形状 (batch, seq_len, d_k)
        V: 值矩阵，形状 (batch, seq_len, d_v)
        mask: 可选。注意力掩码，形状 (batch, seq_len, seq_len)
              为 True 的位置将被设为 -inf
    
    Returns:
        output: 注意力输出，形状 (batch, seq_len, d_v)
        weights: 注意力权重，形状 (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights
```

**何时写文档字符串：**
- 模块级函数：必写
- 类：必写
- 类方法：核心方法写，简单的 `__init__` 可以不写
- 私有函数 `_helper()`：可以不写（如果逻辑足够清晰）

---

## 5. 代码结构

### 5.1 从零实现的设计模式

教学代码遵循"递进式"结构。每一步是一个可以独立运行的单元格/代码块：

```python
# === 第 1 步：最简版本（6 行） ===
def attention_basic(Q, K, V):
    scores = Q @ K.T
    weights = softmax(scores)
    return weights @ V

# === 第 2 步：加入缩放因子 ===
def attention_scaled(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # 新增：防止梯度消失
    weights = softmax(scores)
    return weights @ V

# === 第 3 步：加入可学习参数 ===
class SelfAttention:
    def __init__(self, d_model, d_k):
        self.Wq = np.random.randn(d_model, d_k) * 0.02
        self.Wk = np.random.randn(d_model, d_k) * 0.02
        self.Wv = np.random.randn(d_model, d_k) * 0.02
    
    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        return attention_scaled(Q, K, V)
```

每步只增加一个概念。

### 5.2 代码块大小

- 单个函数不超过 30 行
- 单个代码块（``` 之间）不超过 50 行
- 超过则拆分，并在拆分处加注释说明

### 5.3 输出展示

每次运行后展示输出，帮助读者验证自己的结果：

```python
# 运行
output, weights = attention.forward(X)

print(f"输入形状: {X.shape}")      # 输入形状: (6, 8)
print(f"输出形状: {output.shape}")  # 输出形状: (6, 4)
print(f"注意力权重和: {weights.sum(axis=-1)}")  # 每行均为 1.0
```

输出展示使用 `print()` 并在注释中标注预期结果或直接在代码后用 `text` 代码块展示：

```text
输入形状: (6, 8)
输出形状: (6, 4)
注意力权重和: [1.0 1.0 1.0 1.0 1.0 1.0]
```

---

## 6. 代码组织

### 6.1 文件结构

```python
# === 文件头注释 ===
# bpe_tokenizer.py — 从零实现 BPE 分词器
# 依赖：numpy>=1.24
# 对应课程：阶段 10 · 01（分词器）

# === 导入 ===
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict

# === 常量 ===
BASE_VOCAB_SIZE = 256
UNKNOWN_TOKEN = "<unk>"

# === 类/函数定义 ===
class BPETokenizer:
    """BPE 分词器的教学实现。"""
    
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
    
    def train(self, text: str, num_merges: int) -> "BPETokenizer":
        """在文本上训练 BPE 分词器。"""
        ...

# === 主程序 ===
if __name__ == "__main__":
    # 演示用示例
    corpus = "这是一个测试用的中文语料。"
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=10)
```

### 6.2 导入顺序

```python
# 1. 标准库
import math
import os
from collections import Counter
from typing import List, Optional

# 2. 第三方库（按字母序）
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
```

**不使用 `import *`：**

```python
# ❌ 禁止
from torch import *
from numpy import *

# ✓ 正确
import torch
import torch.nn.functional as F
from torch import Tensor
```

---

## 7. 类型提示

### 7.1 推荐但不强制

教学代码中类型提示有助于读者理解数据流，但不强制要求：

```python
# ✓ 推荐（有类型提示）
def encode(self, text: str) -> List[int]:
    tokens = list(text.encode("utf-8"))
    for pair, new_id in self.merges.items():
        tokens = self._merge_pair(tokens, pair, new_id)
    return tokens

# ✓ 可接受（简单函数可以不加）
def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
```

### 7.2 使用原则

- 公开 API 函数（会被读者直接调用的）：加类型提示
- 内部辅助函数（`_merge_pair` 等）：可以不加
- 返回值类型始终标注（如果加了参数类型）

---

## 8. 错误处理

教学代码中**不写生产级异常处理**。用简单的 `assert` 和注释说明预期行为即可。

```python
# ✓ 教学风格
def attention(Q, K, V):
    assert Q.shape[-1] == K.shape[-1], "Q 和 K 的最后一维必须相等（都是 d_k）"
    d_k = Q.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d_k)) @ V

# ❌ 过度工程（教学代码不需要）
def attention(Q, K, V):
    try:
        d_k = Q.shape[-1]
    except IndexError:
        raise ValueError("输入张量维度不足")
    if not isinstance(Q, torch.Tensor):
        raise TypeError("Q 必须是 torch.Tensor")
    # ...
```

---

## 9. 输出与可视化

### 9.1 print() 格式

```python
# ✓ 使用 f-string，对齐输出
print(f"{'词元':<10} {'ID':>6} {'频率':>8}")
print("-" * 26)
for token, tid, freq in token_stats:
    print(f"{token:<10} {tid:>6} {freq:>8}")

# ✓ 重要结论单独一行，突出显示
print(f"\n压缩率: {len(tokens) / len(text_bytes):.2%}")
print(f"词表中有 {unused_count} 个词元从未被使用")
```

### 9.2 简单可视化

优先使用 ASCII 和纯文本可视化：

```python
# ASCII 热力图
def ascii_heatmap(matrix, labels, chars=" ░▒▓█"):
    """用 ASCII 字符可视化注意力权重矩阵。"""
    for i, label_i in enumerate(labels):
        row = "".join(
            chars[int(w * (len(chars) - 1))]
            for w in matrix[i]
        )
        print(f"{label_i:>6} |{row}|")
```

---

## 10. 必须避免的写法

### 10.1 过度简化的玩具代码

```python
# ❌ 太简化，学不到东西
x = torch.randn(10)
model = nn.Linear(10, 1)(x)
# "就这样！" — 没有解释任何东西

# ✓ 展示完整流程
x = torch.randn(1, 10)       # 1 个样本，10 维特征
layer = nn.Linear(10, 1)     # 线性层：10 输入 → 1 输出
output = layer(x)             # 前向传播
print(f"权重形状: {layer.weight.shape}")  # 权重形状: (1, 10)
print(f"输出: {output.item():.4f}")
```

### 10.2 无意义的缩写

```python
# ❌
def sa(q, k, v, m=None):
    sc = q @ k.T / np.sqrt(q.shape[-1])
    ...

# ✓
def self_attention(query, key, value, mask=None):
    scores = query @ key.T / np.sqrt(query.shape[-1])
    ...
```

### 10.3 魔法数字

```python
# ❌
embeddings = torch.randn(6, 512)
attention = SelfAttention(512, 64, 64)

# ✓
seq_len = 6
d_model = 512
d_k = 64
d_v = 64
embeddings = torch.randn(seq_len, d_model)
attention = SelfAttention(d_model, d_k, d_v)
```

### 10.4 复制粘贴的重复代码

如果同一逻辑出现两次以上，提取为函数：

```python
# ❌
# 第一次
scores_1 = q1 @ k1.T / np.sqrt(q1.shape[-1])
# 第二次（复制粘贴）
scores_2 = q2 @ k2.T / np.sqrt(q2.shape[-1])

# ✓
def scaled_scores(query, key):
    return query @ key.T / np.sqrt(query.shape[-1])

scores_1 = scaled_scores(q1, k1)
scores_2 = scaled_scores(q2, k2)
```

### 10.5 未定义就使用的变量

```python
# ❌ 读者不知道 MAX_LENGTH 从哪来的
tokens = tokens[:MAX_LENGTH]

# ✓ 要么定义，要么注释来源
MAX_LENGTH = 512  # GPT-2 的训练上下文长度
tokens = tokens[:MAX_LENGTH]
```

---

## 11. 程序化检查

### 11.1 交付前必做

```bash
# 1. 确认代码可以运行
python code/main.py

# 2. 确认没有未使用的导入
python -m py_compile code/main.py

# 3. 查看是否有明显的语法问题
python -c "import ast; ast.parse(open('code/main.py').read())"
```

---

## 12. 自检清单

写完代码后，检查：

- [ ] 代码可以直接运行（复制粘贴到新环境不出错）
- [ ] 变量名清晰，不需要注释就能理解
- [ ] 注释使用中文，解释"为什么"而不是"是什么"
- [ ] 每个代码块不超过 50 行
- [ ] 每个函数不超过 30 行
- [ ] 没有 `import *`
- [ ] 没有魔法数字
- [ ] 术语使用符合 `TERMINOLOGY.md`
- [ ] 文件头有注释说明依赖和对应的课程
- [ ] 导入了但没有使用的库已删除
- [ ] 输出有预期结果标注（在注释中或单独展示）
- [ ] 没有过度工程（教学代码不需要完整的异常处理）
