# 数值稳定性——浮点数是个漏水的抽象

> 浮点数是个漏水的抽象。它会在训练中咬你，而你毫无察觉。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 01 阶段 · 01-04
**预计时间：** 120 分钟
**所处阶段：** Tier 1
**关联课程：** 第 03 阶段 · 05（反向传播）— 梯度检查是调试反向传播的金标准

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现数值稳定的 softmax 和 log-sum-exp（最大值减法技巧）
- [ ] 识别浮点计算中的溢出、下溢和灾难性抵消
- [ ] 用有限差分验证解析梯度
- [ ] 解释为什么训练优先使用 bfloat16 而非 float16

---

## 1. 问题

模型训练 3 小时后损失变 NaN。日志显示第 9000 步 logits 正常，第 9001 步变为 inf，第 9002 步所有梯度为 NaN，训练死亡。

或者：模型完成训练但准确率比论文低 2%。架构、超参、数据全部匹配。问题是你用了 float16 而论文用 float32——32 位累积舍入误差悄悄吞掉了你的准确率。

数值稳定性不是理论问题——它是训练成功与静默失败之间的差距。

---

## 2. 核心概念

### 2.1 IEEE 754 浮点格式

| 格式 | 位数 | 指数 | 尾数 | 约精确位数 | 范围 |
|:-----|:-----|:-----|:-----|:---------|:-----|
| float64 | 64 | 11 | 52 | ~15-16 | ±1.8e308 |
| float32 | 32 | 8 | 23 | ~7-8 | ±3.4e38 |
| float16 | 16 | 5 | 10 | ~3-4 | ±65,504 |
| bfloat16 | 16 | 8 | 7 | ~2-3 | ±3.4e38 |

### 2.2 为什么 0.1 + 0.2 ≠ 0.3

0.1 在二进制中是无限循环小数，float32 截断后存储为 ~0.100000001490116。修复：永远不要用 `==` 比较浮点数，用 `abs(a-b) < epsilon`。

### 2.3 灾难性抵消

减去两个几乎相等的浮点数导致有效数字抵消，相对误差可达 19%。在 ML 中发生在计算方差（`E[x²] - E[x]²`）时。

### 2.4 Log-Sum-Exp 技巧

```python
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

减去最大值后溢出消除——这是 softmax 数值稳定的基础。

### 2.5 混合精度训练

```
1. float32 主权重副本
2. 前向传播 fp16（快速）
3. 损失在 fp32（防溢出）
4. 反向传播 fp16
5. 损失缩放防止梯度下溢
6. 更新 fp32 主权重
```

---


## 3. 从零实现

### 第 1 步：稳定 softmax

核心思路：减去最大值后再 exp，消除溢出风险。

```python
import math

def softmax_stable(logits):
    """减去最大值后再 exp，避免溢出。"""
    mx = max(logits)
    exps = [math.exp(z - mx) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]
```

验证：`softmax_stable([100.0, 101.0, 102.0])` 不会溢出。

### 第 2 步：log-sum-exp 与交叉熵

log-sum-exp 是 softmax 的核心组件。交叉熵 = softmax + negative log likelihood。

```python
def logsumexp_stable(values):
    """log(sum(exp(x))) 的数值稳定版本。"""
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

def cross_entropy_stable(true_class, logits):
    """稳定交叉熵：softmax + negative log likelihood。"""
    mx = max(logits)
    shifted = [z - mx for z in logits]
    lse = math.log(sum(math.exp(s) for s in shifted))
    return -(shifted[true_class] - lse)
```

### 第 3 步：梯度检查

有限差分验证解析梯度——调试反向传播的金标准。

```python
def numerical_gradient(f, x, h=1e-5):
    """有限差分计算数值梯度。"""
    return [
        (f([xi + (h if j == i else 0) for j, xi in enumerate(x)]) -
         f([xi - (h if j == i else 0) for j, xi in enumerate(x)])) / (2 * h)
        for i, x in enumerate(x)
    ]

def check_gradient(analytical, numerical, tol=1e-5):
    """对比解析梯度与数值梯度。"""
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        err = abs(a - n) / denom
        status = "OK" if err < tol else "FAIL"
        print(f"  param{i}: a={a:.6f} n={n:.6f} err={err:.2e} [{status}]")

# 测试
f = lambda p: p[0]**2 + 3*p[0]*p[1] + p[1]**3
g = lambda p: [2*p[0]+3*p[1], 3*p[0]+3*p[1]**2]
check_gradient(g([2.0, 1.0]), numerical_gradient(f, [2.0, 1.0]))
```

---

## 4. 工业工具

| 工具 | 用途 |
|:-----|:-----|
| `torch.autocast` | 自动混合精度前向传播 |
| `torch.amp.GradScaler` | 动态损失缩放，防止梯度下溢 |
| `torch.autograd.gradcheck` | 自动化梯度正确性验证 |
| `numpy.finfo(float32).eps` | 获取浮点数机器精度 |

---

## 5. 知识连线

- **第 03 阶段 · 05（反向传播）**：梯度检查是调试反向传播的金标准
- **第 03 阶段 · 10（神经网络框架）**：混合精度训练是大规模训练的标准技术
- **第 10 阶段 · 04（GPT 训练）**：BF16 混合精度在 LLM 训练中不可或缺

---

## 6. 工程最佳实践

- **永远不要用 `==` 比较浮点数**：用 `abs(a-b) < eps` 或 `math.isclose()`
- **实现新操作时做梯度检查**：用 `gradcheck` 验证反向传播正确性
- **BF16 优先于 FP16**：训练中范围比精度更重要，BF16 无需损失缩放
- **中文场景建议**：大模型训练中，BF16 + 梯度累积 + 梯度裁剪是标准三件套

---

## 7. 常见错误

### 错误 1：softmax 未减最大值

**现象：** `exp(100)` 溢出为 `inf`，梯度变为 NaN。

**修复：** 始终使用 `torch.nn.functional.softmax()` 或手动减最大值。

### 错误 2：损失缩放因子不当

**现象：** 梯度全部下溢为零，模型停止学习。

**修复：** 使用动态损失缩放（`GradScaler`），自动调整缩放因子。

### 错误 3：手动反向传播漏写梯度累加

**现象：** `grad += ...` 应该用累加而非赋值。

**修复：** 多输入梯度用 `+=`，单输入用 `=`。

---

## 8. 面试考点

### Q1：BF16 比 FP16 更适合训练的原因？（难度：⭐⭐）

**参考答案：** BF16 拥有与 FP32 相同的 8 位指数范围（±3.4e38），训练时激活值和梯度经常超过 FP16 的 65504 上限。BF16 不会溢出，且不需要损失缩放。BF16 只是截断 FP32 的尾数 16 位，转换几乎无损。

---

## 🔑 关键术语

| 术语 | 含义 |
|:-----|:-----|
| IEEE 754 | 浮点数标准 |
| 机器精度 | 1 + ε ≠ 1 的最小 ε（FP32 ≈ 1.19e-7） |
| 灾难性抵消 | 减去两个相近数导致精度丢失 |
| 梯度检查 | 数值有限差分验证解析梯度 |
| 损失缩放 | 反向传播前放大损失避免梯度下溢 |
| 混合精度 | 低精度计算、高精度参数更新 |
| 稳定 softmax | 减最大值后 exp，无溢出 |

---

## 📚 小结

数值稳定性是 AI 工程的隐形杀手。你实现了稳定 softmax、log-sum-exp 和梯度检查，理解了 FP16/BF16 的权衡。掌握这些可以避免训练中 80% 的 NaN 问题。

---

## ✏️ 练习

1. 【实验】对比 `softmax_naive` 和 `softmax_stable` 在大 logit 上的表现
2. 【实现】实现 Welford 在线算法计算方差
3. 【实验】测试梯度检查在不同 h 值下的数值稳定性

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| 数值稳定性工具 | `code/stability.py` | 稳定 softmax、梯度检查 |

---

## 📖 参考资料

1. [论文] Micikevicius et al. "Mixed Precision Training". https://arxiv.org/abs/1710.03740
2. [官方文档] PyTorch `torch.autocast`. https://pytorch.org/docs/stable/amp.html
3. [文章] Goldberg. "What Every Computer Scientist Should Know About Floating-Point Arithmetic". 1991
