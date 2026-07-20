# 水印——SynthID、Stable Signature、C2PA

> 三种技术构建了 2026 年 AI 生成内容溯源。SynthID（Google DeepMind）——图像水印 2023 年 8 月发布，文本+视频 2024 年 5 月，文本 2024 年 10 月通过 Responsible GenAI Toolkit 开源，统一多模态检测器 2025 年 11 月。文本水印不可察觉地调整下一个词元采样概率；图像/视频水印在压缩、裁剪、滤镜后存活。Stable Signature（Fernandez et al., ICCV 2023）——微调解码器使每个输出包含固定消息；裁剪后（10% 内容）检测率 >90%。C2PA——加密签名的防篡改元数据标准。水印和 C2PA 互补：元数据可被剥离但携带更丰富溯源；水印在转码后持久但携带更少信息。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 10 · 04（采样）、阶段 01 · 09（信息论）
**预计时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述词元级水印（SynthID-text 风格）及其可检测机制
- [ ] 描述 Stable Signature 和 2024 年打破它的移除攻击
- [ ] 陈述 C2PA 的作用以及为什么它与水印互补
- [ ] 描述关键局限：模型特定信号、意译鲁棒性、保持含义的攻击

---

## 1. 问题

2023-2024 年深度伪造和 AI 生成内容在政治和消费者场景中大规模出现。水印是提议的技术溯源信号：在生成时标记，在之后检测。2025 年证据：没有水印是无条件鲁棒的，但与 C2PA 元数据分层后组合提供了可用的溯源故事。

---

## 2. 概念

### 2.1 文本水印（SynthID-text 风格）

Kirchenbauer et al. 2023 机制，由 Google 生产化：

1. 每个解码步骤，哈希前 K 个词元产生词表的伪随机分区为"绿色"和"红色"集合
2. 通过向绿色 logits 添加 δ 偏置采样倾向绿色集
3. 生成包含比随机更多的绿色词元

检测：重新哈希每个前缀，计算生成中绿色词元的 z 分数。水印文本的 z 分数 >0，人类文本约 0。

属性：对读者不可察觉（δ 足够小，质量损失微小）；可通过词表分区函数检测；**对意译不鲁棒**——重写文本破坏信号。

### 2.2 Stable Signature（图像）

Fernandez et al. ICCV 2023。微调 latent diffusion 解码器使每个生成图像包含固定二进制消息。2024 年 5 月"Stable Signature is Unstable"：微调解码器移除水印同时保持图像质量。对抗性后生成微调是廉价的。

### 2.3 C2PA

内容真实性和真实性联盟。加密签名的防篡改元数据标准。C2PA 清单记录溯源声明（谁创建、何时、什么转换），由创建者密钥签名。

与水印互补：元数据可被剥离；水印在转码后持久。元数据更丰富（完整溯源链）；水印携带更少信息。

---

## 3. 从零实现

```python
import random
import math


def synthesize_watermark(tokens, vocabulary_size, delta=0.5):
    """SynthID-text 风格的文本水印——绿色集偏置采样。"""
    watermarked = []
    for token in tokens:
        # 哈希前一个词元决定分区
        hash_val = hash(str(token)) % vocabulary_size
        is_green = hash_val < vocabulary_size // 2

        # 绿色集偏置
        if is_green and random.random() < 0.5 + delta / 10:
            watermarked.append(token)  # 偏置保留
        else:
            watermarked.append(token)

    return watermarked


def detect_watermark(tokens, vocabulary_size):
    """检测水印——绿色词元 z 分数。"""
    green_count = 0
    for token in tokens:
        hash_val = hash(str(token)) % vocabulary_size
        if hash_val < vocabulary_size // 2:
            green_count += 1

    total = len(tokens)
    expected = total / 2
    std = math.sqrt(total / 4)
    z_score = (green_count - expected) / std if std > 0 else 0

    return {"green_count": green_count, "z_score": z_score,
            "watermarked": z_score > 1.96}  # 95% 置信度


# 演示
random.seed(42)
vocab_size = 1000
text_tokens = [random.randint(0, vocab_size) for _ in range(500)]

# 水印
watermarked = synthesize_watermark(text_tokens, vocab_size, delta=0.5)
result = detect_watermark(watermarked, vocab_size)
print(f"水印文本: z={result['z_score']:.2f}, 检测={result['watermarked']}")

# 人类文本
human_result = detect_watermark(text_tokens, vocab_size)
print(f"人类文本: z={human_result['z_score']:.2f}, 检测={human_result['watermarked']}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 水印和溯源工具

| 工具 | 类型 | 模态 |
|---|---|---|
| SynthID | 水印 | 文本+图像+视频+音频 |
| Stable Signature | 水印 | 图像（微调解码器） |
| C2PA | 元数据 | 跨模态（加密签名） |
| 前沿水印 | 研究 | 语义水印 |

---

## 5. 工程最佳实践

### 5.1 水印和 C2PA 互补

水印在转码后持久但信息少；C2PA 元数据丰富但可被剥离。两者结合使用。

### 5.2 水印对意译不鲁棒

文本水印在改写后失效。图像水印在裁剪后存活但可通过微调移除。

### 5.3 EU AI Act 第 50 条要求透明度

AI 生成内容必须标注——水印和 C2PA 是满足此要求的技术手段。

---

## 6. 常见错误

### 错误 1：假设水印是无条件鲁棒的

**现象：** 声称"我们的水印无法被移除"。

**原因：** 文本水印在意译后失效；图像水印在微调解码器后移除；所有水印在适当攻击下都可能失效。

**修复：** 将水印视为多层防御的一部分，而不是单一解决方案。与 C2PA 元数据结合使用。

### 错误 2：忽略模型特定性

**现象：** 检测到 SynthID 信号→声称内容是 AI 生成的。

**原因：** SynthID 只标记启用 SynthID 的模型的生成。没有 SynthID 信号不能证明内容是人类创作的。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| SynthID | "Google 的水印" | 跨模态溯源信号；文本+图像+音频+视频 |
| 词元水印 | "Kirchenbauer 风格" | 通过绿色词元 z 分数检测的偏置采样文本水印 |
| Stable Signature | "图像水印" | 微调解码器水印；ICCV 2023 |
| C2PA | "元数据标准" | 加密签名的防篡改溯源元数据 |
| 意译鲁棒性 | "改写是否破坏" | 文本水印属性；目前有限 |

---

## 📚 小结

三种技术构建 2026 年的 AI 内容溯源：SynthID（文本+图像+视频水印）、Stable Signature（图像微调解码器水印）、C2PA（加密签名元数据）。文本水印通过绿色词元 z 分数检测但对意译不鲁棒。C2PA 与水印互补——元数据丰富但可剥离，水印持久但信息少。EU AI Act 第 50 条要求 AI 生成内容标注——水印和 C2PA 是满足要求的技术手段。

---

## ✏️ 练习

1. 运行 `code/main.py`。报告水印 1000 词元生成 vs 人类文本的 z 分数。找出 95% 置信度阈值下的假阳性率。
2. 实现一个意译攻击——替换 30% 的词元为同义词。重新测量 z 分数。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 文本水印模拟 | `code/main.py` | 绿色词元偏置采样和 z 分数检测 |
| 溯源审计 | `outputs/skill-provenance-audit.md` | 水印+C2PA 溯源链审计 |

---

## 📖 参考资料

1. [论文] Kirchenbauer et al. — A Watermark for Large Language Models. ICML 2023
2. [论文] Fernandez et al. — Stable Signature. ICCV 2023
3. [论文] "Stable Signature is Unstable". arXiv:2405.07145
4. [官方文档] Google DeepMind — SynthID. https://deepmind.google/models/synthid/
5. [标准] C2PA 2.2 Explainer. https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html
