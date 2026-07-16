# 长视频理解：百万词元上下文

> 1 小时 4K 视频在 24 FPS 下切块嵌入产生约 6000 万个词元。Google 的 Gemini 1.5（2024 年 3 月）以 1000 万词元上下文开启了这个时代。LWM 展示了环形注意力的扩展路径。LongVILA 和 Video-XL 进一步扩展了摄取能力。VideoAgent 用智能检索替代了原始上下文。每种方法都是计算、召回和工程复杂度之间的不同权衡。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 17（视频时序词元）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 计算长视频在不同 FPS 和池化策略下的总视觉词元数
- [ ] 对比不同长视频理解方案——环形注意力、VideoAgent、TokenPacker
- [ ] 解释"大海捞针"测试如何评估长上下文能力
- [ ] 设计长视频处理策略——平衡计算成本和召回率

---

## 1. 问题

标准 VLM 的上下文窗口是 4K-8K 词元。但一段 1 小时的视频在 1 FPS 下就有 3600 帧——即使每帧只有 100 个词元也是 360,000 个。如何在有限的上下文窗口中处理如此多的信息？

---

## 2. 概念

### 2.1 长视频的词元预算

```
1 小时视频 @ 1 FPS = 3600 帧
每帧 100 词元（10×10 图块，64 维嵌入）
总词元 = 360,000

解决方案：
1. 降低 FPS（0.5-2 FPS）→ 1800-7200 帧
2. 帧级池化（每帧 100→10 词元）→ 18,000-72,000 词元
3. 长上下文模型（100K-1M 词元）
```

### 2.2 四种长视频方案

| 方案 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| **长上下文** | 扩展 Transformer 窗口 | 简单 | 计算 O(n²) |
| **环形注意力** | 将序列切分为环，跨环注意 | 线性扩展 | 实现复杂 |
| **TokenPacker** | 压缩词元数量 | 减少计算 | 丢失细节 |
| **VideoAgent** | 智能检索 | 最省计算 | 依赖检索质量 |

### 2.3 大海捞针测试

测试长上下文的召回能力——在长序列中插入一个特定信息（"针"），看模型能否在大量干扰信息（"草"）中找到它。

---

## 3. 从零实现

### Step 1：词元预算计算

```python
def compute_video_token_budget(duration_sec, fps=1, patches_per_frame=100):
    """计算视频的总词元数。"""
    num_frames = int(duration_sec * fps)
    total_tokens = num_frames * patches_per_frame
    return {
        "duration_sec": duration_sec,
        "fps": fps,
        "num_frames": num_frames,
        "total_tokens": total_tokens,
        "context_needed": f"需要 {total_tokens} 词元上下文",
    }
```

### Step 2：TokenPacker 压缩

```python
def token_packer_compress(video_tokens, target_tokens=1024):
    """TokenPacker 压缩——将视频词元压缩到目标数量。"""
    num_frames = video_tokens.shape[0]
    tokens_per_frame = video_tokens.shape[1]
    total = num_frames * tokens_per_frame
    if total <= target_tokens:
        return video_tokens  # 不需要压缩

    # 自适应压缩：均匀采样帧，每帧取平均
    ratio = target_tokens / total
    sampled_frames = int(num_frames * ratio)
    indices = np.linspace(0, num_frames - 1, sampled_frames, dtype=int)
    compressed = video_tokens[indices]
    return compressed
```

### Step 3：大海捞针模拟

```python
def needle_in_haystack_test(model, video_tokens, needle_position, needle_text):
    """大海捞针测试——检查模型是否能找到特定信息。"""
    # 在指定位置插入"针"
    # 用模型处理完整序列
    # 检查输出中是否包含"针"的信息
    return {
        "position": needle_position,
        "needle_text": needle_text,
        "found": False,  # 模拟结果
    }
```

---

## 4. 工具

### 4.1 Google Gemini 1.5

Gemini 1.5 Pro 支持 1000 万词元上下文——可以处理完整的长视频。但需要特定的 API 访问。

### 4.2 长上下文评估框架

```python
# 大海捞针测试
def run_needle_in_haystack(model, video_tokens, positions):
    results = []
    for pos in positions:
        result = needle_in_haystack_test(model, video_tokens, pos, "test")
        results.append(result)
    return results
```

---

## 6. 工程最佳实践

### 6.1 长视频处理策略

| 视频长度 | 策略 | 词元预算 |
|---------|------|---------|
| <30s | 直接处理 | 10K-50K |
| 30s-5min | 降 FPS + 帧池化 | 50K-200K |
| 5min-1h | TokenPacker + 长上下文 | 200K-1M |
| >1h | VideoAgent 智能检索 | 动态 |

### 6.2 踩坑经验

- **上下文太长导致注意力崩溃**：使用滑动窗口或分段处理
- **帧率过高导致信息冗余**：降 FPS 通常不会损失关键信息

---

## 7. 常见错误

### 错误 1：不降 FPS 直接处理长视频

**现象：** 显存溢出或推理时间爆炸。

**修复：** 先降 FPS 到 1-2，再用池化进一步压缩。

### 错误 2：大海捞针测试位置不当

**现象：** 位置在序列开头/结尾——这些位置注意力权重最高，测试没有挑战性。

**修复：** 在序列中间 10%-90% 位置放置"针"。

---

## 8. 面试考点

### Q1：环形注意力为什么比标准注意力更高效？（难度：⭐⭐⭐）

**参考答案：**
标准注意力的复杂度是 O(n²)——每个词元关注所有其他词元。环形注意力将序列切分为环，每个环内的词元互相注意，但跨环只通过少量"桥接"词元通信。这将复杂度降低到 O(n·环大小)——对百万词元序列是必要的。环形注意力保留了全局信息的传递，同时避免了 O(n²) 的显存爆炸。

### Q2：VideoAgent 和直接长上下文处理有什么区别？（难度：⭐⭐）

**参考答案：**
直接长上下文处理将所有帧放入上下文窗口——计算和显存都随视频长度增长。VideoAgent 使用"智能检索"——先用小模型识别关键帧，只将关键帧送入长上下文模型。这大幅减少了实际处理的词元数量，但依赖检索的准确性。VideoAgent 更适合超长视频（如完整电影），直接长上下文更适合需要全局信息的任务。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 长上下文 | "能看很长的视频" | Transformer 上下文窗口从 4K 扩展到 100K-1M 词元 |
| TokenPacker | "视频词元压缩" | 减少每帧的词元数——在固定预算内平衡分辨率和帧数 |
| 环形注意力 | "分段注意力" | 将序列切分为环，环内完全注意，跨环部分注意——O(n) 复杂度 |
| 大海捞针 | "长上下文召回测试" | 在长序列中插入特定信息，测试模型能否找到 |

---

## 📚 小结

长视频需要百万词元上下文。四种方案：长上下文扩展、环形注意力、TokenPacker 压缩、VideoAgent 智能检索。大海捞针测试评估长上下文召回能力。2026 年 Gemini 1.5 支持 1000 万词元——使完整视频理解成为可能。

---

## ✏️ 练习

1. **【计算】** 计算 1 小时视频在 1 FPS、2 FPS、4 FPS 下的总词元数
2. **【设计】** 为一段 30 分钟的视频设计处理策略——选择 FPS、池化方法、上下文方案

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 词元预算计算 | `code/main.py` | 视频词元预算计算 + TokenPacker 压缩 |

---

## 📖 参考资料

1. [论文] Gemini Team. "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context". arXiv, 2024.
2. [论文] Liu et al. "LWM: Large World Model -- Scaling Autoregressive Multi-Modal Models". arXiv, 2024.
3. [论文] Fu et al. "LongVILA: Scaling Long-Context Visual Language Models for Long Videos". arXiv, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
