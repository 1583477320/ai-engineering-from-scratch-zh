# 视频-语言模型：时序词元与定位

> 视频不是一堆照片的堆叠。5 秒片段有因果顺序、动作动词和事件时间——图像模型无法表示这些。Video-LLaMA（2023 年 6 月）发布了第一个开源视频 VLM，具有音频-视觉定位。VideoChat 和 Video-LLaVA 扩展了这个模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 与前沿专有模型拉平了差距。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 08（LLaVA-OneVision）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释为什么时序位置编码独立于视觉编码器改变视频 VLM 性能
- [ ] 实现动态帧采样器——在固定视觉预算内平衡帧数和分辨率
- [ ] 构建时序定位查询——给定时间戳定位视频中的事件
- [ ] 对比不同的时序建模方案——Q-Former、拼接池化、TMRoPE

---

## 1. 问题

图像 VLM 处理单帧。但视频需要：(1) 帧间时间关系（动作顺序）；(2) 事件定位（"在第 3 秒发生了什么"）；(3) 固定视觉预算下的帧数-分辨率权衡。

核心挑战：如何在一个固定长度的视觉词元预算内，平衡空间细节（高分辨率）和时间覆盖（多帧数）？

---

## 2. 概念

### 2.1 时序位置编码

三种处理视频时序信息的方案：

| 方案 | 方法 | 代表 |
|------|------|------|
| Q-Former per clip | 每个视频片段一个 Q-Former | Video-LLaMA |
| 拼接池化 | 帧嵌入直接拼接 | Video-LLaVA |
| TMRoPE | 旋转位置编码（时间+空间） | Qwen2.5-VL |

### 2.2 动态帧采样

固定预算下，帧数和分辨率的权衡：

```
预算 = 288 个词元

高分辨率: 384×384 × 1 帧 = 288 词元 → 高空间细节
中分辨率: 256×256 × 2 帧 = 128×2 = 256 词元 → 平衡
低分辨率: 128×128 × 4 帧 = 64×4 = 256 词元 → 高时间覆盖
```

### 2.3 时序定位

给定自然语言查询，定位视频中的时间区间：
- 查询："猫跳起来的时间段"
- 输出：[起始帧, 结束帧]

---

## 3. 从零实现

### Step 1：动态帧采样器

```python
def dynamic_frame_sampler(video_frames, total_budget=288, frame_size=16):
    """动态帧采样——在固定预算内平衡帧数和分辨率。"""
    num_frames = len(video_frames)
    max_patches = total_budget // (frame_size * frame_size)

    # 尝试不同帧数
    for n_frames in range(num_frames, 0, -1):
        patches_per_frame = max_patches // n_frames
        if patches_per_frame >= 1:
            return {"num_frames": n_frames, "patches_per_frame": patches_per_frame}

    return {"num_frames": 1, "patches_per_frame": max_patches}
```

### Step 2：时序定位查询

```python
def temporal_grounding_query(query_text, frame_timestamps):
    """构建时序定位查询。"""
    return {
        "query": query_text,
        "timestamps": frame_timestamps,
        "target": "temporal_localization"
    }
```

---

## 4. 工具

### 4.1 HuggingFace

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
# Qwen2-VL 支持视频理解
```

---

## 6. 工程最佳实践

### 6.1 视频采样策略

| 视频长度 | 推荐帧数 | 分辨率 | 说明 |
|---------|---------|--------|------|
| <5s | 8-16 | 中 | 保留关键帧 |
| 5-30s | 4-8 | 低 | 平衡时间覆盖 |
| >30s | 4-8 | 低 | 关键帧采样 |

---

## 7. 常见错误

### 错误 1：忽略视频的时间信息

**现象：** 将视频帧独立处理——失去时序理解。

**修复：** 使用时序位置编码——确保帧间关系被建模。

---

## 8. 面试考点

### Q1：为什么时序位置编码对视频 VLM 性能很重要？（难度：⭐⭐）

**参考答案：**
没有时序编码，模型无法区分同一帧在视频不同时间点的含义。时序位置编码让模型知道每个词元的"时间戳"——帧 3 和帧 30 的同一物体应该有不同的时间位置。TMRoPE（Qwen2-VL）通过旋转编码同时编码时间和空间位置，使模型能正确理解视频的时序结构。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 时序位置编码 | "给帧编号" | 编码每个视频帧的时间位置——使模型理解帧间顺序 |
| 动态帧采样 | "自适应帧率" | 在固定视觉预算内平衡帧数和分辨率 |
| 时序定位 | "定位时间区间" | 给定自然语言查询，定位视频中的时间区间 |

---

## 📚 小结

视频 VLM 需要时序位置编码来理解帧间关系。动态帧采样在固定预算内平衡分辨率和帧数。Qwen2.5-VL 的 TMRoPE 与前沿模型拉平。时序定位让模型回答"在第几秒发生了什么"。

---

## ✏️ 练习

1. **【计算】** 在 288 词元预算下，计算 4 种不同帧数配置的每帧分辨率
2. **【实验】** 用 Qwen2-VL 处理一段视频——对比不同时序编码的效果

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 动态帧采样器 | `code/main.py` | 帧数-分辨率平衡 + 时序定位查询 |

---

## 📖 参考资料

1. [论文] Zhang et al. "Video-LLaMA: An Instruction-tuned Audio-Visual Language Model". arXiv, 2023.
2. [论文] Wang et al. "Qwen2-VL: Enhancing Vision-Language Model Perception". arXiv, 2024.
3. [论文] Lin et al. "VideoChat: Chat-Centric Video Understanding". arXiv, 2023.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
