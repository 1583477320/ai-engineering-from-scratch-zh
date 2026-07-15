# Qwen-VL 家族与动态 FPS 视频

> Qwen-VL 家族（Qwen-VL 2023 → Qwen2-VL 2024 → Qwen2.5-VL 2025 → Qwen3-VL 2025）是 2026 年最有影响力的开源视觉语言模型谱系。每一代都做出了一个决定性的架构选择——原生动态分辨率（M-RoPE）、动态 FPS 采样与绝对时间对齐、ViT 中的窗口注意力——开源生态在 12 个月内复制了这些选择。到 Qwen3-VL，配方已经稳定：2D-RoPE-ViT 编码器 + MLP 投影器到 Qwen3 LLM 基础 + OCR/定位/代理行为作为首要训练目标。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 06（图块打包）| **时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 计算 M-RoPE 的三轴旋转（时间、高度、宽度）并解释为什么三个都需要
- [ ] 描述动态 FPS 采样如何在不同帧率的视频中保持时序一致性
- [ ] 对比 Qwen-VL 家族各代的架构变化
- [ ] 说明 Qwen-VL 在 OCR 和视觉定位方面的优势

---

## 1. 问题

2023-2025 年的 VLM 大多只能处理固定分辨率的图像。但真实世界的内容差异巨大——手机截图、医学影像、视频帧。Qwen-VL 家族通过**原生动态分辨率**和**动态 FPS 采样**解决了这两个限制。

核心创新：**M-RoPE**——多分辨率旋转位置编码——让 ViT 处理任意分辨率输入，无需填充或调整大小。

---

## 2. 概念

### 2.1 M-RoPE——三轴旋转位置编码

传统 RoPE 只旋转一个轴（位置）。M-RoPE 旋转三个轴：

| 轴 | 含义 | 旋转方式 |
|-----|------|---------|
| 时间轴 | 帧的时间位置 | 在时间维度旋转 |
| 高度轴 | 图像的高度位置 | 在高度维度旋转 |
| 宽度轴 | 图像的宽度位置 | 在宽度维度旋转 |

三轴旋转确保模型知道每个词元在时间、高度、宽度三个维度上的位置。

### 2.2 动态 FPS 采样

视频帧率不同（1fps 到 60fps）——需要动态调整采样策略：

| 帧率 | 策略 | 每秒采样 |
|------|------|---------|
| 高帧率（60fps） | 降采样到 2-4 fps | 更少帧但覆盖更长时间 |
| 低帧率（1fps） | 每帧都采样 | 所有帧 |
| 中等帧率（30fps） | 采样 4-8 fps | 平衡 |

### 2.3 Qwen-VL 家族演进

| 版本 | 年份 | 关键创新 | 分辨率 |
|------|------|---------|--------|
| Qwen-VL | 2023 | 可变分辨率 + 位置感知 | 448×448 |
| Qwen2-VL | 2024 | M-RoPE + 动态 FPS | 任意 |
| Qwen2.5-VL | 2025 | 改进 OCR + 更大词表 | 任意 |
| Qwen3-VL | 2025 | 结构化代理输出 | 任意 |

---

## 3. 从零实现

### Step 1：M-RoPE 三轴旋转

```python
import torch
import torch.nn as nn
import math

class MRoPE(nn.Module):
    """多分辨率旋转位置编码。"""
    def __init__(self, dim, max_position=8192):
        super().__init__()
        self.dim = dim
        # 三个轴的位置编码
        self.temporal_freq = nn.Parameter(torch.randn(max_position, dim) * 0.02)
        self.height_freq = nn.Parameter(torch.randn(max_position, dim) * 0.02)
        self.width_freq = nn.Parameter(torch.randn(max_position, dim) * 0.02)

    def forward(self, x, temporal_ids, height_ids, width_ids):
        """
        Args:
            x: (B, N, D) - 词元嵌入
            temporal_ids: (B, N) - 时间位置
            height_ids: (B, N) - 高度位置
            width_ids: (B, N) - 宽度位置
        """
        # 简化版：加法位置编码
        temporal_emb = self.temporal_freq[temporal_ids]
        height_emb = self.height_freq[height_ids]
        width_emb = self.width_freq[width_ids]

        return x + temporal_emb + height_emb + width_emb
```

### Step 2：动态 FPS 采样

```python
def dynamic_fps_sample(video_frames, target_fps=2, original_fps=30):
    """根据原始帧率动态采样。"""
    frame_interval = original_fps // target_fps
    indices = list(range(0, len(video_frames), frame_interval))
    sampled = [video_frames[i] for i in indices]
    return sampled, len(sampled)


def adaptive_fps(video_duration, max_frames=64):
    """自适应 FPS——根据视频时长决定采样率。"""
    if video_duration <= 10:
        return 4  # 短视频高帧率
    elif video_duration <= 60:
        return 2  # 中等视频中帧率
    else:
        return 1  # 长视频低帧率
```

### Step 3：Qwen-VL 采样配置

```python
def qwen_vl_sample_config(video_duration, target_frames=64):
    """Qwen-VL 的视频采样配置。"""
    fps = adaptive_fps(video_duration)
    num_frames = min(int(video_duration * fps), target_frames)
    return {"fps": fps, "num_frames": num_frames, "duration": video_duration}
```

---

## 4. 工具

### 4.1 HuggingFace Transformers

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
```

### 4.2 工具对比

| 模型 | 分辨率 | 视频支持 | OCR | 代理 |
|------|--------|---------|-----|------|
| Qwen2-VL | 任意 | ✅ 动态 FPS | 强 | 基础 |
| Qwen3-VL | 任意 | ✅ 动态 FPS | 强 | 原生 |
| LLaVA-OneVision | 动态 | ✅ | 中 | 无 |

---

## 6. 工程最佳实践

### 6.1 视频处理策略

| 视频类型 | FPS | 最大帧数 | 说明 |
|---------|-----|---------|------|
| 短视频（<10s） | 4 | 32 | 保留细节 |
| 中视频（10-60s） | 2 | 64 | 平衡 |
| 长视频（>60s） | 1 | 128 | 覆盖更多内容 |

### 6.2 踩坑经验

- **M-RoPE 维度不匹配**：确保三个轴的位置 ID 覆盖所有图块
- **动态 FPS 导致时序错位**：低帧率采样可能跳过关键帧

---

## 7. 常见错误

### 错误 1：固定 FPS 处理变长视频

**现象：** 长视频采样过多帧导致显存溢出。

**修复：** 使用自适应 FPS——根据视频时长动态调整采样率。

### 错误 2：忽略视频的时间信息

**现象：** 将视频帧独立处理——失去时序理解。

**修复：** 确保位置编码包含时间轴信息——M-RoPE 的时间轴旋转。

---

## 8. 面试考点

### Q1：M-RoPE 为什么需要三个轴的旋转？（难度：⭐⭐）

**参考答案：**
视频帧是 3D 数据——时间、高度、宽度。每个图块词元需要知道它在这三个维度上的位置。只用一个轴的旋转（如标准 RoPE）无法区分"同一帧的不同位置"和"不同帧的相同位置"。三轴旋转确保位置信息在所有维度上都被编码。

### Q2：动态 FPS 和固定 FPS 的区别是什么？（难度：⭐⭐⭐）

**参考答案：**
固定 FPS 对所有视频使用相同的采样率——短视频可能采样过多帧（冗余），长视频可能采样过少帧（遗漏）。动态 FPS 根据视频时长调整：短视频高帧率（保留细节），长视频低帧率（覆盖更多内容）。Qwen-VL 的动态 FPS 还结合了绝对时间对齐——确保不同帧率的视频在位置编码中保持时序一致性。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| M-RoPE | "多分辨率位置编码" | 三个轴（时间、高度、宽度）的旋转位置编码——支持任意分辨率视频 |
| 动态 FPS | "自适应帧采样" | 根据视频时长动态调整采样率——短视频高帧率，长视频低帧率 |
| 绝对时间对齐 | "帧率归一化" | 确保不同帧率的视频在位置编码中保持一致的时序信息 |
| 窗口注意力 | "局部注意力" | 每个词元只关注附近的词元——降低 ViT 的计算复杂度 |

---

## 📚 小结

Qwen-VL 家族通过 M-RoPE 实现原生动态分辨率，通过动态 FPS 处理变长视频。三轴旋转确保位置信息在时间、高度、宽度三个维度都被编码。到 Qwen3-VL，配方稳定：2D-RoPE-ViT + MLP 投影器 + OCR/定位作为首要目标。Qwen-VL 是 2026 年最强的开源 VLM 谱系之一。

---

## ✏️ 练习

1. **【计算】** 手动计算 M-RoPE 在 2×2 图块网格上的三个轴旋转——验证每个图元获得唯一的位置编码
2. **【实验】** 用 Qwen2-VL 处理一段 30 秒视频——对比 1fps、2fps、4fps 的生成质量

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| M-RoPE 实现 | `code/main.py` | 三轴旋转位置编码 + 动态 FPS 采样 |

---

## 📖 参考资料

1. [论文] Wang et al. "Qwen2-VL: Enhancing Vision-Language Model Perception". arXiv, 2024.
2. [论文] Qwen Team. "Qwen2.5 Technical Report". arXiv, 2025.
3. [论文] Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv, 2021.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
