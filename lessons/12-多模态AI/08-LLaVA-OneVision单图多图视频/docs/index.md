# LLaVA-OneVision：单图、多图、视频合一

> 在 LLaVA-OneVision（Li 等人，2024 年 8 月）之前，开源 VLM 世界有不同的谱系：LLaVA-1.5 用于单图，Mantis 和 VILA 用于多图，Video-LLaVA 和 Video-LLaMA 用于视频。LLaVA-OneVision 认为一个课程可以训练一个模型在所有三个场景中占优——单图技能可以迁移到视频，多图推理可以增强单图理解。配方简单得令人怀疑：一个跨场景恒定的视觉词元预算，加上从单图到 OneVision（多图）到视频的明确课程。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、06（任意分辨率）| **时间：** ~180 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 设计一个跨单图、多图、视频输入恒定的视觉词元预算
- [ ] 描述 LLaVA-OneVision 的三阶段课程——单图→多图→视频
- [ ] 解释"涌现任务迁移"——单图技能如何增强视频理解
- [ ] 构建 LLaVA-OneVision 格式的视频输入

---

## 1. 问题

2024 年之前，单图 VLM、多图 VLM、视频 VLM 是三个独立的世界。每种模型在自己的基准上赢了但在其他的输了。有没有可能用一个课程训练一个模型在所有场景中都占优？

LLaVA-OneVision 的答案：**是的**——而且更简单。

---

## 2. 概念

### 2.1 视觉词元预算

核心约束：无论输入是单图、多图还是视频，送入 LLM 的视觉词元总数必须相同。这意味着：
- 单图：更多图块分辨率 → 更多细节
- 多图：每张图更少图块 → 降低分辨率
- 视频：每帧更少图块 + 帧采样

### 2.2 三阶段课程

```
阶段 1: 单图课程
  图像 → [高分辨率] → 视觉词元 (B, K, D)
  训练：标题对齐 + 视觉指令

阶段 2: OneVision 课程（多图）
  多张图像 → [低分辨率每张] → 视觉词元 (B, K, D)
  训练：多图推理

阶段 3: 视频课程
  视频帧 → [帧采样 + 低分辨率] → 视觉词元 (B, K, D)
  训练：视频理解
```

### 2.3 涌现任务迁移

- 单图技能（OCR、文档理解）迁移到视频帧处理
- 多图推理（比较、排序）迁移到单图理解
- 视频的时序理解迁移到多图序列分析

### 2.4 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 视觉词元预算 | 128-288 | 跨场景恒定 |
| 单图分辨率 | 384-672 | 根据预算分配 |
| 视频帧采样率 | 1-4 fps | 预算约束下的帧数 |

---

## 3. 从零实现

### Step 1：视觉词元预算管理

```python
def allocate_visual_budget(total_budget, num_images, task_type="single"):
    """根据任务类型分配视觉词元预算。"""
    if task_type == "single":
        # 单图：所有预算给一张图
        return {"patches_per_image": total_budget, "num_images": 1}
    elif task_type == "multi":
        # 多图：平均分配
        per_image = total_budget // num_images
        return {"patches_per_image": per_image, "num_images": num_images}
    elif task_type == "video":
        # 视频：帧数 × 每帧词元 = 总预算
        # 预算有限时减少帧数
        frames = min(total_budget // 8, 16)  # 每帧最少 8 个词元
        per_frame = total_budget // frames
        return {"patches_per_image": per_frame, "num_frames": frames}
```

### Step 2：课程调度器

```python
class CurriculumScheduler:
    """三阶段课程调度器。"""
    def __init__(self, stage=1):
        self.stage = stage

    def get_training_config(self):
        configs = {
            1: {"data": "单图标题+指令", "resolution": "高", "budget": 288},
            2: {"data": "多图推理", "resolution": "中", "budget": 288},
            3: {"data": "视频理解", "resolution": "低", "budget": 288},
        }
        return configs[self.stage]

    def advance_stage(self):
        self.stage = min(self.stage + 1, 3)
```

---

## 4. 工具

### 4.1 HuggingFace Transformers

```python
from transformers import LlavaOnevisionForConditionalGeneration

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-7b-siglip-so400m-patch14-384"
)
```

---

## 6. 工程最佳实践

### 6.1 课程设计原则

- **从简单到复杂**：单图→多图→视频
- **恒定预算**：跨场景的视觉词元数不变——简化模型设计
- **数据递增**：每个阶段增加数据多样性

### 6.2 踩坑经验

- **视频帧采样太多**：预算超限——减少帧数或降低每帧分辨率
- **课程阶段跳跃**：跳过单图阶段直接训练视频——质量差

---

## 7. 常见错误

### 错误 1：视觉词元预算分配不当

**现象：** 多图时每张图分辨率太低——细节丢失。

**原因：** 总预算固定但每张图分配的词元太少。

### 错误 2：视频帧采样率不匹配训练配置

**现象：** 推理时使用与训练不同的帧率——时序理解崩溃。

---

## 8. 面试考点

### Q1：为什么视觉词元预算要跨场景恒定？（难度：⭐⭐）

**参考答案：**
LLM 的上下文窗口有限。如果单图用 288 个词元而视频用 128 个——模型在两种模式下的"视觉带宽"不同。恒定预算确保模型在所有场景下使用相同数量的视觉信息——简化了架构设计，也让跨场景迁移更自然。多图和视频通过降低每张图/帧的分辨率来适应预算。

### Q2：涌现任务迁移是什么意思？（难度：⭐⭐⭐）

**参考答案：**
训练一个任务时无意中提升了另一个任务的表现。在 LLaVA-OneVision 中：(1) 单图 OCR 训练让模型学会了"看文字"——这个能力迁移到视频帧的文字识别；(2) 多图比较训练让模型学会了"关联不同位置的信息"——这个能力增强单图理解。这种迁移是课程学习的副产品——不是显式设计的。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 视觉词元预算 | "视觉带宽" | 送入 LLM 的视觉词元总数——跨场景恒定约束 |
| 课程学习 | "从简单到复杂" | 先单图→再多图→最后视频的三阶段训练策略 |
| 涌现任务迁移 | "意外的技能迁移" | 训练一个任务时无意中提升了另一个任务的表现 |
| OneVision | "一个模型看所有" | LLaVA-OneVision 的核心理念——一个课程训练一个全能模型 |

---

## 📚 小结

LLaVA-OneVision 用一个课程在单图、多图、视频三个场景中训练一个模型。核心约束：视觉词元预算跨场景恒定。三阶段课程：单图→多图→视频。涌现迁移让技能在场景间流动。2026 年的趋势是"一个模型处理所有视觉任务"。

---

## ✏️ 练习

1. **【计算】** 在 288 个词元预算下，单图（384×384）和 4 张图（每张多少分辨率）的分配方案
2. **【设计】** 为 LLaVA-OneVision 设计一个中文场景的课程——加入中文 OCR 数据

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 预算管理器 | `code/main.py` | 视觉词元预算分配 + 课程调度 |

---

## 📖 参考资料

1. [论文] Li et al. "LLaVA-OneVision: Easy Visual Task Transfer". arXiv, 2024. https://arxiv.org/abs/2408.03326
2. [论文] Deitke et al. "Molmo and PixMo". arXiv, 2024.
3. [论文] Prismatic VLMs. arXiv, 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
