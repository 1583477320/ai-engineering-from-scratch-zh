# 具身 VLA 模型：OpenVLA、Pi0 与 GROOT

> 具身智能让 LLM 看见世界、理解指令、控制机器人。2024-2025 年的 VLA（Vision-Language-Action）模型将视觉语言理解与机器人动作生成统一到单一架构中。OpenVLA、Pi0、GROOT 代表了从开源到商业的完整技术栈。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、阶段 09（强化学习）| **时间：** ~120 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释 VLA 模型如何将视觉理解与动作生成统一到单一架构
- [ ] 对比 OpenVLA、Pi0、GROOT 的架构差异和适用场景
- [ ] 理解 sim-to-real 在具身智能中的作用
- [ ] 设计一个简单的 VLA 模型原型

---

## 1. 问题

传统机器人控制：感知（计算机视觉）→ 规划（传统 AI）→ 控制（PID 控制器）。每个模块独立，信息传递损失。

VLA 的答案：**用单一模型同时做感知、理解和动作生成。** 看到图像、理解指令、直接输出机器人动作。

---

## 2. 概念

### 2.1 VLA 架构

```
图像 + 指令 → [视觉编码器] → 视觉词元
                                    ↓
指令 → [文本编码器] → 文本词元 → [LLM 骨干] → [动作头] → 机器人动作
```

### 2.2 主要 VLA 模型

| 模型 | 架构 | 参数量 | 训练数据 |
|------|------|--------|---------|
| OpenVLA | Llama 2 + SigLIP | 7B | 970K 机器人轨迹 |
| RT-2 | PaLM-E + 机器人数据 | 55B | 多任务机器人数据 |
| Pi0 | 闭源 | - | 高质量机器人数据 |
| GROOT | Isaac Lab + PPO | - | 大规模仿真训练 |

### 2.3 Sim-to-Real

在仿真中训练 → 迁移到真实机器人。关键：域随机化——训练时随机化物理参数，使策略对真实世界的不确定性鲁棒。

### 2.4 动作空间

| 类型 | 描述 | 示例 |
|------|------|------|
| 离散动作 | 离散的控制指令 | "前进"、"后退"、"左转" |
| 连续动作 | 连续的关节扭矩 | [0.1, -0.3, 0.5, 0.2] |
| 语言动作 | 自然语言指令 | "把杯子放到桌子上" |

---

## 3. 从零实现

### Step 1：VLA 架构（简化版）

```python
import torch
import torch.nn as nn

class SimpleVLA(nn.Module):
    """简化版 VLA 模型。"""
    def __init__(self, image_dim=512, text_dim=512, hidden_dim=256, action_dim=7):
        super().__init__()
        # 视觉编码器（简化）
        self.image_proj = nn.Linear(3 * 64 * 64, image_dim)
        # 文本编码器（简化）
        self.text_proj = nn.Linear(64, text_dim)
        # 融合 + 动作预测
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, image, text_embed):
        img_feat = self.image_proj(image.view(image.size(0), -1))
        text_feat = self.text_proj(text_embed)
        combined = torch.cat([img_feat, text_feat], dim=-1)
        return self.fusion(combined)
```

---

## 4. 工具

### 4.1 OpenVLA

```python
from transformers import AutoModelForVision2Seq

# OpenVLA 7B
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
```

### 4.2 Isaac Lab（NVIDIA）

大规模并行机器人仿真——4096+ 并行机器人环境。

---

## 6. 工程最佳实践

### 6.1 Sim-to-Real 策略

| 阶段 | 方法 | 说明 |
|------|------|------|
| 训练 | Isaac Lab + DR | 域随机化训练策略 |
| 迁移 | 教师-学生蒸馏 | 从特权信息到传感器观测 |
| 部署 | 安全约束 PPO | 添加安全约束防止硬件损坏 |

### 6.2 踩坑经验

- **域随机化不足**：真实世界的物理参数变化范围可能比预期更大
- **动作空间不匹配**：仿真中的动作空间与真实机器人不同——需要适配层

---

## 7. 常见错误

### 错误 1：仿真中过度拟合物理参数

**现象：** 策略在仿真中完美但在真实世界失败。

**原因：** 仿真中的物理参数固定——策略记住了特定的参数组合。

**修复：** 域随机化——每次训练随机化物理参数。

---

## 8. 面试考点

### Q1：VLA 模型和传统机器人控制有什么根本区别？（难度：⭐⭐）

**参考答案：**
传统方法将感知、规划、控制分为独立模块——信息在模块间传递时损失。VLA 用单一模型同时做感知、理解和动作生成——端到端训练，信息不损失。关键：VLA 可以从语言指令直接生成机器人动作——不需要手写奖励函数或运动规划。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| VLA | "视觉语言动作模型" | 将视觉理解、语言理解、动作生成统一到单一架构的模型 |
| 域随机化 | "在随机仿真中训练" | 训练时随机化物理参数——使策略对真实世界的不确定性鲁棒 |
| Sim-to-Real | "仿真到真实" | 在仿真中训练策略，迁移到真实机器人 |
| 开环推理 | "一次生成全部动作" | 生成整个动作序列而非逐步执行 |

---

## 📚 小结

VLA 模型将视觉理解、语言理解、动作生成统一到单一架构。OpenVLA 是开源代表，Pi0/GROOT 是商业代表。Sim-to-Real 是关键挑战——域随机化和教师-学生蒸馏是主要解决方案。

---

## ✏️ 练习

1. **【对比】** 对比 OpenVLA 和 RT-2 的架构——7B 模型 vs 55B 模型的权衡
2. **【设计】** 为一个桌面整理机器人设计 VLA 系统——定义视觉输入、语言指令、动作输出

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| VLA 架构 | `code/main.py` | 简化版 VLA 模型 |

---

## 📖 参考资料

1. [论文] Kim et al. "OpenVLA: An Open-Source Vision-Language-Action Model". arXiv, 2024.
2. [论文] Brohan et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control". arXiv, 2023.
3. [论文] NVIDIA. "GROOT: Generalist Robot 00 Technology". 2024.

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
