---
name: prompt-world-models-guide
description: 根据任务类型、交互性需求、许可证要求和质量目标，选择最合适的视频生成模型或世界模型
phase: 4
lesson: 28
tags: [video-generation, world-models, model-selection, sora, genie]
---

# 视频模型与 World Model 选型指南

当需要为特定任务选择视频生成模型或世界模型时，本提示词帮助你系统评估各种选项（Sora 2、Runway Gen-5、Wan-Video、HunyuanVideo、Cosmos-Drive、Genie 3 等）。

## 输入

- `task`（任务类型）：creative_video（创意视频）| interactive_world（交互世界）| driving_sim（驾驶模拟）| robotics_sim（机器人模拟）| product_ad（产品广告）| explainer（解说视频）
- `duration_s`（时长）：所需的视频长度（秒）
- `interactivity`（交互需求）：static（静态生成）| mid-rollout-steerable（可中途操控）
- `license_need`（许可证要求）：permissive（宽松许可）| commercial_ok（可商用）| research_ok（研究可用）| api_ok（API 调用）
- `quality_target`（质量目标）：prototype（原型）| production（生产）| premium（精品）
- `deployment`（部署方式）：api（API 调用）| self-host（自托管）

## 决策流程

按优先级从高到低匹配，首个匹配规则胜出：

1. `interactivity == mid-rollout-steerable` → **Runway GWM-1 Worlds**（生产级）或 **Genie 3 研究预览**（研究）
2. `task == driving_sim` → **NVIDIA Cosmos-Drive**（开放权重）
3. `task == robotics_sim` → **Genie Envisioner** 或潜动作微调的 **HunyuanVideo**
4. `quality_target == premium` + `license_need == api_ok` → **Sora 2**（最佳质量 + 同步音频）或 **Runway Gen-5**
5. `quality_target in [prototype, production]` + `license_need == permissive` → **HunyuanVideo（13B）** 或 **Wan-Video 2.1（14B）**
6. `duration_s > 30` → **Sora 2** 唯一选择；开源模型最长约 10-20 秒
7. `deployment == self-host` → **HunyuanVideo**（宽松许可）或 **Wan-Video**（非商用）
8. 默认 → **Runway Gen-5**（API）用于静态视频生成

## 输出格式

```yaml
## 推荐模型

名称: <模型名>
时长上限: <秒>
分辨率上限: <H x W>
交互类型: static | steerable

## 部署方案

托管方式: <API | 自托管>
所需计算资源: <GPU 数量和型号>
成本估算: <每段视频>

## 注意事项

- 许可证说明
- 常见质量失败模式（物体恒存性、运动伪影）
- 是否支持音频
- 物理合理性检查建议
```

## 规则

- `task == product_ad` 时，优先 Sora 2 或 Runway Gen-5；开源模型目前质量仍有差距。
- `task == robotics_sim` 时，视频模型本身不够，需注明配套的逆动力学模型。
- 始终说明物理合理性失败模式——2026 年的视频模型在复杂物理场景中仍有问题。
- 涉及商用场景时，务必确认训练数据的许可证——不要推荐使用非商用数据训练的模型生成商业内容。
- 自托管方案需说明显存需求：HunyuanVideo 13B 需要约 28GB 显存（BF16），Wan-Video 14B 需要约 30GB 显存（BF16）。
- 生成物理合理性检查的推荐方案：对于生产级部署，建议配合 SAM 3.1 物体追踪 + 光流法 + 姿态估计进行自动检查。
