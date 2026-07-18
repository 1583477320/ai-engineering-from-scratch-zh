---
name: engine-picker
description: 根据硬件、规模和工作负载约束，选择自托管推理引擎并设计迁移路径。
version: 1.0.0
phase: 17
lesson: 28
tags: [llamacpp, ollama, vllm, sglang, tgi, 推理引擎]
---

根据硬件（CPU/AMD/NVIDIA）、规模（1/100/10000+用户）和工作负载（通用聊天/智能体/前缀密集），选择自托管推理引擎并设计跨环境的流水线。

**输出：** 引擎选择（硬件→规模→工作负载三层决策）、开发/暂存/生产流水线（Ollama→llama.cpp→vLLM）、量化策略（全程相同格式）、迁移路径（从TGI迁移出，如果适用）。

**硬拒绝：**
- AMD GPU选TRT-LLM——TRT-LLM是NVIDIA锁定的。
- 2026年新项目选TGI——TGI已进入维护模式。

**输出格式：** 引擎方案——引擎选择、流水线、量化策略、迁移路径。
