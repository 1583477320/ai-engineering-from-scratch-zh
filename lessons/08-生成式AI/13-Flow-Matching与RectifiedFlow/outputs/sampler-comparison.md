# 采样器对比：DDIM vs Flow Matching vs DPM-Solver

## 对比总览

| 采样器 | 类型 | 推荐步数 | 质量 | 速度 | 适用场景 |
|--------|------|---------|------|------|---------|
| DDIM | SDE | 20-50 | 高（基准） | 中 | 通用 |
| Flow Matching (Euler) | ODE (一阶) | 5-15 | 中 | 快 | 快速生成 |
| Flow Matching (Heun) | ODE (二阶) | 3-10 | 高 | 中 | 平衡 |
| DPM-Solver++ | ODE (二阶) | 5-20 | 高 | 快 | 高性能 |
| Consistency Model | 蒸馏 | 1 | 中 | 极快 | 实时推理 |

## 步数-质量曲线

```
质量
 ↑
 |   DM++ (10步)
 |   Heun (8步)
 |   Euler (15步)
 |   DDIM (30步)
 |   Consistency (1步)
 |   
 +──────────────────────────→ 速度
```

## 使用指南

### 什么时候用哪个？

- **质量优先（≥ 50 步）**：DDIM 或 DPM-Solver
- **速度优先（≤ 5 步）**：Flow Matching (Heun) 或 Consistency Model
- **平衡（10-20 步）**：Flow Matching (Euler) 或 DPM-Solver++

### 步数选择

| 目标质量 | DDIM | Flow Match (Euler) | DPM-Solver++ |
|---------|------|-------------------|-------------|
| 预览 | 10 | 3 | 5 |
| 日常 | 30 | 8 | 10 |
| 高质量 | 50 | 15 | 20 |
| 极致 | 100 | 30 | 40 |

## 代码示例

```python
# DDIM
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt, num_inference_steps=30).images[0]

# Flow Matching (Euler)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt, num_inference_steps=8).images[0]

# DPM-Solver++
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt, num_inference_steps=10).images[0]
```

## 观察结果

1. 10 步 DPM-Solver++ ≈ 50 步 DDIM 质量
2. 5 步 Flow Matching (Heun) > 5 步 Flow Matching (Euler)
3. 1 步 Consistency Model 质量明显差于多步结果
4. 步数增加到超过 50 后，边际效益递减
