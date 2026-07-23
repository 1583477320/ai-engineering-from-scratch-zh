---
name: skill-heatmap-to-coords
description: 写入子像素热力图到坐标转换例程——每个生产姿态模型的核心模块
version: 1.0.0
phase: 4
lesson: 21
tags: [keypoint, pose, subpixel, inference]
---

# 热力图转坐标

将原始关键点热力图转换为子像素精度的坐标。每一个姿态流水线中性价比最高的精度提升方法。

## 使用场景

- 部署热力图格式的关键点检测模型
- 基准测试姿态指标——OKS（Object Keypoint Similarity）对子像素精度极其敏感
- 将姿态代码从一个框架移植到另一个框架

## 输入

- `heatmaps`: `(N, K, H, W)` 张量，来自模型的逐关键点热力图
- `confidence_threshold`: 丢弃峰值低于此置信度的关键点

## 步骤

1. **Argmax** — 逐热力图取最大值索引，找到整数峰值位置。

2. **一阶差分偏移** — 从邻域像素估计亚像素偏移。`0.25` 系数是为 `sigma >= 1` 的高斯热力图校准的经验值。对于更严格的做法，使用完整的二次拟合（DARK 算法）或高斯拟合。

    ```
    dx = 0.25 * sign(heatmap[y, x+1] - heatmap[y, x-1])
    dy = 0.25 * sign(heatmap[y+1, x] - heatmap[y-1, x])
    ```

    二次拟合变体使用局部二阶导数近似：

    ```
    dx = -0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
             / (heatmap[y, x+1] - 2 * heatmap[y, x] + heatmap[y, x-1] + eps)
    ```

    二次拟合在尖锐热力图上更准确；符号偏移在热力图有噪声时更安全。

3. **累加偏移** 到整数峰值。

4. **置信度** — 返回每个关键点的峰值值；客户端用它来屏蔽低置信度预测。

5. **边界情况处理** — 当峰值位于轴上的第一个或最后一个像素时，对应方向的邻居被 clamp 处理，偏移归零——这是最安全的回退策略。

## 输出模板

```python
import torch

def heatmap_to_coords_subpixel(heatmaps, threshold=0.2):
    """从热力图中提取带子像素精度的坐标。

    Args:
        heatmaps: (N, K, H, W) 热力图张量
        threshold: 置信度阈值，低于此值的预测将被标记为无效

    Returns:
        coords: (N, K, 2) 子像素精度坐标 [x, y]
        conf: (N, K) 置信度得分
        valid_mask: (N, K) 布尔掩码，True 表示有效预测
    """
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    conf, idx = flat.max(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()

    ys_int = ys.long()
    xs_int = xs.long()

    # 边界安全索引
    x_minus = (xs_int - 1).clamp(min=0)
    x_plus = (xs_int + 1).clamp(max=W - 1)
    y_minus = (ys_int - 1).clamp(min=0)
    y_plus = (ys_int + 1).clamp(max=H - 1)

    # 批量索引构建
    batch_idx = torch.arange(N).view(-1, 1).expand(-1, K)
    kp_idx = torch.arange(K).view(1, -1).expand(N, -1)

    # 提取邻域像素值
    dx_raw = (heatmaps[batch_idx, kp_idx, ys_int, x_plus]
              - heatmaps[batch_idx, kp_idx, ys_int, x_minus])
    dy_raw = (heatmaps[batch_idx, kp_idx, y_plus, xs_int]
              - heatmaps[batch_idx, kp_idx, y_minus, xs_int])
    dx = 0.25 * torch.sign(dx_raw)
    dy = 0.25 * torch.sign(dy_raw)

    # 边界归零
    at_left = xs_int == 0
    at_right = xs_int == (W - 1)
    at_top = ys_int == 0
    at_bottom = ys_int == (H - 1)
    dx = torch.where(at_left | at_right, torch.zeros_like(dx), dx)
    dy = torch.where(at_top | at_bottom, torch.zeros_like(dy), dy)

    refined_x = xs + dx
    refined_y = ys + dy
    coords = torch.stack([refined_x, refined_y], dim=-1)
    mask = conf >= threshold
    return coords, conf, mask
```

## 报告

```
[子像素解码报告]
  关键点数量:   K
  阈值:         <浮点数>
  有效率:       高于阈值的关键点比例
```

## 规则

- 始终将邻域索引 clamp 到有效范围；越界的关键点偏移为零但不崩溃。
- 与坐标一起返回置信度，以便客户端屏蔽低置信度预测。
- 子像素优化仅在热力图峰值附近平滑时才有效——检查训练时是否使用了 sigma >= 1 的高斯目标。
- 对于很小的热力图分辨率（< 48×48），考虑在提取坐标前将热力图上采样到全图像尺寸；子像素偏移随 stride 缩放。
