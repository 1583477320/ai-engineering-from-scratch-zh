# DDPM 噪声调度分析器

## 用途

分析任意噪声调度参数对扩散过程的影响，帮助你选择合适的 beta 范围和步数。

## 使用方法

```bash
python -c "
from ddpm import NoiseScheduler
import matplotlib.pyplot as plt
import numpy as np

scheduler = NoiseScheduler(num_steps=1000, beta_start=0.0001, beta_end=0.02)
plt.plot(scheduler.betas.numpy())
plt.title('Beta Schedule (Linear)')
plt.xlabel('Step')
plt.ylabel('Beta')
plt.savefig('beta_schedule.png')
"
```

## 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `num_steps` | 1000 (DDPM) / 50-100 (DDIM) | 去噪步数 |
| `beta_start` | 0.0001 | 初始噪声强度 |
| `beta_end` | 0.02 | 最终噪声强度 |
| `schedule_type` | linear / cosine | 调度策略 |

## 输出

- `beta_schedule.png`：噪声调度曲线图
- 控制台输出：各关键时间步的 SNR 比值
