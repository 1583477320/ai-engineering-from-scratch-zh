# GPU 与云服务——本地加速与云端训练

> 在 CPU 上训练适合学习。真正的训练需要 GPU。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第 00 阶段 · 01（开发环境配置）
**预计时间：** 45 分钟
**所处阶段：** Tier 1
**关联课程：** 第 03 阶段 · 05（PyTorch 入门）— 本课的 GPU 测试代码是该阶段的先导

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 用 `nvidia-smi` 和 PyTorch 验证本地 GPU 可用性
- [ ] 在 Google Colab 上配置 T4 GPU
- [ ] 运行矩阵乘法基准对比 CPU 与 GPU 的速度差
- [ ] 用 fp16 经验法则估算显存能容纳的最大模型

---

## 1. 问题

第 1-3 阶段的大部分课程在 CPU 上运行良好。但一旦开始训练 CNN、Transformer、大语言模型（第 4+ 阶段），就需要 GPU 加速。CPU 上 8 小时的训练任务在 GPU 上只需 10 分钟。

你有三个选择：本地 GPU、云 GPU、Google Colab（免费）。

---

## 2. 核心概念

### 2.1 三种选项对比

```
1. 本地 NVIDIA GPU
   成本：$0（你已经有了）
   安装：CUDA + cuDNN
   适用：日常使用、大数据集

2. Google Colab（免费）
   成本：$0
   安装：无需
   适用：快速实验、没有 GPU 时

3. 云 GPU（Lambda、RunPod、Vast.ai）
   成本：$0.20-2.00/小时
   安装：SSH + 安装
   适用：大规模训练
```

### 2.2 fp16 经验法则

```
可容纳的最大参数数量 ≈ 显存(GB) × 0.5 × 10⁹
例如 8GB 显存 ≈ 4B 参数(fp16)
```

| 精度 | 每参数字节 | 8GB 可容参数 |
|:-----|:----------|:------------|
| fp32 | 4 | ~2B |
| fp16 | 2 | ~4B |
| int8 | 1 | ~8B |

---

## 3. 从零实现

### 第 1 步：验证本地 GPU

```bash
nvidia-smi
```

输出示例（没有 GPU 则报错）：

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08    Driver Version: 545.23.08    CUDA Version: 12.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     |        Memory-Usage  |        Compute Cap   |
| 0  NVIDIA RTX 4090            |         22111MiB     |              8.9      |
+-------------------------------+----------------------+----------------------+
```

```python
import torch

print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 第 2 步：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. 点击 Runtime → Change runtime type → T4 GPU
3. 在代码单元格中运行 `!nvidia-smi` 验证

### 第 3 步：CPU vs GPU 基准

GPU 的核心优势是并行矩阵乘法。以下基准展示了差异：

```python
import torch
import time

size = 5000  # 5000x5000 矩阵

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()  # GPU 计算是异步的，需要同步计时
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"加速比: {cpu_time / gpu_time:.0f}x")
```

### 第 4 步：通用设备句柄

在后续课程中，你将始终使用这个通用的设备检测方式：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

### 第 5 步：显存监控

```python
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        print(f"已分配: {allocated:.2f}GB  缓存: {cached:.2f}GB")
```

---

## 4. 工业工具

### 4.1 GPU 云服务对比

| 服务 | 价格 | 最低配置 | 特点 |
|:-----|:-----|:---------|:-----|
| Google Colab | 免费 / $10 Pro | T4 15GB | 无需配置，Jupyter 原生 |
| Lambda Labs | $0.30-2.00/h | A10 24GB | 学术友好，纯 GPU |
| RunPod | $0.19-2.50/h | RTX 3090 | 按秒计费 |
| Vast.ai | $0.15-1.50/h | 多种 GPU | 社区市场，最便宜 |
| AWS / GCP | $1-10/h | T4 / V100 | 全面但复杂 |

### 4.2 显存估算

```bash
# 查看进程显存占用
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 持续监控
watch -n 1 nvidia-smi
```

---

## 5. 知识连线

- **第 03 阶段 · 05（PyTorch 入门）**：本课的 `torch.cuda.is_available()` 是后续所有 GPU 训练的前置条件
- **第 07 阶段（Transformer 深入）**：本课的矩阵乘法基准解释了为什么 GPU 对 Transformer 如此重要
- **第 10 阶段（大语言模型从零）**：显存估算方法直接决定购买的 GPU 规格

---

## 6. 工程最佳实践

- **GPU 异步计算需要同步**：使用 `torch.cuda.synchronize()` 在计时前确保 GPU 完成计算
- **Colab 会话 12 小时超时**：长时间训练需要保存检查点
- **显存碎片化**：训练大模型时使用 `torch.cuda.empty_cache()` 释放未使用的缓存
- **中文场景特别建议**：Colab 免费版从中国大陆访问可能需要代理；推荐国内使用 AutoDL 或阿里云 GPU 实例

---

## 7. 常见错误

### 错误 1：忘记调用 `to("cuda")`

**现象：** 代码运行在 CPU 上，非常慢。

**原因：** 模型和数据默认在 CPU 上，需要显式转移到 GPU。

**修复：**
```python
model = MyModel().to(device)
batch = batch.to(device)
```

### 错误 2：CPU 密集型预处理与 GPU 没有重叠

**现象：** GPU 利用率低（`nvidia-smi` 显示 20-30%）。

**原因：** 数据加载和预处理在 CPU 上串行执行，GPU 空闲等待。

**修复：** 使用 `DataLoader(num_workers=4)` 增加工作进程数。

### 错误 3：CUDA 版本不匹配

**现象：** `import torch` 报库未找到错误。

**原因：** PyTorch 版本与系统的 CUDA 驱动版本不兼容。

**修复：** 使用 PyTorch 官网提供的匹配版本的安装命令。

---

## 8. 面试考点

### Q1：GPU 为什么在深度学习中比 CPU 快？（难度：⭐⭐）

**参考答案：** GPU 有数千个核心（如 RTX 4090 有 16384 个 CUDA 核心），而 CPU 有 8-16 个核心。深度学习的主要操作是矩阵乘法——天然高度并行。GPU 在内存带宽（1TB/s+）上也远高于 CPU（50-80GB/s）。

### Q2：什么是 Tensor Core？（难度：⭐⭐⭐）

**参考答案：** Tensor Core 是 NVIDIA Volta+ 架构引入的专用硬件单元，可以直接执行 `D = A * B + C`（矩阵乘加），单次操作处理 4x4 矩阵块。它们比常规 CUDA Core 快 4-8 倍，是现代混合精度训练的核心硬件基础。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|:-----|:---------|:---------|
| CUDA | "GPU 编程" | 让 GPU 执行通用计算的 NVIDIA 平台 |
| 显存 | "GPU 内存" | 显卡上的视频 RAM，与系统 RAM 隔离，限制模型大小 |
| fp16 | "半精度" | 16 位浮点数，相比 fp32 内存减半、精度损失极小 |
| Tensor Core | "快速矩阵硬件" | GPU 上专为矩阵乘法优化的单元，比常规核心快 4-8x |

---

## 📚 小结

你学会了三种使用 GPU 的方式：本地 NVIDIA GPU、Google Colab 免费 GPU、云 GPU。你运行了 CPU vs GPU 基准测试，确认了加速比，并且学会了用显存经验法则估算模型规模。没有 GPU 也没关系——大多数课程在 CPU 上可运行。

---

## ✏️ 练习

1. 【实验】运行 CPU vs GPU 基准测试，记录加速比
2. 【实验】如果没有 GPU，在 Google Colab 上运行同一基准，比较结果
3. 【理解】用 `torch.cuda.get_device_properties(0)` 查看你的 GPU 详细信息，计算能容纳的最大模型大小（fp16）

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| GPU 基准测试 | `code/main.py` | CPU vs GPU 矩阵乘法 Benchmark + CUDA 验证 |
| 可复用提示词 | `outputs/skill-gpu-setup.md` | GPU 配置与云服务选择指南 |

---

## 📖 参考资料

1. [官方文档] NVIDIA CUDA 安装指南. https://docs.nvidia.com/cuda/
2. [官方文档] Google Colab 免费 GPU. https://colab.research.google.com/
3. [官方文档] PyTorch CUDA 语义. https://pytorch.org/docs/stable/notes/cuda.html
4. [论文] Micikevicius et al. "Mixed Precision Training". https://arxiv.org/abs/1710.03740
