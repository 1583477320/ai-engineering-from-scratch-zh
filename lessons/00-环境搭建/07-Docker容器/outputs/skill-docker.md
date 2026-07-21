# Docker AI 配方

## 核心命令

```bash
docker build -t ai-dev .
docker run --rm -it --gpus all -v $(pwd):/workspace ai-dev python -c "import torch; print(torch.__version__)"
docker compose up -d
```

## 基础镜像选择

| 场景 | 镜像 | 大小 |
|:-----|:-----|:-----|
| 编译 flash-attn | nvidia/cuda:12.4.1-devel-ubuntu22.04 | ~4GB |
| 运行预构建代码 | nvidia/cuda:12.4.1-runtime-ubuntu22.04 | ~1.5GB |
| 跳过 PyTorch 安装 | pytorch/pytorch:2.3.1-cuda12.4-cudnn9 | ~6GB |
| CPU 推理 | python:3.12-slim | ~150MB |

## 卷挂载

```bash
-v $(pwd):/workspace     # 代码
-v ~/models:/models     # 模型
-v ~/datasets:/data     # 数据集
```
