# Docker 容器——让"在我机器上能跑"成为过去

> 容器让"在我机器上能跑"成为过去。

**类型：** 构建
**编程语言：** Docker
**前置知识：** 第 00 阶段 · 01（开发环境配置）、第 03 节（GPU 与云）
**预计时间：** 60 分钟
**所处阶段：** Tier 1
**关联课程：** 第 00 阶段 · 06（Python 虚拟环境）— 虚拟环境是单机隔离；容器是跨机器隔离

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 构建带有 CUDA、PyTorch 和 AI 库的 GPU 启用 Docker 镜像
- [ ] 挂载主机目录为卷以持久化模型和数据
- [ ] 配置 NVIDIA Container Toolkit 让容器访问 GPU
- [ ] 使用 Docker Compose 编排多服务 AI 应用

---

## 1. 问题

你在笔记本上用 PyTorch 2.3、CUDA 12.4、Python 3.12 训练了模型。同事用 PyTorch 2.1、CUDA 11.8、Python 3.10。你的模型在他机器上崩溃。你的 Dockerfile 在两台机器上都能工作。

AI 项目的典型技术栈包括 Python、PyTorch、CUDA 驱动、cuDNN、系统级 C 库和 flash-attn 等需要精确编译器版本的特殊包。Docker 将所有这些打包成一个镜像，处处运行一致。

---

## 2. 核心概念

### 2.1 核心术语

| 术语 | 含义 |
|:-----|:-----|
| 镜像 | 只读模板。你的配方。从 Dockerfile 构建。 |
| 容器 | 镜像的运行实例。你的厨房。 |
| Dockerfile | 构建镜像的指令。逐层构建。 |
| 卷 | 持久化存储，容器重启后保留。 |
| docker-compose | 用 YAML 定义多容器应用的工具。 |

### 2.2 AI 容器模式

```text
开发容器：完整工具链，编辑器支持，Jupyter。开发调试。
训练容器：最小化，只有训练脚本和依赖。GPU 集群运行。
推理容器：优化服务。小镜像，快速冷启动。生产负载均衡。
```

---

## 3. 从零实现

### 第 1 步：安装 Docker

```bash
# macOS
brew install --cask docker

# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 注销并重新登录
```

### 第 2 步：安装 NVIDIA Container Toolkit（Linux GPU）

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU 访问：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 第 3 步：理解基础镜像

| 镜像 | 内容 | 大小 | 用途 |
|:-----|:-----|:-----|:-----|
| `nvidia/cuda:12.4.1-devel-ubuntu22.04` | 完整 CUDA 工具链 | ~4GB | 编译 flash-attn |
| `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | CUDA 运行时 | ~1.5GB | 运行预构建代码 |
| `pytorch/pytorch:2.3.1-cuda12.4-cudnn9` | PyTorch 预装 | ~6GB | 跳过安装步骤 |
| `python:3.12-slim` | 无 CUDA | ~150MB | CPU 推理 |

### 第 4 步：AI 开发 Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir \
    numpy pandas scikit-learn matplotlib jupyter transformers datasets accelerate

WORKDIR /workspace
VOLUME ["/workspace", "/models"]
EXPOSE 8888
CMD ["python"]
```

构建和运行：

```bash
docker build -t ai-dev .
docker run --rm -it --gpus all -v $(pwd):/workspace -v ~/models:/models \
    ai-dev python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 第 5 步：Docker Compose

```yaml
services:
  ai-dev:
    build:
      context: .
      dockerfile: Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ../../../:/workspace
      - ~/models:/models
    ports:
      - "8888:8888"
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  qdrant:
    image: qdrant/qdrant:v1.12.5
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

```bash
cd phases/00-setup-and-tooling/07-docker-for-ai/code
docker compose up -d
```

---

## 4. 工业工具

| 工具 | 用途 | 特点 |
|:-----|:-----|:-----|
| Docker | 容器化 | 标准工具 |
| NVIDIA Container Toolkit | GPU 穿透 | 让容器访问 GPU |
| Docker Compose | 多服务编排 | YAML 定义，一键启动 |
| Podman | Docker 替代 | 无守护进程 |

---

## 5. 知识连线

- **第 11 阶段（LLM 工程）**：容器化 API 服务器和向量数据库
- **第 17 阶段（基础设施）**：生产环境的容器化部署
- **第 19 阶段（综合项目）**：沙箱运行器使用容器隔离执行

---

## 6. 工程最佳实践

- **层缓存优化**：将不经常变化的层放在 Dockerfile 上部
- **卷挂载模型和数据**：容器销毁不丢失大文件
- **生产用 `docker compose down -v` 清理**：不清理会累积存储
- **中文场景特别建议**：国内 Docker Hub 访问慢，配置镜像源：`/etc/docker/daemon.json` 中添加 `registry-mirrors`

---

## 7. 常见错误

### 错误 1：基础镜像选错

**现象：** 无法安装 flash-attn 等需要 CUDA 编译器的包。

**原因：** 使用了 `runtime` 镜像而非 `devel` 镜像。

**修复：** 需要编译的包用 `devel` 镜像；仅运行用 `runtime` 镜像。

### 错误 2：未使用卷挂载模型

**现象：** 容器重建后 14GB 模型消失。

**原因：** 模型下载到了容器内部，容器销毁后数据丢失。

**修复：** `-v ~/models:/models` 将模型目录从主机映射进来。

### 错误 3：未安装 NVIDIA Container Toolkit

**现象：** `docker run --gpus all` 报错找不到 GPU。

**原因：** 主机上没有安装 NVIDIA Container Toolkit 或未重启 Docker。

**修复：** 安装 toolkit 后执行 `sudo systemctl restart docker`。

---

## 8. 面试考点

### Q1：Docker 镜像和容器的关系是什么？（难度：⭐）

**参考答案：** 镜像是只读模板（类比：菜谱）。容器是镜像的运行实例（类比：正在做的菜）。一个镜像可以运行多个容器。

### Q2：为什么 AI 项目比一般软件项目更需要 Docker？（难度：⭐⭐）

**参考答案：** AI 技术栈依赖链复杂（CUDA 版本、PyTorch 版本、flash-attn 编译）。Docker 将所有依赖打包到镜像中——CUDA 工具链、cuDNN、编译器版本全部锁定。"在我机器上能跑"在 AI 项目中极其常见，Docker 是最直接的解决方案。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|:-----|:---------|:---------|
| 容器 | "轻量虚拟机" | 使用主机内核的隔离进程 |
| 镜像层 | "缓存步骤" | Dockerfile 指令创建的层，未改变的层被缓存 |
| 卷挂载 | "共享文件夹" | 主机目录映射到容器内，容器停止后保留 |
| NVIDIA Container Toolkit | "Docker 中的 GPU" | 将主机 GPU 暴露给容器的运行时钩子 |

---

## 📚 小结

Docker 让 AI 项目的环境可复现。你构建了 GPU 启用的 Docker 镜像，学习了卷挂载和 Docker Compose。有了这些，你的环境可以在任何机器上完全一致地运行。

下一课学习编辑器配置。

---

## ✏️ 练习

1. 【实现】构建 Dockerfile，运行 `python -c "import torch; print(torch.__version__)"` 验证
2. 【实验】启动 docker-compose 栈，从 AI 容器验证 Qdrant 可访问
3. 【理解】在 Docker 中运行 Jupyter：映射 8888 端口

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|:-----|:-----|:-----|
| Dockerfile | `code/Dockerfile` | AI 开发镜像 |
| Docker Compose | `code/docker-compose.yml` | AI 开发 + Qdrant 编排 |

---

## 📖 参考资料

1. [官方文档] Docker. https://docs.docker.com/
2. [官方文档] NVIDIA Container Toolkit. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
3. [官方文档] Docker Compose. https://docs.docker.com/compose/
