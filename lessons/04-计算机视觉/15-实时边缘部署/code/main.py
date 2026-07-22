# main.py — 实时边缘部署：测量、量化、导出与推理优化
# 依赖：torch>=2.0, torchvision>=0.15, onnxruntime (可选)
# 对应课程：阶段 04 · 15（实时边缘部署）

import time
import math
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 第 1 步：正确的延迟测量 —— 预热、同步、百分位
# ============================================================

def measure_latency(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    """测量模型推理延迟，报告 p50 / p95 / p99 / 均值。

    三个规则：
    1. 预热（warmup）—— 第一次前向传播受 JIT 编译和冷缓存影响，数值不代表真实性能
    2. 同步（synchronize）—— GPU 上必须调用 torch.cuda.synchronize()，否则测量的是内核调度时间而非执行时间
    3. 固定输入尺寸 —— 生产中使用的分辨率就是基准测试的分辨率
    """
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)

    with torch.no_grad():
        # 预热阶段：不计时的前向传播
        for _ in range(warmup):
            model(x)

        # CUDA 同步（确保前面的 kernel 全部完成）
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

    # 排序后取百分位数
    times.sort()
    n = len(times)
    return {
        "p50_ms": times[n // 2],
        "p95_ms": times[int(n * 0.95)],
        "p99_ms": times[-1],
        "mean_ms": sum(times) / n,
    }


# ============================================================
# 第 2 步：参数数量和 FLOPs 估算
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型总参数量。"""
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model: nn.Module, input_shape: tuple) -> int:
    """
    通过 forward hook 估算 FLOPs（浮点运算次数）。

    对于卷积层，每个输出的计算量约为：
      2 × C_in × C_out × K_h × K_w × H_out × W_out

    乘 2 是因为一次卷积包含 C_in×K_h×K_w 次乘法和同样数量的加法。

    注意：FLOPs 是硬件无关的代理指标，不是真实的 wall-clock 延迟。
    深度可分离卷积等操作的 FLOPs 较低，但在 GPU 上可能因为内存带宽受限而并不比标准卷积快。
    """
    total_flops = [0]

    def conv_hook(module, input, output):
        c_in = module.in_channels
        c_out = module.out_channels
        k_h, k_w = module.kernel_size
        h_out, w_out = output.shape[-2:]
        # FLOPs = 2 * MACs（MAC = 乘加一次操作）
        total_flops[0] += 2 * c_in * c_out * k_h * k_w * h_out * w_out

    def linear_hook(module, input, output):
        total_flops[0] += 2 * module.in_features * module.out_features

    hooks = []
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_shape))

    for handle in hooks:
        handle.remove()
    return total_flops[0]


# ============================================================
# 第 3 步：深度可分离卷积 —— MobileNet 的核心
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（Depthwise Separable Convolution）的教学实现。

    MobileNet 的核心创新：将标准卷积分解为两步：
      1. Depthwise（逐通道卷积）：每个输入通道独立做空间滤波
      2. Pointwise（1×1 卷积）：跨通道线性组合

    相比标准卷积，参数量减少约 8-9 倍。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 1):
        super().__init__()
        # 第一步：逐通道卷积（每个通道一个核）
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=padding,
            groups=in_channels,  # groups=in_channels 即为逐通道卷积
            bias=False,
        )
        # 第二步：逐点卷积（1×1），负责跨通道信息融合
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            groups=1,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu(x)
        return x


def compare_conv_approaches(in_ch: int, out_ch: int):
    """对比标准卷积和深度可分离卷积的参数/FLOPs。

    标准卷积参数量：C_in × C_out × K_h × K_w
    深度可分离卷积参数量：C_in × K_h × K_w + C_in × C_out × 1 × 1

    当 C_out >> C_in 时，后者可减少 8-9 倍的参数。
    """
    # 标准卷积
    std_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    std_params = sum(p.numel() for p in std_conv.parameters())

    # 深度可分离卷积
    sep_conv = DepthwiseSeparableConv(in_ch, out_ch)
    sep_params = sum(p.numel() for p in sep_conv.parameters())

    reduction = (1 - sep_params / std_params) * 100

    print(f"{'架构':<20} {'参数量':>10} {'压缩率':>10}")
    print("-" * 45)
    print(f"{'标准卷积':<20} {std_params:>10,} {'':>10}")
    print(f"{'深度可分离卷积':<20} {sep_params:>10,} {reduction:>8.1f}%")

    return std_params, sep_params


# ============================================================
# 第 4 步：通道混洗 —— ShuffleNet 的核心操作
# ============================================================

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """通道混洗操作（Channel Shuffle）。

    ShuffleNet 在逐组卷积（Group Convolution）之后引入通道混洗，
    让不同组的特征可以交互。如果不用混洗，各组之间完全隔离，
    网络表达能力会大幅下降。

    假设输入形状 (N, C, H, W)，分为 g 组：
    1. 重塑为 (N, g, C/g, H, W)
    2. 转置为 (N, C/g, g, H, W)
    3. 展平为 (N, C, H, W)

    这样每个输出通道的特征来自不同的输入组。
    """
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0, f"通道数 {channels} 必须是组数 {groups} 的倍数"

    channels_per_group = channels // groups
    x_reshaped = x.view(batch_size, groups, channels_per_group, height, width)
    x_transposed = x_reshaped.transpose(1, 2).contiguous()
    return x_transposed.view(batch_size, channels, height, width)


def demo_channel_shuffle():
    """演示通道混洗的效果。"""
    print("\n--- 通道混洗演示 ---")

    batch, channels, height, width = 1, 8, 4, 4
    groups = 2
    channels_per_group = channels // groups

    # 构造一个示例张量，用组号区分特征来源
    x = torch.zeros(batch, channels, height, width)
    for i in range(channels):
        x[:, i, :, :] = i // channels_per_group  # 前 4 个通道属于第 0 组，后 4 个属于第 1 组

    print(f"混洗前：通道 0-3 的值 = {x[0, 0:4, 0, 0].item():.0f} (第 0 组)")
    print(f"         通道 4-7 的值 = {x[0, 4:8, 0, 0].item():.0f} (第 1 组)")

    shuffled = channel_shuffle(x, groups)
    print(f"混洗后：通道 0 的值 = {shuffled[0, 0, 0, 0].item():.0f}, 通道 1 = {shuffled[0, 1, 0, 0].item():.0f}")
    print(f"         通道 2 = {shuffled[0, 2, 0, 0].item():.0f}, 通道 3 = {shuffled[0, 3, 0, 0].item():.0f}")
    print(f"         混洗后的每个通道现在混合了来自不同组的特征")
    print(f"✓ 通道混洗确保了组间的信息流通")


# ============================================================
# 第 5 步：INT8 量化 —— 训练后静态量化（PTQ）
# ============================================================

def apply_static_quantization(
    model: nn.Module,
    calibration_data: list,
) -> nn.Module:
    """对模型应用训练后静态量化（Post-Training Static Quantization）。

    流程：
    1. 配置量化方案（选择后端：x86 / fbgemm）
    2. prepare —— 在合适位置插入量化/反量化观察器（observer）
    3. 校准 —— 用少量真实数据跑前向传播，收集激活值的范围
    4. convert —— 将可量化层转换为 INT8 版本

    量化后：
    - 模型体积缩小约 4 倍（FP32 → INT8）
    - 显存带宽需求降低约 4 倍
    - 在有 INT8 算子的硬件上推理速度提升 2-4 倍
    """
    import torch.ao.quantization as tq

    model = model.eval().cpu()

    # 设置量化配置
    model.qconfig = tq.get_default_qconfig("fbgemm")  # fbgemm 适合 x86 CPU

    # 保存原始类别数以恢复
    num_classes_backup = None
    if hasattr(model, "classifier"):
        # 兼容 torchvision 模型
        num_classes_backup = model.classifier[-1].in_features if len(model.classifier) > 1 else model.classifier.in_features

    # Step 1: prepare —— 插入观察者
    tq.prepare(model, inplace=True)

    # Step 2: 校准 —— 用真实数据确定激活值范围
    print("正在校准量化参数...")
    with torch.no_grad():
        for i, data in enumerate(calibration_data):
            if isinstance(data, (tuple, list)):
                model(data[0])  # 有些数据集返回 (images, labels)
            else:
                model(data)
            if i >= 19:  # 20 个样本足够校准
                break

    # Step 3: convert —— 融合并量化
    tq.convert(model, inplace=True)

    print("✓ 量化完成。模型已切换为 INT8 推理模式")
    return model


def demonstrate_quantization():
    """演示量化的效果对比。"""
    from torchvision.models import mobilenet_v3_small

    print("\n--- INT8 量化对比 ---")

    # 加载 FP32 模型
    fp32_model = mobilenet_v3_small(weights=None, num_classes=10)
    fp32_model.eval()

    fp32_size = count_parameters(fp32_model)
    input_sample = torch.randn(1, 3, 224, 224)

    # 模拟校准数据
    calib_data = [torch.randn(1, 3, 224, 224) for _ in range(20)]

    # 量化模型
    quantized_model = apply_static_quantization(fp32_model, calib_data)

    # 对比推理结果
    with torch.no_grad():
        fp32_output = fp32_model(input_sample)
        q_output = quantized_model(input_sample)

    # 计算输出差异
    diff = (fp32_output - q_output).abs().max().item()
    print(f"\nFP32 模型输出范围：[{fp32_output.min():.4f}, {fp32_output.max():.4f}]")
    print(f"INT8 模型输出范围：[{q_output.min():.4f}, {q_output.max():.4f}]")
    print(f"最大输出差异：{diff:.6f}")

    if diff < 0.5:
        print("✓ 量化后的输出与原始模型高度一致，精度损失可控")
    else:
        print("⚠ 量化差异较大，可能需要增加校准数据或改用 QAT（量化感知训练）")


# ============================================================
# 第 6 步：稀疏化剪枝 —— 移除不重要的连接
# ============================================================

def magnitude_pruning(
    model: nn.Module,
    sparsity: float = 0.5,
) -> dict:
    """按幅度剪枝（Magnitude-based Pruning）。

    原理：权重绝对值越小，对该层的输出贡献越少。
    将所有权重的绝对值从小到大排序，将最小的 sparsity 比例的权重设为 0。

    注意：结构化剪枝（prune entire channels/layers）比非结构化剪枝
    在实际部署中更高效，因为可以将稀疏矩阵变为非方阵。
    """
    total_params = 0
    pruned_params = 0

    for name, param in model.named_parameters():
        if param.ndim > 1 and "conv" in name:  # 只对卷积层的权重做剪枝
            absolute_weights = param.abs()
            threshold = absolute_weights.flatten().quantile(sparsity).item()
            mask = (absolute_weights >= threshold).float()

            pruned_count = (mask == 0).sum().item()
            total_layer = param.numel()
            pruned_params += pruned_count
            total_params += total_layer

            # 应用掩码
            param.data.mul_(mask)

    achieved_sparsity = pruned_params / max(total_params, 1)
    print(f"\n剪枝统计：")
    print(f"  目标稀疏度：{sparsity * 100:.0f}%")
    print(f"  实际稀疏度：{achieved_sparsity * 100:.1f}%")
    print(f"  被剪枝的权重数：{pruned_params:,}")
    print(f"  总参数量：{total_params:,}")

    return {"total_params": total_params, "pruned_params": pruned_params, "sparsity": achieved_sparsity}


# ============================================================
# 第 7 步：ONNX 导出
# ============================================================

def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str = "model.onnx",
    opset_version: int = 17,
) -> str:
    """将 PyTorch 模型导出为 ONNX 格式。

    ONNX 是模型交换的中立格式，所有主流推理引擎（TensorRT、ONNX Runtime、
    OpenVINO、Core ML、TFLite）都支持从 ONNX 转换。

    opset_version=17 是当前（2026 年）最安全的默认值。
    dynamic_axes 允许在推理时使用任意批次大小。
    """
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True,  # 常量折叠优化
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

    import os
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"\n模型已导出到: {output_path}")
    print(f"文件大小: {file_size_kb:.1f} KB")
    return output_path


# ============================================================
# 第 8 步：边缘端架构对比
# ============================================================

def compare_edge_backbones(resolution: int = 160, device: str = "cpu") -> list:
    """对比多个轻量级骨干网络在 CPU 上的延迟表现。

    这是边缘部署决策的第一步：在没有测量之前，任何关于速度的判断都是幻觉。
    必须在使用目标硬件的情况下实际测量延迟、显存和准确率。
    """
    from torchvision.models import (
        mobilenet_v3_small, resnet18, efficientnet_v2_s, convnext_tiny,
    )

    candidates = [
        ("mobilenet_v3_small", mobilenet_v3_small(weights=None, num_classes=10)),
        ("resnet18", resnet18(weights=None, num_classes=10)),
        ("efficientnet_v2_s", efficientnet_v2_s(weights=None, num_classes=10)),
        ("convnext_tiny", convnext_tiny(weights=None, num_classes=10)),
    ]

    shape = (1, 3, resolution, resolution)
    results = []

    for name, model in candidates:
        params = count_parameters(model)
        flops = estimate_flops(model, shape)
        latency = measure_latency(model, shape, device=device)
        results.append({
            "model": name,
            "params_m": params / 1e6,
            "gflops": flops / 1e9,
            "p50_ms": latency["p50_ms"],
            "p95_ms": latency["p95_ms"],
            "p99_ms": latency["p99_ms"],
        })

    return results


# ============================================================
# 第 9 步：混合精度演示
# ============================================================

def demonstrate_mixed_precision(device: str = "cuda"):
    """演示 FP32 vs FP16 的性能差异（仅在 GPU 上有效）。

    混合精度训练/推理利用 Tensor Core 硬件加速 FP16 矩阵乘法，
    在保持精度的同时显著加速计算。

    注意：FP16 的表示范围是有限的（约 6e-5 ~ 65504），
    对于数值敏感的操作（如 softmax 前的未缩放分数），需要使用 FP32 累加。
    BF16 提供了与 FP32 相同的指数范围，但精度更低，
    在大多数深度学习任务中是比 FP16 更安全的选择。
    """
    if device != "cuda" or not torch.cuda.is_available():
        print("混合精度仅在有 CUDA GPU 的情况下有意义")
        return

    from torchvision.models import mobilenet_v3_small

    model_fp32 = mobilenet_v3_small(weights=None, num_classes=10)
    model_fp16 = mobilenet_v3_small(weights=None, num_classes=10).half()

    model_fp16.load_state_dict(model_fp32.state_dict())
    model_fp16 = model_fp16.to(device).eval()
    model_fp32 = model_fp32.to(device).eval()

    x = torch.randn(1, 3, 224, 224, device=device)
    x_fp16 = x.half()

    lat_fp32 = measure_latency(model_fp32, (1, 3, 224, 224), device=device)
    lat_fp16 = measure_latency(model_fp16, (1, 3, 224, 224), device=device)

    print(f"\nFP32 延迟 p50: {lat_fp32['p50_ms']:.2f} ms")
    print(f"FP16 延迟 p50: {lat_fp16['p50_ms']:.2f} ms")
    speedup = lat_fp32["p50_ms"] / lat_fp16["p50_ms"]
    print(f"加速比: {speedup:.2f}x")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("实时边缘部署：测量 -> 量化 -> 导出 -> 部署")
    print("=" * 60)

    # === 演示 1：架构对比 ===
    print("\n--- 步骤 1: 边缘骨干网络延迟对比 ---")
    results = compare_edge_backbones(resolution=160, device="cpu")

    header = (
        f"{'模型':<22} {'参数(M)':>8s} {'GFLOPs':>8s} "
        f"{'p50(ms)':>9s} {'p95(ms)':>9s} {'p99(ms)':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:22s} {r['params_m']:>8.2f} {r['gflops']:>8.2f} "
            f"{r['p50_ms']:>9.1f} {r['p95_ms']:>9.1f} {r['p99_ms']:>9.1f}"
        )

    # === 演示 2：深度可分离卷积参数效率 ===
    print("\n--- 步骤 2: 标准卷积 vs 深度可分离卷积 ---")
    compare_conv_approaches(in_ch=64, out_ch=128)

    # === 演示 3：通道混洗 ===
    demo_channel_shuffle()

    # === 演示 4：INT8 量化 ===
    demonstrate_quantization()

    # === 演示 5：剪枝 ===
    print("\n--- 步骤 5: 幅度剪枝演示 ---")
    from torchvision.models import shufflenet_v2_x1_0
    prune_model = shufflenet_v2_x1_0(weights=None, num_classes=10)
    stats = magnitude_pruning(prune_model, sparsity=0.5)

    # === 演示 6：ONNX 导出 ===
    print("\n--- 步骤 6: ONNX 导出 ---")
    from torchvision.models import mobilenet_v3_small
    model_for_export = mobilenet_v3_small(weights=None, num_classes=10)
    sample = torch.randn(1, 3, 224, 224)
    try:
        export_to_onnx(model_for_export, sample, "model_test.onnx")
        # 清理
        if os.path.exists("model_test.onnx"):
            os.remove("model_test.onnx")
    except Exception as e:
        print(f"ONNX 导出失败（可能是环境限制）: {e}")

    # === 演示 7：混合精度 ===
    print("\n--- 步骤 7: 混合精度 ---")
    if torch.cuda.is_available():
        demonstrate_mixed_precision()
    else:
        print("无可用 GPU，跳过混合精度演示")

    print("\n" + "=" * 60)
    print("所有演示完成。核心结论：")
    print("  1. 测量必须在目标硬件上进行——工作站上的数字不代表部署效果")
    print("  2. 深度可分离卷积让 MobileNet 的参数量仅为 ResNet 的 1/10")
    print("  3. INT8 量化可以在几乎不损失准确率的前提下将模型缩小 4 倍")
    print("  4. ONNX 是中立的模型交换格式，是所有边缘推理引擎的共同起点")
    print("=" * 60)
