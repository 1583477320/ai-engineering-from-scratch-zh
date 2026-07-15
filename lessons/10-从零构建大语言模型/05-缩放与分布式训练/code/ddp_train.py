# 分布式训练：数据并行 + 显存计算 + FSDP 模拟
# 演示 DDP、张量并行、流水线并行的核心概念

import numpy as np


# ============================================================================
# 第 1 步：数据并行模拟
# ============================================================================

def simulate_data_parallelism(data, num_gpus, model_fn):
    """模拟数据并行——每个 GPU 处理一个数据分片。"""
    shard_size = len(data) // num_gpus
    gpu_losses = []
    gpu_gradients = []

    offset = 0
    for gpu_id in range(num_gpus):
        shard = data[offset:offset + shard_size]
        offset += shard_size
        loss, grad = model_fn(shard)
        gpu_losses.append(loss)
        gpu_gradients.append(grad)

    # AllReduce：平均梯度
    avg_loss = np.mean(gpu_losses)
    avg_gradient = np.mean(gpu_gradients, axis=0)
    return avg_loss, avg_gradient


# ============================================================================
# 第 2 步：张量并行模拟
# ============================================================================

def simulate_tensor_parallelism(input_data, weight_matrix, num_gpus):
    """模拟张量并行——权重矩阵按列拆分到多个 GPU。"""
    d_in, d_out = weight_matrix.shape
    shard_size = d_out // num_gpus

    partial_results = []
    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = start + shard_size
        weight_shard = weight_matrix[:, start:end]
        partial = input_data @ weight_shard
        partial_results.append(partial)

    full_output = np.concatenate(partial_results, axis=-1)
    direct_output = input_data @ weight_matrix
    error = np.abs(full_output - direct_output).max()
    return full_output, error


# ============================================================================
# 第 3 步：流水线并行模拟
# ============================================================================

def simulate_pipeline_parallelism(num_layers, num_stages, num_microbatches):
    """模拟流水线并行——模型层分配到不同 GPU，微批次流水线执行。"""
    layers_per_stage = num_layers // num_stages
    timeline = {}

    # 前向传播
    for mb in range(num_microbatches):
        for stage in range(num_stages):
            start = max(
                timeline.get((stage, mb - 1, "fwd"), (0, 0))[1] if mb > 0 else 0,
                timeline.get((stage - 1, mb, "fwd"), (0, 0))[1] if stage > 0 else 0,
            )
            end = start + layers_per_stage
            timeline[(stage, mb, "fwd")] = (start, end)

    last_fwd_end = max(v[1] for v in timeline.values())

    # 反向传播
    for mb in range(num_microbatches - 1, -1, -1):
        for stage in range(num_stages - 1, -1, -1):
            deps = [last_fwd_end]
            if mb < num_microbatches - 1 and (stage, mb + 1, "bwd") in timeline:
                deps.append(timeline[(stage, mb + 1, "bwd")][1])
            if stage < num_stages - 1 and (stage + 1, mb, "bwd") in timeline:
                deps.append(timeline[(stage + 1, mb, "bwd")][1])
            start = max(deps)
            end = start + layers_per_stage
            timeline[(stage, mb, "bwd")] = (start, end)

    total_time = max(v[1] for v in timeline.values())
    compute_time = num_microbatches * num_stages * layers_per_stage * 2
    bubble_fraction = 1.0 - compute_time / (total_time * num_stages)
    return timeline, total_time, bubble_fraction


# ============================================================================
# 第 4 步：显存计算器
# ============================================================================

def memory_calculator(params_billions, precision_bytes=2, optimizer="adam",
                     num_gpus=1, sharding="none"):
    """计算给定模型大小和配置下的显存需求。"""
    params = params_billions * 1e9
    weight_memory = params * precision_bytes

    if optimizer == "adam":
        optimizer_memory = params * 4 * 2  # 一阶矩 + 二阶矩
    else:
        optimizer_memory = params * 4

    gradient_memory = params * precision_bytes
    activation_memory = params * precision_bytes * 0.5  # 估算

    # FSDP / ZeRO 分片
    if sharding in ("fsdp", "zero3"):
        weight_memory /= num_gpus
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero2":
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero1":
        optimizer_memory /= num_gpus

    per_gpu_total = weight_memory + optimizer_memory + gradient_memory + activation_memory
    return {
        "params_billions": params_billions,
        "weights_gb": weight_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "per_gpu_total_gb": per_gpu_total / 1e9,
        "fits_on_80gb": per_gpu_total / 1e9 <= 80,
    }


# ============================================================================
# 第 5 步：混合精度对比
# ============================================================================

def mixed_precision_comparison(params_billions):
    """对比 FP32、FP16、BF16 混合精度的显存占用。"""
    params = params_billions * 1e9

    fp32_total = params * 4 + params * 4 * 2 + params * 4  # 权重+优化器+梯度
    mixed_total = params * 2 + params * 4 * 2 + params * 2  # BF16 权重+FP32 优化器+BF16 梯度

    return {
        "fp32_total_gb": fp32_total / 1e9,
        "mixed_bf16_gb": mixed_total / 1e9,
        "savings_vs_fp32": 1 - mixed_total / fp32_total,
    }


# ============================================================================
# 第 6 步：训练成本估算
# ============================================================================

def training_cost_estimator(params_billions, tokens_trillions, gpu_type="h100", num_gpus=512):
    """估算训练成本。"""
    gpu_specs = {
        "a100": {"tflops": 312, "cost_per_hour": 2.00, "memory_gb": 80},
        "h100": {"tflops": 990, "cost_per_hour": 3.50, "memory_gb": 80},
        "h200": {"tflops": 990, "cost_per_hour": 4.50, "memory_gb": 141},
    }
    spec = gpu_specs[gpu_type]
    params = params_billions * 1e9
    tokens = tokens_trillions * 1e12
    flops_total = 6 * params * tokens
    flops_per_gpu = spec["tflops"] * 1e12 * 0.4  # 40% 利用率

    total_gpu_hours = flops_total / (flops_per_gpu * num_gpus) / 3600
    total_cost = total_gpu_hours * num_gpus * spec["cost_per_hour"]

    return {
        "model_size": f"{params_billions}B",
        "tokens": f"{tokens_trillions}T",
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "wall_clock_days": total_gpu_hours / 24,
        "total_gpu_hours": total_gpu_hours * num_gpus,
        "estimated_cost": total_cost,
    }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. 数据并行
    print("=" * 70)
    print("数据并行模拟")
    print("=" * 70)
    data = np.random.randn(64, 32)
    weight = np.random.randn(32, 16)

    def model_fn(batch):
        output = batch @ weight
        loss = np.mean(output ** 2)
        grad = 2 * batch.T @ (batch @ weight) / len(batch)
        return loss, grad

    for n_gpus in [1, 2, 4, 8]:
        loss, grad = simulate_data_parallelism(data, n_gpus, model_fn)
        print(f"  {n_gpus} GPU: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}")

    # 2. 张量并行
    print("\n" + "=" * 70)
    print("张量并行模拟")
    print("=" * 70)
    x = np.random.randn(4, 8192)
    W = np.random.randn(8192, 8192)
    for n_gpus in [1, 2, 4, 8]:
        output, error = simulate_tensor_parallelism(x, W, n_gpus)
        print(f"  {n_gpus} GPU: shape={output.shape}, max_error={error:.2e}")

    # 3. 流水线并行
    print("\n" + "=" * 70)
    print("流水线并行模拟（32层，4阶段）")
    print("=" * 70)
    for n_mb in [1, 4, 8, 16, 32]:
        _, total_t, bubble = simulate_pipeline_parallelism(32, 4, n_mb)
        print(f"  {n_mb:2d} 微批次: 总时间={total_t:4d}, 气泡={bubble:.1%}")

    # 4. 显存计算器
    print("\n" + "=" * 70)
    print("显存计算器")
    print("=" * 70)
    configs = [
        (7, "none", 1), (7, "fsdp", 8),
        (70, "none", 1), (70, "fsdp", 8), (70, "fsdp", 16),
        (405, "fsdp", 64), (405, "fsdp", 128),
    ]
    print(f"  {'模型':>6} {'分片':>6} {'GPU':>5} {'单卡显存':>10} {'80GB可用':>10}")
    for params, shard, gpus in configs:
        r = memory_calculator(params, num_gpus=gpus, sharding=shard)
        fits = "是" if r["fits_on_80gb"] else "否"
        print(f"  {params:>5}B {shard:>6} {gpus:>5} {r['per_gpu_total_gb']:>8.1f}GB {fits:>10}")

    # 5. 混合精度对比
    print("\n" + "=" * 70)
    print("混合精度显存对比")
    print("=" * 70)
    for params_b in [7, 70, 405]:
        r = mixed_precision_comparison(params_b)
        print(f"  {params_b}B: FP32={r['fp32_total_gb']:.0f}GB, "
              f"BF16混合={r['mixed_bf16_gb']:.0f}GB, "
              f"节省={r['savings_vs_fp32']:.0%}")

    # 6. 训练成本
    print("\n" + "=" * 70)
    print("训练成本估算")
    print("=" * 70)
    estimates = [
        (8, 15.0, "h100", 512),
        (70, 15.0, "h100", 2048),
        (405, 15.0, "h100", 16384),
    ]
    print(f"  {'模型':>8} {'数据':>6} {'GPU':>6} {'天数':>8} {'成本':>14}")
    for params, tokens, gpu, n_gpus in estimates:
        r = training_cost_estimator(params, tokens, gpu, n_gpus)
        print(f"  {params:>6}B {tokens:>6.1f}T {n_gpus:>6} {r['wall_clock_days']:>7.0f}天 ${r['estimated_cost']:>13,.0f}")
