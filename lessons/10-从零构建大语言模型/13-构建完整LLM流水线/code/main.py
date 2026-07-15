# 完整 LLM 流水线演示——从选择基础模型到部署

import numpy as np


# ============================================================================
# 第 1 步：基础模型选型
# ============================================================================

MODEL_CATALOG = {
    "Llama 3.1 8B": {"params": 8, "vocab": 128256, "context": 8192, "strength": "通用英文"},
    "Qwen2.5 7B": {"params": 7, "vocab": 151936, "context": 32768, "strength": "中文"},
    "Mistral 7B": {"params": 7, "vocab": 32000, "context": 8192, "strength": "高效推理"},
    "Phi-3 Mini": {"params": 3.8, "vocab": 32064, "context": 4096, "strength": "轻量级"},
}


def suggest_model(use_case):
    """根据使用场景推荐基础模型。"""
    model_scores = {}
    for name, info in MODEL_CATALOG.items():
        score = 0
        if use_case == "中文" and "中文" in info["strength"]:
            score += 3
        if use_case == "通用英文" and "通用" in info["strength"]:
            score += 3
        if use_case == "轻量部署":
            score += 5 - info["params"]
        if use_case == "长上下文" and info["context"] >= 8192:
            score += 2
        model_scores[name] = score

    best = max(model_scores, key=model_scores.get)
    return best, MODEL_CATALOG[best]


# ============================================================================
# 第 2 步：LoRA 参数计算
# ============================================================================

def lora_parameter_count(base_params, rank=16, target_modules=4):
    """计算 LoRA 微调的可训练参数量。"""
    base = base_params * 1e9
    per_module_params = base * 0.01  # 假设 FFN 占 1%
    lora_params = per_module_params * target_modules * 2 * rank / (2048 * 2)
    return lora_params / 1e6  # 返回百万参数


# ============================================================================
# 第 3 步：训练成本估算
# ============================================================================

def training_cost_estimate(params_b, tokens_t, gpu_type="h100", gpus=8):
    """估算训练成本和时间。"""
    specs = {"a100": 312, "h100": 990, "h200": 990}
    tflops = specs.get(gpu_type, 312)
    flops = 6 * params_b * 1e9 * tokens_t * 1e12
    gpu_seconds = flops / (tflops * 1e12 * 0.4) / gpus
    days = gpu_seconds / 86400

    return {
        "total_flops": f"{flops:.2e}",
        "days": f"{days:.1f}",
        "gpu_hours": f"{gpu_seconds*gpus/3600:.0f}",
    }


# ============================================================================
# 第 4 步：完整流水线
# ============================================================================

def full_pipeline(use_case="中文"):
    """演示完整 LLM 流水线。"""
    print(f"== 完整 LLM 流水线 (场景: {use_case}) ==\n")

    # 1. 选模型
    best, info = suggest_model(use_case)
    print(f"1. 基础模型选择: {best}")
    print(f"   参数量: {info['params']}B, 词表: {info['vocab']}, 上下文: {info['context']}")

    # 2. LoRA 参数
    lora_params = lora_parameter_count(info["params"], rank=16, target_modules=4)
    total_params = info["params"] * 1e3
    print(f"\n2. LoRA 微调")
    print(f"   可训练参数: {lora_params:.1f}M (占总参数的 {lora_params/(total_params):.2%})")

    # 3. 训练成本
    if info["params"] >= 7:
        cost = training_cost_estimate(info["params"], 1.0, "h100", 8)
        print(f"\n3. SFT 训练估算 (1T token)")
        print(f   f"   估计时间: {cost['days']} 天, GPU小时: {cost['gpu_hours']}")

    # 4. DPO
    print(f"\n4. DPO 对齐 (推荐 > RLHF)")
    print(f"   数据需求: 1K-10K 偏好对")

    # 5. 量化
    fp16_mem = info["params"] * 1e9 * 2 / 1e9
    int4_mem = info["params"] * 1e9 * 0.5 / 1e9
    print(f"\n5. 量化")
    print(f"   FP16 推理: {fp16_mem:.1f}GB")
    print(f"   INT4 推理: {int4_mem:.1f}GB (推荐)")

    # 6. 部署
    print(f"\n6. 部署")
    print(f"   vLLM (GPU) / llama.cpp (CPU)")
    print(f"   推荐引擎: vLLM")

    return best


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("完整 LLM 流水线规划器")
    print("=" * 60)

    # 中文场景
    full_pipeline("中文")

    print("\n" + "=" * 60)
    # 轻量部署
    full_pipeline("轻量部署")

    print("\n\n推荐阅读:")
    print("  1. LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory")
    print("  2. vLLM: https://github.com/vllm-project/vllm")
    print("  3. Axolotl: https://github.com/OpenAccess-AI-Collective/axolotl")
