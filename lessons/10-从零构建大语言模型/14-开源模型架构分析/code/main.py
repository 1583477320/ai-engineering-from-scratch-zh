# 开源模型架构分析——Llama 3、Qwen2.5、Mistral、Phi-3、DeepSeek-V3

import numpy as np


# ============================================================================
# 第 1 步：模型架构配置数据库
# ============================================================================

MODEL_CONFIGS = {
    "Llama 3.1 8B": {
        "layers": 32, "hidden": 4096, "heads": 32, "kv_heads": 8,
        "vocab": 128256, "norm": "RMSNorm", "activation": "SwiGLU",
        "pe": "RoPE", "attention": "GQA", "mlp_ratio": 3.5,
        "params": 8.0, "training_tokens": "15T",
    },
    "Llama 3.1 70B": {
        "layers": 80, "hidden": 8192, "heads": 64, "kv_heads": 8,
        "vocab": 128256, "norm": "RMSNorm", "activation": "SwiGLU",
        "pe": "RoPE", "attention": "GQA", "mlp_ratio": 3.5,
        "params": 70.0, "training_tokens": "15T",
    },
    "Qwen2.5 7B": {
        "layers": 28, "hidden": 3584, "heads": 28, "kv_heads": 4,
        "vocab": 151936, "norm": "RMSNorm", "activation": "SwiGLU",
        "pe": "RoPE", "attention": "GQA", "mlp_ratio": 4.0,
        "params": 7.0, "training_tokens": "18T",
    },
    "Mistral 7B": {
        "layers": 32, "hidden": 4096, "heads": 32, "kv_heads": 8,
        "vocab": 32000, "norm": "RMSNorm", "activation": "SwiGLU",
        "pe": "RoPE", "attention": "SWA+GQA", "mlp_ratio": 3.5,
        "params": 7.0, "training_tokens": "8T",
    },
    "Phi-3 Mini": {
        "layers": 32, "hidden": 3072, "heads": 24, "kv_heads": 24,
        "vocab": 32064, "norm": "LayerNorm", "activation": "GELU",
        "pe": "RoPE", "attention": "MHA", "mlp_ratio": 4.0,
        "params": 3.8, "training_tokens": "3.3T",
    },
    "DeepSeek-V3": {
        "layers": 60, "hidden": 7168, "heads": 128, "kv_heads": 128,
        "vocab": 129280, "norm": "RMSNorm", "activation": "SwiGLU",
        "pe": "RoPE", "attention": "MLA", "mlp_ratio": 3.5,
        "params": 671.0, "training_tokens": "14.8T",
        "architecture": "MoE(256专家,激活8)",
    },
}


# ============================================================================
# 第 2 步：参数量计算器
# ============================================================================

def calculate_parameters(cfg):
    """估算模型参数量。"""
    d = cfg["hidden"]
    V = cfg["vocab"]
    L = cfg["layers"]
    h = cfg["heads"]
    kv = cfg["kv_heads"]
    ff = int(d * cfg["mlp_ratio"])

    # 嵌入层
    tok_emb = V * d
    # 每层
    attn_q = d * h * d // h * 3  # Q/K/V
    attn_o = d * d
    ff_w1 = d * ff
    ff_w2 = ff * d
    per_layer = attn_q + attn_o + ff_w1 + ff_w2
    total = (tok_emb + L * per_layer) / 1e9
    return total


def print_architecture_comparison():
    """打印所有模型的架构对比。"""
    print(f"{'模型':<22} {'层':>4} {'维度':>6} {'头':>5} {'KV头':>5} {'词表':>8} {'参数量':>8} {'激活':>12} {'注意':>12}")
    print("-" * 90)
    for name, cfg in MODEL_CONFIGS.items():
        calc_params = calculate_parameters(cfg)
        param_str = f"{cfg['params']:.1f}B"
        act = cfg.get("activation", "SwiGLU")
        attn = cfg.get("attention", "GQA")
        print(f"{name:<22} {cfg['layers']:>4} {cfg['hidden']:>6} {cfg['heads']:>5} "
              f"{cfg['kv_heads']:>5} {cfg['vocab']:>8} {param_str:>8} {act:>12} {attn:>12}")
    print(f"\n注: MoE 模型(如DeepSeek-V3)的总参数量远大于激活参数量。")


# ============================================================================
# 第 3 步：架构分析
# ============================================================================

def analyze_gqa_ratio(model_name):
    """分析 GQA 分组的 KV 缓存节省。"""
    cfg = MODEL_CONFIGS[model_name]
    ratio = cfg["heads"] / cfg["kv_heads"]
    cache_saving = 1 - 1/ratio
    print(f"{model_name}:  {cfg['heads']}头/{cfg['kv_heads']}KV头 = {ratio:.0f}:1 GQA 比例")
    print(f"  KV 缓存节省: {cache_saving:.0%}")


def print_vocab_comparison():
    """打印词表大小对比。"""
    print(f"\n词表大小对嵌入层的影响:")
    print(f"{'模型':<22} {'词表':>8} {'嵌入维度':>8} {'嵌入参数':>12}")
    for name, cfg in MODEL_CONFIGS.items():
        embed_params = cfg["vocab"] * cfg["hidden"]
        print(f"{name:<22} {cfg['vocab']:>8} {cfg['hidden']:>8} {embed_params/1e6:>10.0f}M")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("开源 LLM 架构分析")
    print("=" * 70)

    # 架构对比
    print("\n1. 架构参数对比")
    print_architecture_comparison()

    # GQA 分析
    print("\n\n2. GQA (分组查询注意力) 对比")
    for model in ["Llama 3.1 8B", "Llama 3.1 70B", "Qwen2.5 7B"]:
        analyze_gqa_ratio(model)
        print()

    # 词表对比
    print("3. 词表大小分析")
    print_vocab_comparison()

    # 总结
    print("\n\n4. 选型建议")
    print(f"  通用英文 → Llama 3.1 8B/70B")
    print(f"  中文场景 → Qwen2.5 7B/14B")
    print(f"  边缘设备 → Phi-3 Mini")
    print(f"  高吞吐推理 → Mistral/Mixtral")
    print(f"  极致质量 → Llama 3.1 405B / DeepSeek-V3")
