# main.py — Stable Diffusion 流水线演示
# 依赖：torch>=2.0, diffusers>=0.25, transformers>=4.36, accelerate, Pillow
# 安装：pip install torch diffusers transformers accelerate Pillow
# 对应课程：阶段 04（计算机视觉）· 11（Stable Diffusion）

import os
import math


# ===================================================================
# 第 1 部分：扩散模型参数计算工具
# ===================================================================

def calculate_diffusion_params():
    """计算扩散模型各阶段的张量形状和参数量（不依赖 PyTorch）。"""

    # 图像到潜空间的映射
    # SD 1.5 / 2.x：512x512 RGB -> 4x64x64 潜空间
    image_pixels = 3 * 512 * 512      # 786,432
    latent_dims = 4 * 64 * 64          # 16,384
    compression_ratio = image_pixels / latent_dims

    print("=" * 60)
    print("扩散模型张量形状与压缩比")
    print("=" * 60)
    print(f"  原始图像：{image_pixels:,} 个像素值（3 x 512 x 512）")
    print(f"  潜空间：  {latent_dims:,} 个维度  （4 x 64 x 64）")
    print(f"  压缩比：  {compression_ratio:.0f}x  （节省 {(1 - 1/compression_ratio)*100:.1f}% 计算量）")
    print()

    # VAE scale factor
    vae_scale_factor_15 = 0.18215
    vae_scale_factor_sdxl = 0.13025
    print(f"  VAE 缩放因子（SD 1.5）：{vae_scale_factor_15}")
    print(f"  VAE 缩放因子（SDXL）：  {vae_scale_factor_sdxl}")
    print()

    # U-Net 参数量对比
    models = [
        ("SD 1.5", 860, "CLIP-L (文本编码器)", "FP16"),
        ("SDXL", 2600, "CLIP-L + CLIP-G", "FP16"),
        ("SD3", 8000, "T5-XXL + CLIP", "BF16"),
        ("FLUX.1-dev", 12000, "T5-XXL", "BF16"),
    ]

    print("=" * 60)
    print("U-Net 参数量对比（单位：百万）")
    print("=" * 60)
    print(f"  {'模型':<15} {'参数(M)':<10} {'文本编码器':<25} {'精度'}")
    for name, params, encoder, precision in models:
        print(f"  {name:<15} {params:>8,}   {encoder:<25} {precision}")
    print()

    # 生成时间估算
    print("=" * 60)
    print("单图生成时间估算（RTX 4090，FP16）")
    print("=" * 60)
    steps_and_time = [
        ("DDIM 50 步", 50, "约 7.5 秒"),
        ("DPM-Solver++ 20 步", 20, "约 3.0 秒"),
        ("Euler Ancestral 30 步", 30, "约 4.5 秒"),
        ("LCM 4 步", 4, "约 0.6 秒"),
    ]
    for name, steps, time_est in steps_and_time:
        per_step_ms = 150  # RTX 4090 近似值
        total_ms = steps * per_step_ms
        print(f"  {name}: 每步 ~{per_step_ms}ms，总计 {time_est}")
    print()


# ===================================================================
# 第 2 部分：CFG（无分类器引导）参数扫描演示
# ===================================================================

def cfg_guide_analysis():
    """分析 CFG scale 不同值对生成结果的影响。"""

    guidance_values = [0.0, 1.0, 3.0, 5.0, 7.5, 10.0, 15.0]

    descriptions = {
        0.0: "无条件采样 — 完全忽略提示词，随机图像",
        1.0: "纯条件预测 — 提示词仅轻微影响，多样性高但语义弱",
        3.0: "低引导 — 提示词开始生效，但仍有大量创造性偏差",
        5.0: "中低引导 — 平衡点偏创意方向",
        7.5: "标准引导 — SD 默认值，语义忠告性与多样性的最佳平衡",
        10.0: "强引导 — 高度忠实于提示词，颜色可能偏饱和",
        15.0: "过度引导 — 超出 VAE 解码流形，出现伪影和色带化",
    }

    print("=" * 60)
    print("CFG Scale（无分类器引导系数）影响分析")
    print("=" * 60)
    print(f"  {'CFG Scale':<12} {'描述'}")
    print("  " + "-" * 58)
    for w in guidance_values:
        print(f"  {w:<12.1f} {descriptions.get(w, '未知')}")
    print()

    # 解释公式
    print("  核心公式：eps = eps_uncond + w * (eps_cond - eps_uncond)")
    print("  其中 w 就是 guidance_scale。当 w > 1 时，放大条件方向，")
    print("  使输出更贴近提示词描述的内容。但过大的 w 会将潜变量推出")
    print("  VAE 解码流形，导致视觉伪影。")
    print()


# ===================================================================
# 第 3 部分：从噪声逐步重建的示意
# ===================================================================

def latent_diffusion_process_demo():
    """展示潜空间扩散的核心步骤——从数学角度（纯 Python）。"""

    num_steps = 10

    print("=" * 60)
    print("潜空间扩散前向过程示意（以 8x8 潜层为例）")
    print("=" * 60)
    latent_dim = 4 * 8 * 8
    print(f"  潜张量形状：(1, 4, 8, 8)")
    print(f"  总维度：{latent_dim}（对比原图的 512x512x3 = 786,432）")
    print()

    # 线性 beta 噪声调度（纯 Python）
    beta_values = [1e-4 + i * (0.02 - 1e-4) / (num_steps - 1) for i in range(num_steps)]
    alpha_values = [1.0 - b for b in beta_values]

    # 累积保留因子 alpha_bar
    alpha_bar_values = []
    running_prod = 1.0
    for a in alpha_values:
        running_prod *= a
        alpha_bar_values.append(running_prod)

    print("  时间步 | beta(t)  | sqrt(alpha_bar_t) | sqrt(1-alpha_bar_t)")
    print("  " + "-" * 60)
    for t in range(num_steps):
        sqrt_ab = math.sqrt(alpha_bar_values[t])
        sqrt_1ab = math.sqrt(1.0 - alpha_bar_values[t])
        print(f"    {t+1:2d}     | {beta_values[t]:.5f}    | {sqrt_ab:.4f}"
              f"                | {sqrt_1ab:.4f}")
    print()
    print("  当 t 接近末尾时，sqrt(alpha_bar_t) -> 0，信号几乎消失，")
    print("  x_t approx sqrt(1 - alpha_bar_t) * epsilon，即近似纯高斯噪声。")
    print()


# ===================================================================
# 第 4 部分：LoRA 微调参数示意
# ===================================================================

def lora_parameter_demo():
    """展示 LoRA 微调如何大幅减少需要更新的参数量。"""

    unet_total_params = 860_000_000  # SD 1.5
    lora_rank_range = [4, 8, 16, 32]

    print("=" * 60)
    print("LoRA 微调参数量分析（SD 1.5，8.6 亿参数）")
    print("=" * 60)
    print(f"  完整 U-Net 参数量：{unet_total_params:,}")
    print(f"  LoRA 仅注入到 Self-Attention 和 Cross-Attention 层")
    print()
    print(f"  {'秩(r)':<8} {'A矩阵参数量':<18} {'B矩阵参数量':<18} {'总新增参数量':<18} {'占原始比例'}")
    print("  " + "-" * 80)

    for r in lora_rank_range:
        d_attn = 320  # SD 1.5 attention 维度（最低层）
        # 每层有 Q/K/V/O 四个投影，简化估算
        # 假设有 ~20 个注意力层受影响
        num_layers = 20
        params_a = d_attn * r * num_layers
        params_b = r * d_attn * num_layers
        total_new = params_a + params_b
        ratio = total_new / unet_total_params * 100
        print(f"  {r:<8} {params_a:<18,} {params_b:<18,} {total_new:<18,} {ratio:.4f}%")

    print()
    print("  实际 LoRA 文件大小（含优化器状态）通常在 10-50 MB。")
    print("  微调可在 8GB 显存的消费级 GPU 上 10-60 分钟内完成。")
    print()


# ===================================================================
# 第 5 部分：工业流水线概览
# ===================================================================

def describe_industrial_pipeline():
    """描述生产环境中的 Stable Diffusion 流水线。"""

    pipeline_components = [
        {
            "组件": "提示词编码",
            "SD 1.5": "CLIP ViT-L/14 (768维)",
            "SDXL": "CLIP-L (1024维) + CLIP-G (1280维)",
            "SD3/FLUX": "T5-XXL (4096维)",
        },
        {
            "组件": "U-Net 架构",
            "SD 1.5": "自注意力 + 交叉注意力 × 18",
            "SDXL": "Refiner-Base 双分支",
            "SD3": "Transformer (DiT)",
            "FLUX": "Transformer + Context Parallelism",
        },
        {
            "component": "采样步数",
            "SD 1.5": "DPM-Solver++ 20-30 步",
            "SDXL": "DPM-Solver++ 25-30 步",
            "SD3": "DDIM 50 步 / LCM 4 步",
            "FLUX": "Fast/Dev/Degiz 4-50 步可调",
        },
    ]

    print("=" * 60)
    print("工业流水线组件对比")
    print("=" * 60)
    for row in pipeline_components:
        print(f"\n  组件：{row.get('组件', row.get('component'))}")
        print(f"  {'模型':<10} 配置")
        for key, value in row.items():
            if key not in ("组件", "component"):
                print(f"    {key}: {value}")
    print()

    # 推荐配置表
    print("=" * 60)
    print("生产环境推荐配置速查")
    print("=" * 60)
    print(f"  {'场景':<20} {'模型':<10} {'精度':<8} {'采样器':<25} {'步数'}")
    configs = [
        ("快速预览", "SDXL", "FP16", "DPM-Solver++ 2M", "20"),
        ("高质量产出", "SDXL", "BF16", "DPM-Solver++ 2M", "30"),
        ("极低延迟 (<1s)", "FLUX-fast", "FP8", "LCM", "4"),
        ("极致质量", "FLUX-dev", "BF16", "DPM-Solver++", "50"),
        ("社区生态", "SD 1.5", "FP16", "Euler Ancestral", "30"),
    ]
    for scene, model, precision, scheduler, steps in configs:
        print(f"  {scene:<20} {model:<10} {precision:<8} {scheduler:<25} {steps}")
    print()


# ===================================================================
# 主程序入口
# ===================================================================

def main():
    print("\nStable Diffusion 教学演示程序")
    print("=" * 60)
    print()

    calculate_diffusion_params()
    cfg_guide_analysis()
    latent_diffusion_process_demo()
    lora_parameter_demo()
    describe_industrial_pipeline()

    print("=" * 60)
    print("演示完成。要实际运行推理，需要 GPU 和 diffusers 库。")
    print("=" * 60)


if __name__ == "__main__":
    main()
