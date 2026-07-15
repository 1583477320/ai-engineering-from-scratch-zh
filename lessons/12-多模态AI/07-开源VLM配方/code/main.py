# 开源 VLM 配方分析器

from collections import Counter


# VLM 设计空间配置
VLM_CONFIGS = {
    "LLaVA-1.5": {"encoder": "CLIP ViT-L/14", "connector": "MLP", "llm": "Vicuna-7B",
                  "resolution": 336, "data": "1.2M 指令"},
    "LLaVA-NeXT": {"encoder": "CLIP ViT-L/14", "connector": "MLP", "llm": "Vicuna-13B",
                   "resolution": "AnyRes", "data": "1.5M 指令"},
    "LLaVA-OneVision": {"encoder": "SigLIP", "connector": "MLP", "llm": "Qwen2-7B",
                        "resolution": "动态", "data": "1.6M 指令+视频"},
    "Molmo": {"encoder": "EVA-CLIP", "connector": "Perceiver", "llm": "OLMo-7B",
              "resolution": "动态", "data": "人工标注"},
    "Cambrian-1": {"encoder": "混合", "connector": "多连接器", "llm": "Llama-3",
                   "resolution": "动态", "data": "多数据集混合"},
}


def compare_encoders(configs):
    """对比不同配置的编码器选择。"""
    encoder_counts = Counter(c["encoder"] for c in configs.values())
    print("编码器选择统计:")
    for enc, count in encoder_counts.most_common():
        print(f"  {enc}: {count} 个模型使用")


def analyze_design_space(configs):
    """分析设计空间的分布。"""
    print("VLM 设计空间分析:")
    for key in ["encoder", "connector", "resolution"]:
        values = Counter(c[key] for c in configs.values())
        print(f"\n{key}:")
        for val, count in values.most_common():
            print(f"  {val}: {count}")


if __name__ == "__main__":
    print("开源 VLM 配方分析\n")
    compare_encoders(VLM_CONFIGS)
    analyze_design_space(VLM_CONFIGS)

    print("\n关键发现:")
    print("  1. 编码器选择 > 连接器架构")
    print("  2. 数据质量 > 数据数量")
    print("  3. 人工标题 > 合成数据")
