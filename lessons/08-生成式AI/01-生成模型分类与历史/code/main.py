"""生成模型分类与历史——可视化工具"""

def model_family_tree():
    """生成模型家族分类可视化"""
    families = {
        "显式密度(可处理)": {
            "代表": ["GPT (自回归)", "WaveNet", "PixelCNN", "Glow (流)"],
            "优点": "精确似然、干净的训练损失",
            "缺点": "自回归推理串行(慢)；流需要可逆架构"
        },
        "显式密度(近似)": {
            "代表": ["VAE", "DDPM", "Flow Matching"],
            "优点": "可处理似然的下界(ELBO)，扩散模型是2026图像/视频主力",
            "缺点": "近似似然，ELBO 不等于真实似然"
        },
        "隐式密度": {
            "代表": ["GAN", "StyleGAN 1/2/3"],
            "优点": "推理快(一次前向)，样本锐利",
            "缺点": "训练不稳定，模式坍塌，无似然"
        },
        "基于分数/连续时间": {
            "代表": ["Score Matching", "SDE", "Flow Matching"],
            "优点": "无模拟训练，更直的路径，采样快 4-10 倍",
            "缺点": "需要学习分数函数，理论复杂"
        },
        "基于 token 的自回归": {
            "代表": ["VQ-VAE + Transformer", "Sora", "VALL-E"],
            "优点": "高维数据→离散 token 序列，Transformer 建模",
            "缺点": "两阶段：量化+自回归，质量受限于量化器"
        }
    }

    print("=== 五大生成模型家族 ===\n")
    for family, info in families.items():
        print(f"【{family}】")
        print(f"  代表: {', '.join(info['代表'])}")
        print(f"  优点: {info['优点']}")
        print(f"  缺点: {info['缺点']}")
        print()

def history_timeline():
    """生成模型历史时间线"""
    milestones = [
        (2013, "VAE (Kingma)", "第一个有可用训练损失的深度生成模型"),
        (2014, "GAN (Goodfellow)", "隐式密度，无似然——样本惊人地锐利"),
        (2017, "Progressive GAN", "第一张百万像素人脸"),
        (2019, "StyleGAN/StyleGAN2", "人脸照片级真实感至今难超越"),
        (2020, "DDPM (Ho)", "扩散变得实际可用"),
        (2022, "Stable Diffusion 1", "潜在扩散 + 文本条件 = 商品化"),
        (2024, "Sora, SD3, Flow Matching", "视频扩散；Flow Matching 胜出"),
        (2026, "Consistency + Rectified Flow", "从扩散骨干一步采样"),
    ]

    print("=== 生成模型历史里程碑 ===\n")
    for year, model, desc in milestones:
        print(f"  {year}: {model}")
        print(f"         {desc}")
        print()

if __name__ == "__main__":
    model_family_tree()
    print()
    history_timeline()
