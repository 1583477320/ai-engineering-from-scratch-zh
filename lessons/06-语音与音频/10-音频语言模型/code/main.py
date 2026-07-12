# 音频语言模型概念演示：三组件架构可视化
# 对应课程：阶段 06 · 10
"""
音频语言模型（LALM）由三部分组成：
1. 音频编码器（Whisper/BEATs）→ 音频特征
2. 投影层（线性/MLP）→ 映射到 LLM token 空间
3. LLM（Qwen/Llama）→ 文本输出
"""

import math
from typing import List, Dict


def audio_encoder(audio: List[float], sr: int = 16000, chunk_ms: int = 20) -> Dict:
    """模拟音频编码器——将波形压缩为特征向量。"""
    samples_per_chunk = int(sr * chunk_ms / 1000)
    n_chunks = len(audio) // samples_per_chunk
    return {
        "n_frames": n_chunks,
        "frame_dim": 16,  # 编码器输出维度
        "content": [sum(audio[i:i+samples_per_chunk]) / samples_per_chunk
                   for i in range(0, min(n_chunks * samples_per_chunk, len(audio)), samples_per_chunk)]
    }


def projector(features: Dict, llm_dim: int = 512) -> List[float]:
    """投影层：将音频特征映射到 LLM token 空间。"""
    # 简化：随机投影模拟线性层
    return [features["content"][i % len(features["content"])]
            for i in range(llm_dim)]


def llm_inference(audio_tokens: List[float], text_context: str) -> str:
    """模拟 LLM 推理——实际上需要 Qwen/Llama 模型。"""
    if "bark" in text_context.lower() or "狗" in text_context:
        return "检测到狗叫声。没有检测到人类语音。"
    elif "音乐" in text_context.lower() or "music" in text_context.lower():
        return "检测到钢琴旋律，约 120 BPM。"
    return f"音频分析完成：{len(audio_tokens)} 个特征向量。"


def main():
    # 模拟 1 秒 16kHz 单声道音频
    sr = 16000
    audio = [math.sin(2 * math.pi * 440 * i / sr) for i in range(sr)]

    print("=== 音频语言模型三组件演示 ===\n")

    # Step 1: 音频编码
    features = audio_encoder(audio, sr)
    print(f"Step 1 音频编码器: {len(audio)} 采样点 → {features['n_frames']} 帧")
    print(f"  帧维度: {features['frame_dim']} (模拟 Whisper/BEATs)")

    # Step 2: 投影
    audio_tokens = projector(features, llm_dim=512)
    print(f"\nStep 2 投影层: {features['frame_dim']}d → 512d (LLM token 空间)")

    # Step 3: LLM 推理
    print(f"\nStep 3 LLM 推理:")
    for ctx in ["这是一段狗叫声", "这段音频里有钢琴音乐", "安静的背景噪声"]:
        response = llm_inference(audio_tokens, ctx)
        print(f"  上下文: {ctx}")
        print(f"  输出: {response}")

    print("\n=== 2026 模型对比 ===")
    models = [
        ("Qwen2.5-Omni-7B", "Qwen2.5-7B", "文本+语音", "Apache-2.0"),
        ("Audio Flamingo 3", "Qwen2", "文本", "NVIDIA 非商用"),
        ("SALMONN", "Vicuna", "文本", "Apache-2.0"),
        ("GPT-4o", "GPT-4o", "文本+语音", "API"),
    ]
    print(f"  {'模型':<22} {'骨干':<14} {'输出':<10} {'许可'}")
    for name, backbone, output, license in models:
        print(f"  {name:<22} {backbone:<14} {output:<10} {license}")


if __name__ == "__main__":
    main()
