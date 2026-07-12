# Moshi 架构概念演示
# 对应课程：阶段 06 · 15
"""Moshi 全双工架构：双 Mimi 流 + 内心独白文本流"""

def main():
    print("=== Moshi 架构演示 ===\n")
    print("架构核心：")
    print("  输入: 两个 Mimi 编解码器流（12.5Hz × 8 个 codebook）")
    print("  流 1: 用户音频（持续到达）")
    print("  流 2: Moshi 自己的音频（生成中）")
    print("  文本流: Moshi 的"内心独白"（转录）\n")

    print("每 80ms 步骤：")
    print("  1. 消费用户最新的 Mimi token")
    print("  2. 消费 Moshi 最新的 Mimi token（已生成）")
    print("  3. 生成下一个 Moshi 文本 token（内心独白）")
    print("  4. 生成下一个 Moshi Mimi token（8 个 codebook）")

    print(f"\n=== 延迟对比 ===")
    print("  Moshi 全双工:        ~200ms（80ms Mimi帧 + 80ms 声学延迟）")
    print("  GPT-4o-realtime:      ~320ms（级联流水线）")
    print("  Pipecat (Whisper+LLM+TTS): ~500ms（多组件流水线）")

    print(f"\n=== Mimi 编解码器关键参数 ===")
    print("  帧率: 12.5 Hz → 1秒 = 12.5 帧 token")
    print("  Codebook 0: 语义（从 WavLM 蒸馏）")
    print("  Codebook 1-7: 声学残差")
    print("  比特率: 4.4 kbps")

    print(f"\n=== Hibiki 流式语音翻译 ===")
    print("  架构: 与 Moshi 相同，但训练在翻译对上")
    print("  特点: 源音频→目标语言音频，逐 chunk 连续输出")
    print("  语言对: 4 种语言对，可用 ~1000 小时数据适配新语言")


if __name__ == "__main__":
    main()
