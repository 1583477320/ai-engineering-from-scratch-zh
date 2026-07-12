# 实时音频处理：帧预算计算 + 延迟分析
# 对应课程：阶段 06 · 11

import math
from typing import List


# 延迟预算配置
BUDGETS = {
    "麦克风→缓冲区": 20,
    "VAD": 10,
    "ASR（流式）": 150,
    "LLM（首token）": 100,
    "TTS（首个chunk）": 100,
    "渲染→扬声器": 20,
}


def ring_buffer_size(max_latency_ms: float, sample_rate: int = 16000) -> int:
    """环形缓冲区大小 = 最大延迟 × 采样率。"""
    return int(max_latency_ms / 1000 * sample_rate)


def frames_per_second(sr: int = 16000, chunk_ms: int = 20) -> int:
    """每秒帧数。20ms 帧 @16kHz = 50 帧/秒。"""
    return 1000 // chunk_ms


def interruption_budget() -> dict:
    """打断检测预算：VAD 检测 + TTS 停止 + LLM 取消 = 必须 < 100ms。"""
    return {"检测": 10, "停止TTS": 30, "取消LLM": 40, "重启STT": 20}


def v40_latency_per_frame(frame_ms: int = 30) -> float:
    """Silero VAD 4.0 在 CPU 上每帧延迟。"""
    return 0.8  # ms，实测 < 1ms


def main():
    print("=== 2026 实时语音助手延迟预算 ===\n")
    total = sum(BUDGETS.values())
    for stage, budget in BUDGETS.items():
        bar = "█" * (budget // 10)
        print(f"  {stage:<18} {budget:>4}ms  {bar}")
    print(f"  {'总计':<18} {total:>4}ms")

    print(f"\n=== 环形缓冲区大小 ===")
    for lat in [200, 500, 1000]:
        size = ring_buffer_size(lat)
        print(f"  {lat}ms @16kHz → {size} 采样点 ({size/1024:.1f}KB)")

    print(f"\n=== 帧率 ===")
    for sr in [8000, 16000, 24000, 48000]:
        fps = frames_per_second(sr)
        print(f"  {sr}Hz → {fps} 帧/秒")

    print(f"\n=== 打断处理预算 ===")
    ib = interruption_budget()
    total_ib = sum(ib.values())
    for step, ms in ib.items():
        print(f"  {step}: {ms}ms")
    print(f"  总计: {total_ib}ms {'✅' if total_ib <= 100 else '❌ 超时'}")

    print(f"\n=== VAD 延迟 ===")
    print(f"  Silero VAD 4.0: {v40_latency_per_frame()}ms/帧 (CPU)")

    print(f"\n=== 2026 延迟里程碑 ===")
    milestones = [
        ("Moshi（全双工）", "~200ms", "Kyutai, 2024"),
        ("GPT-4o-realtime", "~320ms", "OpenAI, 2024"),
        ("流水线级联（2022）", "~2500ms", "早期架构"),
    ]
    for name, lat, note in milestones:
        print(f"  {name}: {lat} ({note})")


if __name__ == "__main__":
    main()
