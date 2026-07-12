# VAD + 端点检测演示
# 对应课程：阶段 06 · 14
"""VAD 三级流水线：能量门控 → Silero VAD → 语义端点检测"""
import math
import random


def energy_gate(signal, threshold_db=-40):
    """第一层：能量门控——RMS 能量阈值。"""
    rms = math.sqrt(sum(s*s for s in signal) / max(1, len(signal)))
    db = 20 * math.log10(max(rms, 1e-10))
    return db > threshold_db


def simulate_vad(signal, threshold=0.5, chunk_size=160):
    """模拟 Silero VAD——每块输出语音概率。"""
    results = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        energy = sum(s*s for s in chunk) / max(1, len(chunk))
        prob = min(1.0, energy * 10)
        results.append(prob > threshold)
    return results


def turn_detection(vad_outputs, hangover_chunks=25):
    """端点检测：VAD 返回非语音后等待 hangover 再宣布结束。"""
    turns, current, silent = [], False, 0
    for speech in vad_outputs:
        if speech and not current:
            current, silent = True, 0
            turns.append("开始说话")
        elif not speech and current:
            silent += 1
            if silent >= hangover_chunks:
                turns.append("结束说话（hangover）")
                current = False
        else:
            silent = 0
    return turns


def main():
    random.seed(42)
    sr = 16000
    signal = []
    for _ in range(int(sr * 1.5)):
        t = random.uniform(0, 1)
        if t > 0.2 and t < 0.5:
            signal.append(random.gauss(0, 0.1))
        elif t > 0.7 and t < 0.9:
            signal.append(random.gauss(0, 0.1))
        else:
            signal.append(random.gauss(0, 0.001))

    print("=== VAD 三级流水线演示 ===")
    energy_results = energy_gate(signal)
    silero_results = simulate_vad(signal)
    turns = turn_detection(silero_results)

    print(f"  能量门控: {sum(energy_results)} 帧有语音 / {len(energy_results)} 帧")
    print(f"  Silero VAD: {sum(silero_results)} 帧有语音 / {len(silero_results)} 帧")
    print(f"  端点检测: {turns}")

    print("\n=== VAD 对比 ===")
    print("  | VAD          | TPR@5%FPR | 延迟    | 许可   |")
    for name, tpr, lat, lic in [("WebRTC VAD", "50.0%", "30ms", "BSD"),
                                  ("Silero VAD", "87.7%", "~1ms", "MIT"),
                                  ("Pyannote VAD", "92.0%", "15ms", "MIT")]:
        print(f"  | {name:<12} | {tpr:<9} | {lat:<7} | {lic:<5} |")

    print("\n=== Flush Trick ===")
    print("  普通流程: VAD 500ms + STT 500ms = 1000ms")
    print("  Flush 信号: VAD 触发后立即通知 STT 输出 → 125ms")


if __name__ == "__main__":
    main()
