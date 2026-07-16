# 视频时序采样器 + 时序定位


def dynamic_frame_sampler(total_budget=288, num_frames_available=30):
    """动态帧采样——在固定预算内平衡帧数和分辨率。"""
    best = {"num_frames": 1, "patches_per_frame": total_budget}
    for n_frames in range(num_frames_available, 0, -1):
        patches_per_frame = total_budget // n_frames
        if patches_per_frame >= 1:
            best = {"num_frames": n_frames, "patches_per_frame": patches_per_frame}
    return best


def temporal_grounding(query_text, frame_timestamps, confidence_threshold=0.5):
    """简化时序定位——返回置信度高于阈值的时间区间。"""
    results = []
    for start, end, score in frame_timestamps:
        if score >= confidence_threshold:
            results.append({"query": query_text, "start": start, "end": end, "score": score})
    return results


if __name__ == "__main__":
    print("视频时序采样器演示\n")
    for budget in [128, 256, 512]:
        config = dynamic_frame_sampler(total_budget=budget, num_frames_available=30)
        print(f"  预算 {budget}: {config['num_frames']} 帧 × {config['patches_per_frame']} 词元/帧")

    print("\n时序定位:")
    frames = [(0.0, 2.0, 0.9), (2.0, 5.0, 0.3), (5.0, 8.0, 0.8)]
    results = temporal_grounding("猫跳起来", frames, threshold=0.5)
    for r in results:
        print(f"  {r['start']:.1f}s - {r['end']:.1f}s: {r['score']:.2f}")
