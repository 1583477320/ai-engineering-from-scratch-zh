# 长视频词元预算计算 + TokenPacker

import numpy as np


def compute_video_token_budget(duration_sec, fps=1, patches_per_frame=100):
    """计算视频的总词元数。"""
    num_frames = int(duration_sec * fps)
    total_tokens = num_frames * patches_per_frame
    return {
        "duration_sec": duration_sec,
        "fps": fps,
        "num_frames": num_frames,
        "total_tokens": total_tokens,
    }


def token_packer_compress(video_tokens, target_tokens=1024):
    """TokenPacker 压缩。"""
    num_frames, tokens_per_frame = video_tokens.shape
    total = num_frames * tokens_per_frame
    if total <= target_tokens:
        return video_tokens
    sampled_frames = max(1, int(num_frames * target_tokens / total))
    indices = np.linspace(0, num_frames - 1, sampled_frames, dtype=int)
    return video_tokens[indices]


if __name__ == "__main__":
    print("长视频词元预算计算\n")
    for dur in [10, 60, 3600]:
        for fps in [1, 2, 4]:
            r = compute_video_token_budget(dur, fps)
            print(f"  {dur}s @ {fps}fps: {r['total_tokens']} 词元 ({r['num_frames']} 帧)")

    print("\nTokenPacker 压缩:")
    tokens = np.random.randn(100, 100)  # 100 帧, 每帧 100 词元
    for target in [512, 1024, 2048]:
        compressed = token_packer_compress(tokens, target)
        print(f"  目标 {target}: {tokens.shape} -> {compressed.shape}")
