# Qwen-VL M-RoPE + 动态 FPS

import torch
import torch.nn as nn


class MRoPE(nn.Module):
    """多分辨率旋转位置编码。"""
    def __init__(self, dim, max_pos=8192):
        super().__init__()
        self.dim = dim
        self.temporal_freq = nn.Parameter(torch.randn(max_pos, dim) * 0.02)
        self.height_freq = nn.Parameter(torch.randn(max_pos, dim) * 0.02)
        self.width_freq = nn.Parameter(torch.randn(max_pos, dim) * 0.02)

    def forward(self, x, temporal_ids, height_ids, width_ids):
        t_emb = self.temporal_freq[temporal_ids]
        h_emb = self.height_freq[height_ids]
        w_emb = self.width_freq[width_ids]
        return x + t_emb + h_emb + w_emb


def dynamic_fps(video_duration, max_frames=64):
    """自适应 FPS。"""
    if video_duration <= 10:
        return 4, min(int(video_duration * 4), max_frames)
    elif video_duration <= 60:
        return 2, min(int(video_duration * 2), max_frames)
    else:
        return 1, min(int(video_duration), max_frames)


if __name__ == "__main__":
    print("Qwen-VL M-RoPE + 动态 FPS 演示\n")
    mrope = MRoPE(dim=256)
    x = torch.randn(1, 64, 256)
    temporal_ids = torch.arange(64).unsqueeze(0)
    height_ids = torch.arange(8).repeat(8, 1).flatten().unsqueeze(0)
    width_ids = torch.arange(8).repeat_interleave(8).unsqueeze(0)
    output = mrope(x, temporal_ids, height_ids, width_ids)
    print(f"M-RoPE: {x.shape} -> {output.shape}")

    for dur in [5, 30, 120]:
        fps, frames = dynamic_fps(dur)
        print(f"视频 {dur}s: {fps} fps, {frames} 帧")
