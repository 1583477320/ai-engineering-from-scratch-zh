# Flamingo 门控交叉注意力实现

import torch
import torch.nn as nn


class GatedCrossAttention(nn.Module):
    """Flamingo 的门控交叉注意力层。"""
    def __init__(self, llm_dim=768, image_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        self.gate = nn.Parameter(torch.zeros(1))  # 初始化为 0

    def forward(self, text_hidden, image_features):
        gate_value = torch.tanh(self.gate)
        normed = self.norm1(text_hidden)
        attn_out = self.cross_attn(normed, image_features, image_features)[0]
        return text_hidden + gate_value * attn_out


class PerceiverResampler(nn.Module):
    """Perceiver 重采样器：N 图块 -> K 潜在查询。"""
    def __init__(self, input_dim=1024, output_dim=768, num_queries=32, num_heads=8):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, output_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        B = image_features.shape[0]
        projected = self.projection(image_features)
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        return self.cross_attn(queries, projected, projected)[0]


if __name__ == "__main__":
    print("Flamingo 门控交叉注意力演示\n")

    # 测试门控
    gate = GatedCrossAttention(llm_dim=256, image_dim=512, num_heads=4)
    print(f"初始 gate 值: {gate.gate.item():.4f} (应为 0)")
    print(f"tanh(gate): {torch.tanh(gate.gate).item():.4f} (应为 0)")

    text_hidden = torch.randn(2, 10, 256)
    image_features = torch.randn(2, 196, 512)
    output = gate(text_hidden, image_features)
    print(f"输出: {output.shape}")
    print(f"输出应与输入相同（gate=0 时）: {torch.allclose(output, text_hidden)}")

    # Perceiver
    perceiver = PerceiverResampler(input_dim=512, output_dim=256, num_queries=16)
    image_features = torch.randn(2, 196, 512)
    output = perceiver(image_features)
    print(f"\nPerceiver: 196 patches -> {output.shape[1]} queries")
