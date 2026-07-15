# BLIP-2 Q-Former 桥接实现

import torch
import torch.nn as nn


class QFormer(nn.Module):
    """简化版 Q-Former。"""
    def __init__(self, num_queries=32, embed_dim=768, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, image_features):
        B = image_features.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, image_features, image_features)[0]
        return self.norm3(queries)


class VisualBridge(nn.Module):
    def __init__(self, num_queries=32, vit_dim=1024, llm_dim=4096):
        super().__init__()
        self.qformer = QFormer(num_queries=num_queries, embed_dim=vit_dim)
        self.projection = nn.Linear(vit_dim, llm_dim)

    def forward(self, image_features):
        visual_tokens = self.qformer(image_features)
        projected = self.projection(visual_tokens)
        return projected


if __name__ == "__main__":
    print("BLIP-2 Q-Former 桥接演示\n")
    bridge = VisualBridge(num_queries=32, vit_dim=1024, llm_dim=4096)
    image_features = torch.randn(2, 196, 1024)  # 224x224, 16x16 patches
    output = bridge(image_features)
    print(f"输入: {image_features.shape} (196 patches)")
    print(f"输出: {output.shape} (32 visual tokens)")
    params = sum(p.numel() for p in bridge.parameters())
    print(f"参数量: {params/1e6:.1f}M")
