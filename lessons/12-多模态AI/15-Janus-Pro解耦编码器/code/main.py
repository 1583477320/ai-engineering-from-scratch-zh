# Janus-Pro 双编码器路由

import torch
import torch.nn as nn


class DualEncoderRouter(nn.Module):
    """将不同任务路由到不同编码器。"""
    def __init__(self, siglip_dim=1024, vq_dim=256, hidden=768):
        super().__init__()
        self.siglip_proj = nn.Linear(siglip_dim, hidden)
        self.vq_proj = nn.Linear(vq_dim, hidden)

    def route(self, task_type, features):
        if task_type == "understanding":
            return self.siglip_proj(features)
        else:
            return self.vq_proj(features)


class SharedTransformer(nn.Module):
    """共享的 Transformer 主体。"""
    def __init__(self, hidden=768, n_layers=6, n_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(hidden, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.output_head = nn.Linear(hidden, 32000)

    def forward(self, x):
        return self.output_head(self.transformer(x))


if __name__ == "__main__":
    print("Janus-Pro 双编码器路由演示\n")
    router = DualEncoderRouter()
    siglip_features = torch.randn(2, 196, 1024)
    vq_features = torch.randn(2, 64, 256)

    understanding_out = router.route("understanding", siglip_features)
    generation_out = router.route("generation", vq_features)

    print(f"理解特征: {siglip_features.shape} -> {understanding_out.shape}")
    print(f"生成特征: {vq_features.shape} -> {generation_out.shape}")
    print(f"两者维度对齐: {understanding_out.shape[-1] == generation_out.shape[-1]}")
