# 具身 VLA 模型：视觉语言动作

import torch
import torch.nn as nn


class SimpleVLA(nn.Module):
    """简化版 VLA 模型。"""
    def __init__(self, image_dim=512, text_dim=512, hidden_dim=256, action_dim=7):
        super().__init__()
        self.image_proj = nn.Linear(3 * 64 * 64, image_dim)
        self.text_proj = nn.Linear(64, text_dim)
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, image, text_embed):
        img_feat = self.image_proj(image.view(image.size(0), -1))
        text_feat = self.text_proj(text_embed)
        combined = torch.cat([img_feat, text_feat], dim=-1)
        return self.fusion(combined)


if __name__ == "__main__":
    print("VLA 模型演示\n")
    model = SimpleVLA(image_dim=256, text_dim=256, hidden_dim=128, action_dim=7)
    image = torch.randn(2, 3, 64, 64)
    text_embed = torch.randn(2, 64)
    action = model(image, text_embed)
    print(f"图像: {image.shape}, 文本嵌入: {text_embed.shape} -> 动作: {action.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params/1e6:.2f}M")
