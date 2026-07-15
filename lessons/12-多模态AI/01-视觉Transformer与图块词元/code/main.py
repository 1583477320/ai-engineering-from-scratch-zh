# ViT 从零实现：图块切分 + 位置编码 + Transformer

import torch
import torch.nn as nn
import math


class SimpleViT(nn.Module):
    """简化版 Vision Transformer。"""
    def __init__(self, image_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, num_layers=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 图块投影
        self.patch_proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)
        # CLS 词元 + 位置嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)
        B, C, Hp, Wp, P1, P2 = patches.shape
        patches = patches.reshape(B, C, Hp * Wp, P1 * P2).permute(0, 2, 1, 3)
        patches = patches.reshape(B, Hp * Wp, -1)
        patch_emb = self.patch_proj(patches)
        cls = self.cls_token.expand(B, -1, -1)
        patch_emb = torch.cat([cls, patch_emb], dim=1)
        patch_emb = patch_emb + self.pos_embed
        out = self.transformer(patch_emb)
        cls_out = out[:, 0]
        return self.head(cls_out)


def vit_stats(image_size, patch_size, embed_dim, num_layers):
    """计算 ViT 参数量和 FLOPs。"""
    num_patches = (image_size // patch_size) ** 2
    patch_dim = 3 * patch_size * patch_size
    per_layer = 4 * embed_dim * embed_dim + 2 * embed_dim * embed_dim
    total_params = patch_dim * embed_dim + num_layers * per_layer + embed_dim
    return {
        "num_patches": num_patches,
        "total_params_M": total_params / 1e6,
    }


if __name__ == "__main__":
    print("ViT 从零实现\n")
    # 参数量统计
    for name, p, d, l in [("ViT-Base", 16, 768, 12), ("ViT-Small", 16, 384, 6), ("ViT-Tiny", 8, 192, 4)]:
        s = vit_stats(224, p, d, l)
        print(f"  {name}: {s['num_patches']} patches, {s['total_params_M']:.1f}M params")

    # 前向传播
    model = SimpleViT(image_size=224, patch_size=16, embed_dim=128, num_layers=2, num_heads=4)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"\n前向传播: {dummy.shape} -> {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
