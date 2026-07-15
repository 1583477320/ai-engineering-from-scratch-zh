# Emu3 图像 Tokenizer + 自回归生成

import torch
import torch.nn as nn


class SimpleImageTokenizer(nn.Module):
    def __init__(self, n_codes=512, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, padding=1),
        )
        self.codebook = nn.Embedding(n_codes, embed_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh(),
        )

    def encode(self, x):
        z = self.encoder(x)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        dists = torch.cdist(z_flat, self.codebook.weight.unsqueeze(0))
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)
        return z_q.reshape(B, H, W, -1), indices.reshape(B, H, W)

    def decode(self, z_q):
        return self.decoder(z_q.permute(0, 3, 1, 2))


if __name__ == "__main__":
    print("Emu3 图像 Tokenizer 演示\n")
    tokenizer = SimpleImageTokenizer(n_codes=512, embed_dim=256)
    img = torch.randn(1, 3, 32, 32)
    z_q, indices = tokenizer.encode(img)
    recon = tokenizer.decode(z_q)
    print(f"输入: {img.shape}")
    print(f"词元: {indices.shape} (每张图 {indices[0].numel()} 个词元)")
    print(f"重建: {recon.shape}")
    print(f"参数量: {sum(p.numel() for p in tokenizer.parameters())/1e6:.1f}M")
