# 任意分辨率图块打包

import torch


def patchify_and_pack(images, patch_size=16):
    """将可变分辨率图像打包为一个序列。"""
    all_patches = []
    boundaries = []
    offset = 0
    for img in images:
        C, H, W = img.shape
        H_pad = (H + patch_size - 1) // patch_size * patch_size
        W_pad = (W + patch_size - 1) // patch_size * patch_size
        padded = torch.nn.functional.pad(img, (0, W_pad - W, 0, H_pad - H))
        patches = padded.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        _, _, Hp, Wp, _, _ = patches.shape
        patches = patches.reshape(1, Hp * Wp, C * patch_size * patch_size)
        num_patches = Hp * Wp
        boundaries.append((offset, offset + num_patches))
        all_patches.append(patches)
        offset += num_patches
    packed = torch.cat(all_patches, dim=1)
    return packed, boundaries


def create_block_diagonal_mask(boundaries, seq_len):
    """块对角注意力掩码。"""
    mask = torch.ones(seq_len, seq_len) * float("-inf")
    for start, end in boundaries:
        mask[start:end, start:end] = 0.0
    return mask


if __name__ == "__main__":
    print("任意分辨率图块打包演示\n")
    images = [
        torch.randn(3, 224, 224),   # 196 patches
        torch.randn(3, 448, 224),   # 392 patches
        torch.randn(3, 224, 448),   # 392 patches
    ]
    packed, bounds = patchify_and_pack(images, patch_size=16)
    print(f"打包后: {packed.shape} (1, 980, 768)")
    print(f"图像边界: {bounds}")
    mask = create_block_diagonal_mask(bounds, packed.shape[1])
    print(f"掩码形状: {mask.shape}")
