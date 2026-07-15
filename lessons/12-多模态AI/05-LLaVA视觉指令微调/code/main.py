# LLaVA MLP 投影器与两阶段训练

import torch
import torch.nn as nn


class LLaVAProjector(nn.Module):
    """LLaVA 的 MLP 投影器。"""
    def __init__(self, vit_dim=1024, llm_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, image_features):
        return self.net(image_features)


def build_llava_prompt(user_message, has_image=True):
    """构建 LLaVA 格式提示词。"""
    image_tag = "<image>\n" if has_image else ""
    return f"""{image_tag}<|user|>
{user_message}
<|assistant|>"""


def evaluate_vlm(model, test_cases):
    """简化 VLM 评估。"""
    correct = 0
    for case in test_cases:
        prompt = build_llava_prompt(case["question"])
        response = "mock response"
        if case["answer"] in response.lower():
            correct += 1
    return correct / max(len(test_cases), 1)


if __name__ == "__main__":
    print("LLaVA MLP 投影器演示\n")

    projector = LLaVAProjector(vit_dim=1024, llm_dim=4096)
    vit_features = torch.randn(2, 196, 1024)
    projected = projector(vit_features)
    print(f"ViT 特征: {vit_features.shape} (196 patches)")
    print(f"投影后: {projected.shape} (196 visual tokens)")
    print(f"LLM 嵌入维度: {projected.shape[-1]}")

    params = sum(p.numel() for p in projector.parameters())
    print(f"MLP 参数量: {params / 1e6:.1f}M")
    print(f"Q-Former 参数量: ~188M（对比 > MLP 的 {params/1e6:.1f}M）")

    prompt = build_llava_prompt("这张图片里有什么？")
    print(f"\nLLaVA 提示词:\n{prompt}")
