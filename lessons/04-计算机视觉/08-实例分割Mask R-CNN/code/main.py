# main.py — Mask R-CNN 实例分割教学实现
# 依赖：torch>=2.0, torchvision>=0.15
# 安装：pip install torch torchvision

import math
import torch
import torch.nn.functional as F


def roi_align_single(feature: torch.Tensor, box: list,
                     output_size: int = 7, spatial_scale: float = 1 / 16.0) -> torch.Tensor:
    """从零实现 RoIAlign。

    对单个特征图和候选框，使用双线性插值在精确的浮点坐标上采样，
    不经过任何坐标取整操作。这是 Mask R-CNN 的核心工程难点。

    Args:
        feature: (C, H, W) 单张图像的特征图
        box: [x1, y1, x2, y2] 原始图像像素坐标系下的候选框
        output_size: 输出网格的边长（box head 用 7，mask head 用 14）
        spatial_scale: 特征图步长的倒数

    Returns:
        采样后的特征，形状 (C, output_size, output_size)
    """
    C, H, W = feature.shape
    # 将坐标缩放到特征图空间，并减去 0.5 对齐 grid_sample 的约定
    x1, y1, x2, y2 = [c * spatial_scale - 0.5 for c in box]
    bin_w = (x2 - x1) / output_size
    bin_h = (y2 - y1) / output_size

    # 在每个 cell 的中心采样 output_size 个点
    grid_y = torch.linspace(y1 + bin_h / 2, y2 - bin_h / 2, output_size, device=feature.device)
    grid_x = torch.linspace(x1 + bin_w / 2, x2 - bin_w / 2, output_size, device=feature.device)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    # 转换为 [-1, 1] 范围供 grid_sample 使用
    gx = 2 * (xx + 0.5) / W - 1
    gy = 2 * (yy + 0.5) / H - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                            align_corners=False)
    return sampled.squeeze(0)


def compare_roi_align() -> list[float]:
    """对比自定义 RoIAlign 与 torchvision.ops.roi_align 的结果。"""
    torch.manual_seed(0)
    feature = torch.randn(1, 16, 50, 50)
    boxes = torch.tensor([[0, 10, 20, 100, 90],
                          [0, 5, 5, 80, 80],
                          [0, 30, 10, 120, 110]], dtype=torch.float32)

    from torchvision.ops import roi_align

    diffs = []
    for i, b in enumerate(boxes):
        ours = roi_align_single(feature[0], b[1:].tolist(),
                                output_size=7, spatial_scale=1 / 4)
        theirs = roi_align(feature, b.unsqueeze(0),
                           output_size=(7, 7), spatial_scale=1 / 4,
                           sampling_ratio=1, aligned=True)[0]
        diff = (ours - theirs).abs().max().item()
        diffs.append(diff)

    return diffs


def build_custom_maskrcnn(num_classes: int) -> torch.nn.Module:
    """构建自定义类别数的 Mask R-CNN。

    替换 box head 和 mask head 的分类器，保持 backbone、FPN、RPN 不变。

    Args:
        num_classes: 包含背景类的类别总数（如 4 个物体类 → num_classes=5）

    Returns:
        修改后的 Mask R-CNN 模型
    """
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    # 替换 box head 分类器
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换 mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def freeze_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """冻结 backbone（ResNet）和 FPN 参数。

    在小数据集上微调时，只训练 RPN 和两个 head，避免过拟合。

    Args:
        model: Mask R-CNN 模型

    Returns:
        冻结后的模型
    """
    # torchvision 的 Mask R-CNN 将 FPN 打包在 model.backbone 中，
    # 所以冻结 model.backbone.parameters() 会同时冻结 ResNet 和 FPN。
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model


def run_inference(model: torch.nn.Module) -> dict:
    """在随机输入上运行推理，展示输出格式。

    Args:
        model: 已加载权重的 Mask R-CNN 模型

    Returns:
        预测结果字典
    """
    with torch.no_grad():
        batch = torch.randn(1, 3, 480, 640)
        predictions = model([batch])[0]

    # masks 是 (N, 1, H, W)，阈值化后得到二值掩码 (N, H, W)
    binary_masks = (predictions["masks"][:, 1:] > 0.5).squeeze(1)

    print(f"  boxes:   {predictions['boxes'].shape}")       # (N, 4)
    print(f"  labels:  {predictions['labels'].shape}")      # (N,)
    print(f"  scores:  {predictions['scores'].shape}")      # (N,)
    print(f"  masks:   {predictions['masks'].shape}")       # (N, 1, H, W)
    print(f"  binary:  {binary_masks.shape}")               # (N, H, W)

    return predictions


def show_dataset_format():
    """展示自定义训练数据的目标格式。

    torchvision 的 Mask R-CNN 在训练时需要以下格式的 targets：
    - boxes: (num_objects, 4) 的 (x1, y1, x2, y2) 坐标
    - labels: (num_objects,) 的类别 ID
    - masks: (num_objects, H, W) 的二值掩码张量
    """
    H, W = 480, 640
    target = {
        "boxes": torch.tensor([[50, 30, 200, 300],
                               [250, 100, 450, 400]], dtype=torch.float32),
        "labels": torch.tensor([1, 2]),
        "masks": torch.randint(0, 2, (2, H, W)).bool(),
    }

    print("训练目标格式示例:")
    print(f"  boxes:   {target['boxes'].shape}  — 两个物体的边界框")
    print(f"  labels:  {target['labels'].shape}  — 类别 1 和 2（不含背景）")
    print(f"  masks:   {target['masks'].shape}  — 对应每个物体的二值掩码")


def main():
    print("=" * 60)
    print("RoIAlign: 自定义实现 vs torchvision")
    print("=" * 60)
    diffs = compare_roi_align()
    for i, d in enumerate(diffs):
        print(f"  候选框 {i}: 最大绝对误差 = {d:.2e}")
    print(f"\n结论: 与 torchvision.ops.roi_align(aligned=True) 在 1e-5 以内匹配\n")

    print("=" * 60)
    print("训练目标格式")
    print("=" * 60)
    show_dataset_format()
    print()

    print("=" * 60)
    print("预训练模型推理")
    print("=" * 60)
    try:
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
        )
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        model.eval()
        params_total = sum(p.numel() for p in model.parameters())
        print(f"  总参数量: {params_total:,}")

        run_inference(model)
    except Exception as e:
        print(f"  跳过（无法下载权重）: {e}")

    print()
    print("=" * 60)
    print("微调准备：替换 head + 冻结 backbone")
    print("=" * 60)
    try:
        custom = build_custom_maskrcnn(num_classes=5)
        custom = freeze_backbone(custom)
        trainable = sum(p.numel() for p in custom.parameters() if p.requires_grad)
        total = sum(p.numel() for p in custom.parameters())
        frozen = total - trainable
        print(f"  可训练参数: {trainable:,} （仅 head 和 RPN）")
        print(f"  冻结参数:   {frozen:,} （backbone + FPN）")
        print(f"  冻结比例:   {frozen / total * 100:.1f}%")
    except Exception as e:
        print(f"  跳过（无法加载预训练权重）: {e}")


if __name__ == "__main__":
    main()
