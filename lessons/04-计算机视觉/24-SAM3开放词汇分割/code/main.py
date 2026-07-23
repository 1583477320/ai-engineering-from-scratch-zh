# main.py — SAM3 开放词汇分割：从零实现核心接口
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：阶段 04 · 第 24 课（SAM3 开放词汇分割）

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import json
import numpy as np


# === 第 1 步：定义检测结果的数据结构 ===

@dataclass
class ConceptDetection:
    """单个概念检测结果。

    Attributes:
        concept:    文本提示的概念（如 "黄色校车"）
        instance_id: 实例编号，同一概念的不同实例使用不同 ID
        box:        边界框 (x1, y1, x2, y2)，像素坐标
        score:      置信度分数 [0, 1]
        mask_rle:   运行长度编码的掩码（节省存储空间）
    """
    concept: str
    instance_id: int
    box: tuple
    score: float
    mask_rle: str


# === 第 2 步：多概念分词器 ===

def split_concepts(sentence: str) -> List[str]:
    """将用户输入的自然语言拆分为独立的概念提示词。

    SAM3 每次前向传播只处理一个概念。对于多概念查询，
    需要先拆分再逐个调用。

    支持的分隔符：逗号、分号、"and"、"or"、"&"

    Args:
        sentence: 用户输入的自然语言字符串

    Returns:
        拆分后的概念列表

    Examples:
        >>> split_concepts("猫, 狗和气球")
        ['猫', '狗', '气球']
        >>> split_concepts("黄色校车")
        ['黄色校车']
    """
    normalized = sentence
    # 将常见分隔符（英文和中文）统一替换为逗号
    for sep in [" and ", " or ", "和", "或", "&", ";"]:
        normalized = normalized.replace(sep, ",")
    if "," in normalized:
        parts = [p.strip() for p in normalized.split(",")]
        return [p for p in parts if p]
    return [sentence.strip()]


# === 第 3 步：运行长度编码（RLE）——掩码的高效存储格式 ===

def rle_encode(binary_mask: np.ndarray) -> str:
    """将二值掩码编码为运行长度格式（Run-Length Encoding）。

    RLE 将连续的 0 和 1 压缩为 (值, 连续长度) 对。
    例如：[0, 0, 1, 1, 1, 0] → "0x2;1x3;0x1"

    这种编码在高分辨率掩码上可以实现 10-50 倍的压缩，
    是 COCO 数据集和 SAM 系列模型的标准存储格式。

    Args:
        binary_mask: 形状为 (H, W) 的二值掩码，值为 0 或 1

    Returns:
        RLE 编码字符串
    """
    flat = binary_mask.flatten().astype("uint8")
    if flat.size == 0:
        return ""
    runs = []
    prev = int(flat[0])
    count = 0
    for v in flat:
        iv = int(v)
        if iv == prev:
            count += 1
        else:
            runs.append((prev, count))
            prev, count = iv, 1
    runs.append((prev, count))
    return ";".join(f"{v}x{c}" for v, c in runs)


def rle_decode(rle_str: str, shape: tuple) -> np.ndarray:
    """将 RLE 字符串解码回二值掩码。

    Args:
        rle_str: RLE 编码字符串
        shape:   目标掩码形状 (H, W)

    Returns:
        形状为 shape 的二值掩码
    """
    if not rle_str:
        return np.zeros(shape, dtype=np.uint8)
    flat = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    idx = 0
    for part in rle_str.split(";"):
        v, c = part.split("x")
        v = int(v)
        c = int(c)
        flat[idx:idx + c] = v
        idx += c
    return flat.reshape(shape)


# === 第 4 步：开放词汇分割的抽象接口 ===

class OpenVocabSeg(ABC):
    """开放词汇分割的统一接口。

    所有后端（SAM3、Grounded SAM 2、YOLO-World + SAM 2）
    都实现同一个 `detect` 方法。下游代码无需因更换后端而修改。

    这种设计模式在生产环境中非常常见：
    - A/B 测试不同模型
    - 根据许可证约束切换后端
    - 渐进式迁移
    """

    @abstractmethod
    def detect(self, image: np.ndarray, concept: str) -> List[ConceptDetection]:
        """检测并分割图像中所有匹配概念的实例。

        Args:
            image:   输入图像，形状 (H, W, 3)
            concept: 文本概念提示词（如 "红色汽车"）

        Returns:
            检测结果列表，每个元素包含一个实例的掩码、边界框和置信度
        """
        ...


class StubOpenVocabSeg(OpenVocabSeg):
    """用于流水线测试的确定性存根。

    当真实模型未加载时，用固定输出替代，确保下游代码可以运行。
    这是生产环境中的常见做法——先用 stub 跑通流水线，再接真实模型。
    """

    def detect(self, image: np.ndarray, concept: str) -> List[ConceptDetection]:
        h, w = image.shape[:2]
        # 构造两个模拟实例的掩码
        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_a[int(h * 0.3):int(h * 0.8), int(w * 0.2):int(w * 0.5)] = 1
        mask_b = np.zeros((h, w), dtype=np.uint8)
        mask_b[int(h * 0.25):int(h * 0.75), int(w * 0.55):int(w * 0.85)] = 1
        return [
            ConceptDetection(
                concept=concept,
                instance_id=0,
                box=(w * 0.2, h * 0.3, w * 0.5, h * 0.8),
                score=0.89,
                mask_rle=rle_encode(mask_a),
            ),
            ConceptDetection(
                concept=concept,
                instance_id=1,
                box=(w * 0.55, h * 0.25, w * 0.85, h * 0.75),
                score=0.74,
                mask_rle=rle_encode(mask_b),
            ),
        ]


# === 第 5 步：多概念查询流水线 ===

def run_multi_concept(
    model: OpenVocabSeg,
    image: np.ndarray,
    user_utterance: str,
) -> List[ConceptDetection]:
    """处理用户的多概念自然语言查询。

    流程：拆分概念 → 逐个调用模型 → 合并结果。

    在生产环境中，这里可能还需要：
    - 并行调用以降低延迟
    - 后处理（NMS 去重、置信度过滤）
    - 结果缓存
    """
    concepts = split_concepts(user_utterance)
    all_detections = []
    for concept in concepts:
        detections = model.detect(image, concept)
        all_detections.extend(detections)
    return all_detections


# === 第 6 步：运行演示 ===

def main():
    # 演示 1：多概念拆分
    print("=" * 60)
    print("[演示 1] 多概念拆分")
    print("=" * 60)
    test_sentences = [
        "猫, 狗和气球",
        "黄色校车",
        "条纹红伞; 绿帽子",
        "一条大船 or 小船",
    ]
    for s in test_sentences:
        result = split_concepts(s)
        print(f"  输入: {s!r:45s} -> {result}")

    # 演示 2：RLE 编码/解码往返
    print(f"\n{'=' * 60}")
    print("[演示 2] RLE 编码/解码往返验证")
    print("=" * 60)
    rng = np.random.default_rng(0)
    mask = (rng.random((16, 16)) > 0.5).astype(np.uint8)
    rle = rle_encode(mask)
    restored = rle_decode(rle, mask.shape)
    diff = int(np.abs(mask.astype(int) - restored.astype(int)).sum())
    print(f"  掩码形状: {mask.shape}")
    print(f"  RLE 长度: {len(rle)} 字符")
    print(f"  往返差异: {diff}（应为 0）")
    print(f"  压缩率: {len(rle) / (mask.size * 2):.1%}")

    # 演示 3：使用存根模型进行多概念检测
    print(f"\n{'=' * 60}")
    print("[演示 3] 多概念检测（存根模型）")
    print("=" * 60)
    # 模拟一张 240x320 的图像
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    stub = StubOpenVocabSeg()
    detections = run_multi_concept(stub, image, "橙子, 苹果")
    print(f"  共检测到 {len(detections)} 个实例")
    for d in detections:
        summary = {
            "概念": d.concept,
            "实例": d.instance_id,
            "边界框": tuple(round(x, 1) for x in d.box),
            "置信度": round(d.score, 2),
            "掩码段数": len(d.mask_rle.split(";")),
        }
        print(f"    {json.dumps(summary, ensure_ascii=False)}")

    # 演示 4：RLE 掩码的可视化（ASCII）
    print(f"\n{'=' * 60}")
    print("[演示 4] 掩码可视化（ASCII 热力图）")
    print("=" * 60)
    small_mask = mask[:8, :8]  # 取 8x8 子区域
    print("  二值掩码（1=前景, 0=背景）:")
    for row in small_mask:
        print("    " + "".join("##" if v else ".." for v in row))
    print(f"  对应 RLE: {rle_encode(small_mask)}")


if __name__ == "__main__":
    main()
