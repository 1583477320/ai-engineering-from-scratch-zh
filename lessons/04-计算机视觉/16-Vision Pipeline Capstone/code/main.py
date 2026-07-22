# main.py — 端到端视觉管线 Capstone
# 依赖：torch>=2.0, torchvision>=0.15, pydantic>=2.0, numpy, fastapi>=0.100, uvicorn, Pillow
# 安装：pip install torch torchvision pydantic numpy fastapi uvicorn Pillow
# 对应课程：阶段 04 · 16（Vision Pipeline Capstone）

import time
import json
import hashlib
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field


# ===========================================================================
# 第 1 步：数据契约 —— 用 Pydantic 定义每一步的输入输出类型
# ===========================================================================

class Detection(BaseModel):
    """单次检测的结构化结果。"""

    box: Tuple[float, float, float, float] = (
        Field(ge=0, description="左上角 x 坐标")
    )  # type: ignore[assignment]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    """对单个检测区域的分类结果。"""

    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class ModelVersion(BaseModel):
    """模型版本追踪信息。"""

    detector_name: str
    detector_weights_hash: str
    classifier_name: str
    classifier_weights_hash: str


class PipelineResult(BaseModel):
    """整条管线的最终输出。"""

    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
    model_version: Optional[ModelVersion] = None


# ===========================================================================
# 第 2 步：占位模型 —— 不依赖预训练权重的教学用模型
# ===========================================================================

class StubDetector(nn.Module):
    """模拟 Mask R-CNN 输出的检测器占位。

    在实际管线中，这会是 torchvision.models.detection.maskrcnn_resnet50_fpn_v2
    或 ultralytics.YOLO 等。这里用固定框输出演示完整流程。
    """

    def __init__(self):
        super().__init__()
        self.name = "stub_detector_v1"
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, images):
        results = []
        for img in images:
            H, W = img.shape[-2:]
            # 生成三个占位边界框
            boxes = torch.tensor(
                [
                    [W * 0.1, H * 0.1, W * 0.4, H * 0.6],
                    [W * 0.5, H * 0.3, W * 0.9, H * 0.9],
                    [W * 0.2, H * 0.6, W * 0.45, H * 0.85],
                ],
                device=img.device,
            )
            scores = torch.tensor([0.92, 0.85, 0.71], device=img.device)
            labels = torch.tensor([1, 2, 1], device=img.device)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results


class StubClassifier(nn.Module):
    """模拟图像分类器占位。

    在实际管线中，这会是 ConvNeXt-Tiny、ResNet-18 等。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.name = "stub_classifier_v1"
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, num_classes),
        )

    def forward(self, x):
        return self.head(x)


def _weights_hash(model: nn.Module) -> str:
    """计算模型权重的 SHA256 校验和，用于模型版本管理。"""
    h = hashlib.sha256()
    for param in model.parameters():
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


# ===========================================================================
# 第 3 步：核心管线类 —— 串联预处理 → 检测 → 裁剪 → 分类 → 验证
# ===========================================================================

class VisionPipeline:
    """端到端视觉处理管线。

    流程：原始图像 → 预处理 → 目标检测 → ROI 裁剪 → 图像分类 → 结构化输出
    每一步的输入输出都经过 Pydantic 类型契约约束。
    """

    def __init__(
        self,
        detector: nn.Module,
        classifier: nn.Module,
        class_names: List[str],
        device: str = "cpu",
        min_crop_size: int = 16,
        trace_id: Optional[str] = None,
    ):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop_size = min_crop_size
        self.trace_id = trace_id or f"req-{int(time.time() * 1000)}"
        self.stage_timings: dict = {}

    def preprocess(self, image) -> torch.Tensor:
        """将输入转换为模型可用的张量。

        支持 NumPy ndarray (HWC, uint8)、PIL Image、以及 Tensor (CHW, float)。

        Args:
            image: PIL.Image / np.ndarray 或 torch.Tensor

        Returns:
            CHW 格式的浮点张量，值域 [0, 1]
        """
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"期望 HxWx3 RGB 图像，收到形状 {image.shape}"
                )
            tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
        elif isinstance(image, torch.Tensor):
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"期望 (3, H, W) 张量，收到形状 {tuple(image.shape)}"
                )
            tensor = image.float()
        else:
            raise TypeError(f"image 必须是 ndarray 或 Tensor，收到 {type(image)}")
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor: torch.Tensor) -> dict:
        """执行目标检测，返回框、分数、类别 ID。"""
        t0 = time.perf_counter()
        result = self.detector([image_tensor])[0]
        elapsed = (time.perf_counter() - t0) * 1000
        self.stage_timings["detect"] = elapsed
        return result

    @torch.no_grad()
    def classify(self, crops: List[torch.Tensor]) -> list:
        """批量分类裁剪后的检测区域。

        当 crops 为空列表时返回空结果，而不是崩溃。
        """
        if len(crops) == 0:
            return []
        t0 = time.perf_counter()
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        elapsed = (time.perf_counter() - t0) * 1000
        self.stage_timings["classify"] = elapsed
        return list(zip(cls.tolist(), scores.tolist()))

    def run(
        self, image, image_id: str = "anonymous"
    ) -> PipelineResult:
        """执行完整的检测+分类管线。

        每个失败路径都有具体的处理策略，而不是一个大的 try/except 吞掉一切。
        """
        t_start = time.perf_counter()
        self.stage_timings = {}

        # === 阶段 1：预处理 ===
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        self.stage_timings["preprocess"] = (time.perf_counter() - t0) * 1000

        # === 阶段 2：目标检测 ===
        det = self.detect(tensor)

        crops = []
        valid_indices = []
        detections = []

        # === 阶段 3：构建检测对象 + 提取 ROI ===
        for i, (box, score, label) in enumerate(
            zip(det["boxes"], det["scores"], det["labels"])
        ):
            x1, y1, x2, y2 = [max(0, int(b.item())) for b in box]
            # 防止越界裁剪
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])

            detections.append(Detection(
                box=(float(x1), float(y1), float(x2), float(y2)),
                score=float(score),
                class_id=int(label),
            ))

            # 太小的裁剪跳过分类 —— 避免小补丁导致分类器报错
            if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
                continue

            crop = tensor[:, y1:y2, x1:x2]
            # 缩放到分类器要求的输入尺寸
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        # === 阶段 4：对有效裁剪执行分类 ===
        class_preds = self.classify(crops)
        self.stage_timings["aggregate"] = (time.perf_counter() - t0) * 1000

        # === 阶段 5：构建分类结果并验证 ===
        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            name = (
                self.class_names[cls_id]
                if cls_id < len(self.class_names)
                else f"class_{cls_id}"
            )
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=name,
                score=float(cls_score),
            ))

        total_ms = (time.perf_counter() - t_start) * 1000
        self.stage_timings["total"] = total_ms

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=total_ms,
        )


# ===========================================================================
# 第 4 步：基准测试工具 —— 量化每条管线阶段的延迟分布
# ===========================================================================

def benchmark(
    pipe: VisionPipeline,
    num_runs: int = 10,
    image_size: Tuple[int, int] = (400, 600),
):
    """对管线进行分阶段基准测试。

    分别测量 preprocess / detect / classify 三阶段的 p50 和 p95 延迟，
    帮助定位性能瓶颈。
    """
    # 创建随机测试图像
    test_img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)

    # 预热一次
    pipe.run(test_img, image_id="warmup")

    if pipe.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}

    for _ in range(num_runs):
        pipe.run(test_img, image_id=f"bench-{_}")

        # 从记录的 stage_timings 中收集时间（检测在内部计时）
        stages["preprocess"].append(pipe.stage_timings.get("preprocess", 0))
        stages["detect"].append(pipe.stage_timings.get("detect", 0))
        stages["classify"].append(pipe.stage_timings.get("classify", 0))
        stages["total"].append(pipe.stage_timings.get("total", 0))

    # 打印百分位统计
    print("\n分阶段延迟基准测试")
    print("-" * 55)
    for stage_name, times in stages.items():
        times.sort()
        n = len(times)
        p50 = times[n // 2]
        p95 = times[int(n * 0.95)]
        print(f"  {stage_name:10s}  p50={p50:7.2f} ms  p95={p95:7.2f} ms")

    # 识别瓶颈
    max_stage = max(stages.items(), key=lambda x: sum(x[1]) / len(x[1]))
    bottleneck_name = max_stage[0]
    avg_ms = sum(max_stage[1]) / len(max_stage[1])
    print(f"\n瓶颈阶段: {bottleneck_name} ({avg_ms:.2f} ms, 平均)")


# ===========================================================================
# 第 5 步：FastAPI 服务 —— 将管线包装为 HTTP API
# ===========================================================================

def create_fastapi_app(pipe: VisionPipeline):
    """创建一个 FastAPI 应用，提供图像检测和分类接口。

    生产部署时的完整服务还会包含：
    - 健康检查端点（/health）
    - 速率限制
    - 请求日志与链路追踪
    - 多模态版本管理
    """
    from fastapi import FastAPI, UploadFile, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Vision Pipeline Service")

    @app.get("/health")
    async def health_check():
        """健康检查——负载均衡器依赖此端点判断服务是否可用。"""
        return {"status": "ok", "trace_id": pipe.trace_id}

    @app.post("/detect")
    async def detect_endpoint(file: UploadFile):
        """接收上传的图片文件，运行检测 + 分类管线，返回结构化 JSON。"""
        # 验证文件类型
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "unsupported_content_type", "message": f"不支持的内容类型: {file.content_type}"},
            )

        # 读取并解码图像
        data = await file.read()
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "image_decode_failed", "message": "无法解码图像文件"},
            )

        # 运行管线
        image_id = file.filename or hashlib.md5(data).hexdigest()[:8]
        result = pipe.run(img, image_id=image_id)

        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"X-Trace-ID": pipe.trace_id},
        )

    return app


# ===========================================================================
# 主程序：演示完整流程
# ===========================================================================

def main():
    """演示端到端视觉管线：初始化 → 推理 → 基准测试。"""

    # --- 初始化占位模型 ---
    detector = StubDetector()
    classifier = StubClassifier(num_classes=10)
    class_names = [f"类别_{i}" for i in range(10)]

    # --- 创建管线实例 ---
    pipe = VisionPipeline(
        detector=detector,
        classifier=classifier,
        class_names=class_names,
        device="cpu",
        min_crop_size=16,
        trace_id="demo-trace-001",
    )

    # --- 用合成图像测试 ---
    test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
    result = pipe.run(test_image, image_id="demo-image")

    print("=" * 55)
    print("检测结果")
    print("=" * 55)
    output_json = result.model_dump_json(indent=2)
    print(output_json[:600])
    print("...")

    print(f"\n检测数量: {len(result.detections)}")
    print(f"分类数量: {len(result.classifications)}")
    print(f"总推理时间: {result.inference_ms:.2f} ms")

    # --- 运行基准测试 ---
    print("\n" + "=" * 55)
    benchmark(pipe, num_runs=10)


if __name__ == "__main__":
    main()
