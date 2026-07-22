# main.py — 从零实现简易 YOLO 检测器核心组件
# 依赖：torch>=2.0, numpy
# 对应课程：阶段 04 · 06（目标检测 YOLO）

import math
import numpy as np


# ===========================================================================
# 工具函数
# ===========================================================================

def sigmoid(x):
    """数值稳定的 Sigmoid 函数。"""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# ===========================================================================
# 第 1 步：IoU 计算 — 目标检测中一切比较的基石
# ===========================================================================

def box_iou(boxes_a, boxes_b):
    """
    计算两组边界框之间的交并比矩阵。

    每个框用 (x1, y1, x2, y2) 表示，即左上角和右下角坐标。

    Args:
        boxes_a: 形状 (N, 4) 的 NumPy 数组
        boxes_b: 形状 (M, 4) 的 NumPy 数组

    Returns:
        IoU 矩阵，形状 (N, M)，值为 [0, 1]
    """
    ax1 = boxes_a[:, 0]
    ay1 = boxes_a[:, 1]
    ax2 = boxes_a[:, 2]
    ay2 = boxes_a[:, 3]

    bx1 = boxes_b[:, 0]
    by1 = boxes_b[:, 1]
    bx2 = boxes_b[:, 2]
    by2 = boxes_b[:, 3]

    # 交集的左上角和右下角
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    # 交集面积（clip 保证非负）
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    # 各自的面积
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    # 并集面积
    union = area_a[:, None] + area_b[None, :] - inter_area

    # 防止除以零
    return inter_area / np.clip(union, 1e-8, None)


# ===========================================================================
# 第 2 步：GIoU 损失 — 解决 IoU 无法回传梯度为零的问题
# ===========================================================================

def calc_giou(box_a, box_b):
    """
    计算两个框的 GIoU（Generalized IoU）。

    当两个框没有交集时，标准 IoU 返回 0 且梯度为零，导致无法优化。
    GIoU 通过引入最小闭包区域解决了这个问题，保证了即使在完全不相交时
    也能回传有意义的梯度。

    公式：GIoU = IoU - |C \ (A ∪ B)| / |C|

    其中 C 是包含 A 和 B 的最小闭包矩形。

    Args:
        box_a: 框 A (x1, y1, x2, y2)
        box_b: 框 B (x1, y1, x2, y2)

    Returns:
        GIoU 值，范围 (-1, 1]
    """
    iou = box_iou(
        box_a.reshape(1, -1),
        box_b.reshape(1, -1)
    )[0, 0]

    # 最小闭包区域的左上和右下
    c_x1 = min(box_a[0], box_b[0])
    c_y1 = min(box_a[1], box_b[1])
    c_x2 = max(box_a[2], box_b[2])
    c_y2 = max(box_a[3], box_b[3])

    # 闭包区域面积
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    if c_area == 0:
        return float("-inf")

    # 并集面积
    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = a_area + b_area - iou * min(a_area, b_area)

    # GIoU = IoU - |C \\ (A ∪ B)| / |C|
    giou = iou - (c_area - union_area) / c_area

    return giou


def giou_loss(box_pred, box_target):
    """
    GIoU 损失：1 - GIoU，用于回归头训练。

    相比于 L1/MSE 位置损失，GIoU 直接优化检测质量指标本身，
    能让模型学到更紧致的边界框。
    """
    g = calc_giou(box_pred, box_target)
    return 1.0 - g


# ===========================================================================
# 第 3 步：非极大值抑制（NMS）— 消除重复检测
# ===========================================================================

def nms(boxes, scores, iou_threshold=0.45):
    """
    非极大值抑制：保留最高分，消除重叠度高的冗余预测。

    工作流程：
      1. 按分数从高到低排序
      2. 选出当前最高分的框，加入结果列表
      3. 与所有剩余框计算 IoU，移除超过阈值的重复框
      4. 对剩下的框重复上述步骤

    Args:
        boxes: 形状 (N, 4)，格式 (x1, y1, x2, y2)
        scores: 形状 (N,) 对应的置信度分数
        iou_threshold: IoU 阈值，默认 0.45

    Returns:
        保留的框索引数组
    """
    # 按分数降序排列
    order = np.argsort(-scores)
    keep = []

    while len(order) > 0:
        # 取出最高分的框
        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        # 计算最高分框与其余框的 IoU
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]

        # 只保留 IoU <= 阈值的框（消除高重叠的冗余框）
        order = rest[ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)


# ===========================================================================
# 第 4 步：编码与解码 — 像素坐标 <-> 网络回归目标
# ===========================================================================

def encode_box(gt_xyxy, cell_x, cell_y, stride, anchor_wh):
    """
    将真实边界框编码为网络需要学习的回归目标。

    网络不直接预测 (x, y, w, h)，而是预测偏移量 (tx, ty, tw, th)：
      - tx, ty：中心点相对于网格单元格的偏移
      - tw, th：宽度高度相对于 anchor 的对数比例

    Args:
        gt_xyxy: 真实框 (x1, y1, x2, y2)
        cell_x: 网格单元格 x 坐标
        cell_y: 网格单元格 y 坐标
        stride: 下采样步长（如 32）
        anchor_wh: anchor 宽高

    Returns:
        [tx, ty, tw, th] 四个回归目标
    """
    x1, y1, x2, y2 = gt_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = x2 - x1
    bh = y2 - y1

    # 中心点相对于网格的位置 [0, 1]，取 logit 反变换
    off_x = np.clip(cx / stride - cell_x, 1e-6, 1 - 1e-6)
    off_y = np.clip(cy / stride - cell_y, 1e-6, 1 - 1e-6)
    tx = float(np.log(off_x / (1 - off_x)))
    ty = float(np.log(off_y / (1 - off_y)))

    # 宽高相对于 anchor 的对数缩放
    tw = np.log(bw / anchor_wh[0] + 1e-8)
    th = np.log(bh / anchor_wh[1] + 1e-8)

    return np.array([tx, ty, tw, th])


def decode_box(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    """
    将网络的原始输出解码为像素坐标系中的边界框。

    解码公式（每个 YOLO 版本通用）：
      cx = (sigmoid(tx) + cell_x) * stride
      cy = (sigmoid(ty) + cell_y) * stride
      w  = anchor_w * exp(tw)
      h  = anchor_h * exp(th)

    Args:
        tx_ty_tw_th: 网络的原始回归输出 (tx, ty, tw, th)
        cell_x: 网格单元格 x 坐标
        cell_y: 网格单元格 y 坐标
        stride: 下采样步长
        anchor_wh: anchor 宽高

    Returns:
        解码后的边界框 (x1, y1, x2, y2)
    """
    tx, ty, tw, th = tx_ty_tw_th

    # 限制 exp 的范围防止爆炸
    tw = np.clip(tw, -10.0, 10.0)
    th = np.clip(th, -10.0, 10.0)

    # 中心点坐标
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride

    # 宽高
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)

    # 转换为 (x1, y1, x2, y2)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ===========================================================================
# 第 5 步：锚框设计 — Anchor-Based vs Anchor-Free 对比
# ===========================================================================

def assign_anchor(boxes_xyxy, classes, anchors, num_classes, grid_size, stride=32):
    """
    将真实标签分配到网格单元和锚框。

    每个真实框分配给其中心点落入的那个网格单元，以及与该框 IoU 最大的 anchor。
    这就是 Anchor-Based 策略的核心思想。

    Anchor-Free 变体（如 FCOS、CenterNet）不需要此分配步骤——
    每个单元格直接预测物体中心，Anchor-Free 的 label assignment 更简单：
    找到离中心最近的单元格，分配给它即可。

    Args:
        boxes_xyxy: 真实框列表 [(x1,y1,x2,y2), ...]
        classes: 类别标签列表
        anchors: anchor 列表 [(w, h), ...]
        num_classes: 类别数量
        grid_size: 网格尺寸（如 13、26、52）
        stride: 下采样步长

    Returns:
        target: 目标张量 (grid, grid, num_anchors, 5+num_classes)
        has_obj: 有效对象掩码 (grid, grid, num_anchors)
    """
    num_anchors = len(anchors)
    target = np.zeros(
        (grid_size, grid_size, num_anchors, 5 + num_classes),
        dtype=np.float32
    )
    has_obj = np.zeros(
        (grid_size, grid_size, num_anchors),
        dtype=bool
    )

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = x2 - x1
        bh = y2 - y1

        # 确定属于哪个网格单元
        gx_raw = int(cx / stride)
        gy_raw = int(cy / stride)

        # 越界的裁剪到边缘
        gx = min(max(gx_raw, 0), grid_size - 1)
        gy = min(max(gy_raw, 0), grid_size - 1)

        # 找出 IoU 最大的 anchor
        best_anchor = 0
        best_iou = -1.0
        for i, (aw, ah) in enumerate(anchors):
            inter = min(bw, aw) * min(bh, ah)
            union = bw * bh + aw * ah - inter
            iou = inter / max(union, 1e-8)
            if iou > best_iou:
                best_iou = iou
                best_anchor = i

        # 存储编码后的回归目标
        aw, ah = anchors[best_anchor]
        enc = encode_box(box, gx, gy, stride, (aw, ah))
        target[gy, gx, best_anchor, 0] = enc[0]  # tx
        target[gy, gx, best_anchor, 1] = enc[1]  # ty
        target[gy, gx, best_anchor, 2] = enc[2]  # tw
        target[gy, gx, best_anchor, 3] = enc[3]  # th
        target[gy, gx, best_anchor, 4] = 1.0      # 有物体
        target[gy, gx, best_anchor, 5 + cls] = 1.0  # 类别概率
        has_obj[gy, gx, best_anchor] = True

    return target, has_obj


# ===========================================================================
# 第 6 步：YOLO Loss — 三个损失的加权组合
# ===========================================================================

def compute_yolo_loss(pred, target, has_obj,
                      lambda_coord=5.0,
                      lambda_obj=1.0,
                      lambda_noobj=0.5,
                      lambda_cls=1.0):
    """
    计算 YOLO 风格的综合损失（纯 NumPy 版）。

    YOLO 损失 = 位置损失 + 置信度损失 + 分类损失

    权重设计的核心逻辑：
      - lambda_coord > 1：位置回归的损失值较小，需要放大以平衡梯度
      - lambda_noobj < 1：绝大多数网格不含物体，不放大会主导总损失
      - 只有含物体的网格才参与位置和分类损失

    Args:
        pred: 模型输出 numpy 数组，形状 (B, H, W, A, 5+C)
        target: 编码后的目标，形状与 pred 一致
        has_obj: 有效对象掩码，形状 (B, H, W, A)
        lambda_coord: 坐标损失权重（通常 5.0）
        lambda_obj: 正样本置信度权重（通常 1.0）
        lambda_noobj: 负样本置信度权重（通常 0.5）
        lambda_cls: 分类损失权重（通常 1.0）

    Returns:
        total_loss: 总损失标量
        loss_parts: 各分量损失的字典
    """
    # 展开为 (B*H*W*A, 5+C) 方便索引
    bs = pred.shape[0]
    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    flat_has_obj = has_obj.astype(bool).reshape(-1)

    # === 坐标回归损失（仅有效网格，MSE）===
    if flat_has_obj.any():
        box_pred = flat_pred[flat_has_obj, :4]
        box_true = flat_target[flat_has_obj, :4]
        loss_box = np.sum((box_pred - box_true) ** 2)
    else:
        loss_box = 0.0

    # === 置信度损失（BCE-like）===
    def bce_loss(raw, label):
        """二分类交叉熵：log(sigmoid(x)) * l + log(1-sigmoid(x)) * (1-l)"""
        s = sigmoid(raw)
        return -(label * np.log(s + 1e-8) + (1 - label) * np.log(1 - s + 1e-8)).sum()

    loss_obj_pos = bce_loss(
        flat_pred[flat_has_obj, 4],
        flat_target[flat_has_obj, 4]
    ) if flat_has_obj.any() else 0.0
    loss_obj_neg = bce_loss(
        flat_pred[~flat_has_obj, 4],
        flat_target[~flat_has_obj, 4]
    )

    # === 分类损失（仅有效网格）===
    loss_cls = bce_loss(
        flat_pred[flat_has_obj, 5:],
        flat_target[flat_has_obj, 5:]
    ) if flat_has_obj.any() else 0.0

    # === 加权汇总 ===
    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)

    return float(total), {
        "box": float(loss_box),
        "obj_pos": float(loss_obj_pos),
        "obj_neg": float(loss_obj_neg),
        "cls": float(loss_cls),
    }


# ===========================================================================
# 第 7 步：模拟 YOLO 检测头输出
# ===========================================================================

def simulate_yolo_head(batch_size, in_channels, grid_h, grid_w,
                       num_anchors, num_classes, seed=0):
    """
    模拟 YOLO 检测头的输出（纯 NumPy 版）。

    在真实训练中，这里是一个 1x1 卷积层；
    我们用随机张量代替以演示完整的损失计算流程。

    YOLOv3/v5 的实际做法：
      - 多尺度预测：在 52x52、26x26、13x13 三个分辨率各出一个检测头
      - FPN + PANet：特征金字塔 + 路径增强
      - 解耦头：分类头和回归头用独立卷积分支
    """
    rng = np.random.default_rng(seed)
    pred = rng.standard_normal(
        (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
    ).astype(np.float32)
    return pred


# ===========================================================================
# 第 8 步：推理后处理 —— 解码 + 阈值过滤 + NMS
# ===========================================================================

def postprocess_predictions(
    pred_tensor,
    anchors,
    stride,
    conf_threshold=0.25,
    iou_threshold=0.45,
    num_classes=5,
):
    """
    将检测头的原始输出解码为最终的检测框。

    流程：解码所有锚框 -> 按置信度阈值过滤 -> NMS 去重

    Args:
        pred_tensor: 检测头输出 (B, H, W, num_anchors, 5+C)
        anchors: anchor 列表 [(w, h), ...]
        stride: 网格步长
        conf_threshold: 置信度阈值
        iou_threshold: NMS 的 IoU 阈值
        num_classes: 类别数量

    Returns:
        kept_boxes: 保留的框 (N, 4)
        kept_scores: 保留的分数 (N,)
        kept_classes: 保留的类别 (N,)
    """
    pred = pred_tensor
    batch_size, grid_h, grid_w, num_anchors, _ = pred.shape

    all_boxes = []
    all_scores = []
    all_classes = []

    for b in range(batch_size):
        for gy in range(grid_h):
            for gx in range(grid_w):
                for a in range(num_anchors):
                    row = pred[b, gy, gx, a]
                    tx, ty, tw, th, obj_logit = row[:5]
                    cls_logits = row[5:]

                    # 计算最终得分：置信度 * 最大类概率
                    obj_score = sigmoid(obj_logit)
                    cls_probs = sigmoid(cls_logits)
                    final_score = float(obj_score * cls_probs.max())

                    # 低于阈值的直接丢弃
                    if final_score < conf_threshold:
                        continue

                    cls_idx = int(np.argmax(cls_probs))
                    cx = (sigmoid(tx) + gx) * stride
                    cy = (sigmoid(ty) + gy) * stride
                    w = anchors[a][0] * np.exp(np.clip(tw, -10.0, 10.0))
                    h = anchors[a][1] * np.exp(np.clip(th, -10.0, 10.0))

                    all_boxes.append([
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2,
                    ])
                    all_scores.append(final_score)
                    all_classes.append(cls_idx)

        if not all_boxes:
            return (
                np.zeros((0, 4)),
                np.zeros((0,)),
                np.zeros((0,), dtype=int),
            )

        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        classes = np.array(all_classes)
        keep = nms(boxes, scores, iou_threshold=iou_threshold)

    return boxes[keep], scores[keep], classes[keep]


# ===========================================================================
# 第 9 步：Anchor-Free 简化版本（模拟 YOLOv8 思路）
# ===========================================================================

def anchor_free_assign(gt_boxes, gt_classes, grid_h, grid_w, stride=8):
    """
    Anchor-Free 的标签分配方式（简化版，模拟 FCOS/CenterNet 思路）。

    不再有 anchor 维度，每个网格单元只需预测：
      - 物体概率 (1 维)
      - 类别概率 (C 维)
      - 四个偏移量：left, right, top, bottom（到物体边界的距离）

    这比 Anchor-Based 更简洁——不需要 k-means 找 anchor，
    也不需要选择 anchor（Anchor-Free 不存在这个难题）。

    Args:
        gt_boxes: 真实框列表
        gt_classes: 类别标签列表
        grid_h: 网格行数
        grid_w: 网格列数
        stride: 下采样步长

    Returns:
        target: 目标张量 (grid_h, grid_w, 5+C)
    """
    target = np.zeros(
        (grid_h, grid_w, 5 + gt_classes.shape[0] if hasattr(gt_classes, 'shape') else 5),
        dtype=np.float32
    )

    for box, cls in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        gx = min(max(int(cx / stride), 0), grid_w - 1)
        gy = min(max(int(cy / stride), 0), grid_h - 1)

        # 记录类别和物体存在
        target[gy, gx, 4] = 1.0       # objectness
        target[gy, gx, 5 + cls] = 1.0  # 类别概率

    return target


# ===========================================================================
# 主程序：运行所有演示
# ===========================================================================

def main():
    print("=" * 60)
    print("YOLO 目标检测器 -- 从零实现演示")
    print("=" * 60)

    # --- 测试 1: IoU ---
    print("\n--- 测试 1: IoU 计算 ---")
    identical = np.array([[10.0, 10.0, 50.0, 50.0]])
    same_box = np.array([[10.0, 10.0, 50.0, 50.0]])
    half_overlap = np.array([[0.0, 0.0, 10.0, 10.0]])
    another_box = np.array([[5.0, 0.0, 15.0, 10.0]])
    no_over = np.array([[100.0, 100.0, 200.0, 200.0]])

    print(f"  相同框的 IoU:           {box_iou(identical, same_box)[0, 0]:.3f}  (预期 1.0)")
    print(f"  一半重叠的 IoU:         {box_iou(half_overlap, another_box)[0, 0]:.3f}  (预期 1/3 = 0.333)")
    print(f"  不重叠的 IoU:           {box_iou(identical, no_over)[0, 0]:.3f}  (预期 0.0)")

    # --- 测试 2: GIoU ---
    print("\n--- 测试 2: GIoU 损失 ---")
    close = np.array([20.0, 20.0, 60.0, 60.0])
    far = np.array([100.0, 100.0, 200.0, 200.0])
    anchor_box = np.array([10.0, 10.0, 50.0, 50.0])
    print(f"  接近目标的 GIoU:    {calc_giou(anchor_box, close):.3f}")
    print(f"  远离目标的 GIoU:    {calc_giou(anchor_box, far):.3f}")
    print(f"  GIoU 始终有梯度，即使不相交 (标准 IoU 会是 0)")

    # --- 测试 3: NMS ---
    print("\n--- 测试 3: NMS 去重 ---")
    nms_boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [2.0, 2.0, 12.0, 12.0],
        [20.0, 20.0, 30.0, 30.0],
        [21.0, 21.0, 31.0, 31.0],
    ], dtype=float)
    nms_scores = np.array([0.9, 0.8, 0.7, 0.85, 0.6])
    keep = nms(nms_boxes, nms_scores, iou_threshold=0.4)
    print(f"  输入 {len(nms_boxes)} 个框，NMS 后保留 {len(keep)} 个")
    print(f"  保留索引: {keep.tolist()}  (预期: 每组最高的 0 和 3)")

    # --- 测试 4: 编码/解码往返 ---
    print("\n--- 测试 4: 编码/解码往返 ---")
    test_anchor = [(30.0, 60.0), (75.0, 170.0), (200.0, 380.0)]
    test_stride = 32
    test_grid = 13
    orig_box = np.array([120.0, 80.0, 240.0, 220.0])
    cx, cy = 0.5 * (orig_box[0] + orig_box[2]), 0.5 * (orig_box[1] + orig_box[3])
    cell_x = int(cx / test_stride)
    cell_y = int(cy / test_stride)
    encoded = encode_box(orig_box, cell_x, cell_y, test_stride, test_anchor[1])
    decoded = decode_box(encoded, cell_x, cell_y, test_stride, test_anchor[1])
    error = np.max(np.abs(orig_box - decoded))
    print(f"  原始框:   {orig_box}")
    print(f"  编码后:   {[f'{e:.4f}' for e in encoded]}")
    print(f"  解码后:   {decoded.round(2)}")
    print(f"  最大误差: {error:.4f}")

    # --- 测试 5: 完整 YOLO 管线（损失计算）---
    print("\n--- 测试 5: 完整 YOLO 管线 ---")
    gt_boxes = [[50.0, 50.0, 200.0, 200.0]]
    gt_classes = np.array([1])
    num_classes = 3
    anchors_list = [(30.0, 60.0), (75.0, 170.0), (200.0, 380.0)]

    # 分配标签
    target, has_obj = assign_anchor(
        gt_boxes, gt_classes, anchors_list,
        num_classes, test_grid, test_stride
    )
    print(f"  目标形状: {target.shape}  (13, 13, 3, 5+{num_classes})")
    print(f"  有效对象数: {int(has_obj.sum())}")

    # 模拟前向传播
    raw_pred = simulate_yolo_head(
        batch_size=1,
        in_channels=128,
        grid_h=test_grid,
        grid_w=test_grid,
        num_anchors=3,
        num_classes=num_classes,
        seed=0,
    )
    print(f"  模拟检测头输出形状: {tuple(raw_pred.shape)}")

    # 广播 has_obj 从 (H,W,A) 到 (B,H,W,A)
    has_obj_batched = np.expand_dims(has_obj, axis=0)

    # 计算损失
    loss_val, loss_parts = compute_yolo_loss(raw_pred, target, has_obj_batched)
    print(f"  总损失:           {loss_val:.4f}")
    for name, val in loss_parts.items():
        print(f"    - {name:>10s}: {val:.4f}")

    # --- 测试 6: 推理后处理演示 ---
    print("\n--- 测试 6: 推理后处理 ---")
    boxes, scores, classes = postprocess_predictions(
        raw_pred, anchors_list, test_stride,
        conf_threshold=0.01,
        num_classes=num_classes,
    )
    print(f"  置信度阈值 0.01 以下，NMS (0.45) 后保留 {len(boxes)} 个框")
    if len(boxes) > 0:
        print(f"  分数范围: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  前 3 个框: ")
        for i in range(min(3, len(boxes))):
            print(f"    框 {i+1}: {boxes[i].round(1)}  分数: {scores[i]:.3f}  类别: {classes[i]}")

    # --- 测试 7: Anchor-Based vs Anchor-Free ---
    print("\n--- 测试 7: Anchor-Based vs Anchor-Free ---")
    gt_classes_arr = np.array([0, 1, 2, 0, 2])
    gt_box_arr = [[10, 10, 80, 80], [100, 100, 200, 200], [50, 50, 150, 150], [200, 10, 250, 60], [150, 150, 220, 220]]
    af_target = anchor_free_assign(gt_box_arr, gt_classes_arr, test_grid, test_grid, stride=test_stride)
    af_shape = af_target.shape
    ab_shape = (test_grid, test_grid, 3, 5 + num_classes)
    print(f"  Anchor-Based 输出形状: {ab_shape}")
    print(f"  Anchor-Free 输出形状:  {af_shape}")
    print(f"  Anchor-Free 少了 anchor 维度，结构更简洁")

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
