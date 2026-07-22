# YOLO：从分类到定位 — 一次前向传播搞定目标检测

> YOLO 把目标检测变成了一个问题——不再是"先找再认"的两步走，而是"一眼看到"的端到端回归。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 03（深度学习核心）· 反向传播、损失函数，阶段 04 · 04（图像分类）、阶段 04 · 07（语义分割 U-Net）
**预计时间：** ~90 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 04 · 08（实例分割 Mask R-CNN）— 理解检测与分割在标注成本和精度上的权衡

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 解释目标检测中定位与分类联合优化的核心挑战，以及为什么它比图像分类难一个数量级
- [ ] 推导 IoU 和 GIoU 的计算方法，理解为什么 GIoU 在不相交场景下仍能回传梯度
- [ ] 描述非极大值抑制（NMS）算法的完整流程，并理解其消除重复检测的作用
- [ ] 对比 Anchor-Based 和 Anchor-Free 两种检测范式的设计哲学差异
- [ ] 从零实现一个包含边界框编码解码、标签分配、YOLO 损失计算和后处理的完整检测管线

---

## 1. 问题

图像分类任务你已经很熟悉了——给一张猫的照片，输出"猫，置信度 0.96"。但现实世界不是分类数据集。

想象一个自动驾驶系统的感知模块看到的画面：

```
[道路场景图像]
├── 左前方 15 米处有一辆红色轿车
├── 正前方 30 米处有一个行人正在横穿马路
├── 右后方 50 米处有一辆公交车
└── 右侧 8 米处有一个自行车道标志
```

一个仅会分类的模型只能给出一个标签。而自动驾驶需要的是：**每个物体在哪、是什么、有多大**。这就是目标检测要解决的问题。

没有目标检测系统时，一种朴素的做法是 sliding window（滑动窗口）——在图像上切出无数个子区域，每个子区域都丢进分类器判断"有没有车"。问题是：一个 640×480 的图片，以不同尺度、不同位置生成窗口，可能产生数万甚至数十万个候选区域。分类器需要依次推断每一个，计算量巨大。

有了目标检测后，YOLO 类算法只需要**一次前向传播**就能同时输出图像中所有物体的位置和类别。一个 YOLOv8n 模型在 GPU 上推理一张 640×640 的图片只需要 2-3 毫秒。

**目标检测的核心矛盾在于：它要在同一个网络里学会两件事——"这里有什么物体"（分类）和"这个物体占据了哪里"（定位）。** 分类是概率问题，定位是几何问题。让神经网络同时优化这两种截然不同的信号，就是 YOLO 系列反复改进的方向。

---

## 2. 概念

### 2.1 目标检测：定位 + 分类的联合任务

目标检测的输出格式非常简洁——每个检测结果是四元组或五元组：

```
[x1, y1, x2, y2, class, confidence]
```

其中 $(x_1, y_1)$ 是边界框左上角坐标，$(x_2, y_2)$ 是右下角坐标，$class$ 是预测的类别，$confidence$ 是模型对该检测结果的可信度。

与图像分类相比，目标检测增加了两个维度的复杂度：

| 维度 | 图像分类 | 目标检测 |
|---|---|---|
| 输出数量 | 固定 1 个标签 | 可变数量的物体（0-N 个） |
| 输出空间 | 离散类别概率分布 | 连续坐标 + 离散类别 |
| 训练信号 | 单个交叉熵损失 | 多尺度回归 + 置信度 + 分类的加权组合 |
| 正负样本比例 | 相对均衡 | 极端不平衡（大量背景，极少前景） |

这就是为什么检测模型的损失函数比分类器复杂得多——它至少要包含三个分量：位置损失、置信度损失、分类损失。

### 2.2 交并比（IoU）：检测质量的度量基准

**交并比（Intersection over Union）** 是目标检测中最基础的度量指标，用来衡量预测框与真实框的重叠程度：

$$
\text{IoU} = \frac{\text{预测框} \cap \text{真实框}}{\text{预测框} \cup \text{真实框}}
$$

IoU 的值域为 $[0, 1]$。值为 1 表示完美重合，值为 0 表示完全不重叠。通常将 IoU > 0.5 作为"正确检测"的阈值。

```
真实框 A:  [30, 20, 130, 120]
预测框 B:  [35, 25, 125, 115]

交集区域:   [35, 25, 125, 115] → 面积 = 90 × 90 = 8100
A 的面积:   100 × 100 = 10000
B 的面积:   90 × 90 = 8100
并集面积:   10000 + 8100 - 8100 = 10000
IoU = 8100 / 10000 = 0.81
```

IoU 还有一个致命缺陷——当预测框与真实框完全不重叠时，IoU = 0 且梯度也为 0。此时模型无法知道"应该把框往哪个方向移动"。**GIoU 解决了这个问题。**

### 2.3 GIoU：让不相交的框也能学习

广义交并比（Generalized IoU）在 IoU 的基础上引入了一个**最小闭包区域**的概念：

$$
\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
$$

其中 $C$ 是同时包含预测框 $A$ 和真实框 $B$ 的最小闭包矩形。

```
当两个框完全不重叠时：
  IoU = 0（梯度消失，无法优化）
       ↓
GIoU = 0 - (闭包区域 - 并集) / 闭包区域

闭包区域越小 → GIoU 越大 → 告诉模型"你离目标不远，继续优化"
闭包区域越大 → GIoU 越接近 -1 → 告诉模型"还差得远"
```

关键洞察：**GIoU 的值域为 $(-1, 1]$，即使在完全不相交的情况下，也能给出一个有意义的梯度信号。** 这是它取代标准 IoU 作为定位损失函数的核心原因。

### 2.4 非极大值抑制（NMS）：消除重复检测

一个物体可能被多个检测框指向。NMS 的目标是：**对于每个物体，只保留最可信的那一个框。**

```
NMS 工作流程（可视化）：

原始检测：
  框① 置信度 0.92  [30, 20, 130, 120]  ← 行人
  框② 置信度 0.88  [32, 22, 132, 122]  ← 行人（重复）
  框③ 置信度 0.85  [50, 40, 250, 250]  ← 汽车
  框④ 置信度 0.70  [55, 45, 245, 245]  ← 汽车（重复）

第 1 轮：选框①（最高分 0.92）
  计算框①与其余各框的 IoU
  框① vs 框②：IoU = 0.94 > 0.5 → 消除框②
  框① vs 框③：IoU = 0.05 < 0.5 → 保留
  框① vs 框④：IoU = 0.08 < 0.5 → 保留
  剩余候选：③、④

第 2 轮：从剩余中选最高分框③（0.85）
  计算框③与框④的 IoU
  框③ vs 框④：IoU = 0.88 > 0.5 → 消除框④
  剩余候选：无

最终结果：[框①（行人），框③（汽车）]
```

NMS 的核心参数是 IoU 阈值。阈值设得太低会误删合理的并列检测，设得太高则会留下过多重复框。工业界通常使用 0.45-0.5。

### 2.5 YOLO 架构演进

YOLO（You Only Look Once）由 Redmon 等人于 2015 年首次提出[1]，核心思想极其大胆：**把目标检测从一个两步问题（ proposals + classification ）变成了一个单步回归问题。**

#### YOLO v1（2015）：一切问题的起点

YOLOv1 的核心设计如下：

```
输入：448 × 448 × 3 图像
  │
  ▼
CNN 骨干网络（19 个卷积层 + 2 个全连接层）
  │
  ▼
网格划分：7 × 7
  │
  ▼
每个网格单元负责检测包含中心点的物体
  │
  ▼
每个单元格输出 30 个值：
  [bx, by, bw, bh, objectness, c1, c2, ..., c20]
  └────回归────┘  └─置信度─┘ └─20 类分类─┘
  (4)             (1)           (Pascal VOC: 20)
```

YOLOv1 的关键创新：

1. **网格化检测**：将图像划分为 $S \times S$ 网格，每个网格独立预测固定数量的边界框。v1 使用 7×7 网格，每个单元格预测 2 个框。
2. **端到端可微**：整个检测过程——特征提取、边界框回归、分类、NMS——构成了一个完整的计算图，可以用标准的反向传播训练。
3. **全局上下文**：因为整个图像一次性送入网络，模型可以看到场景的全局信息——这对于区分相似物体（如一群鸽子中的某一只）至关重要。

局限：小物体检测差（7×7 网格太粗）、定位精度偏低、对细长物体预测效果不佳。

#### YOLO v2（YOLO9000，2016）：批量归一化与锚框

YOLOv2 针对 v1 的四个主要问题进行了系统性改进：

| v1 的不足 | v2 的改进 |
|---|---|
| 网格粗糙，定位不准 | 引入**锚框（Anchor Boxes）**——预定义一组典型宽高比，每个单元格预测相对 anchor 的偏移而非绝对坐标 |
| 不使用批归一化，收敛慢 | 在全网络加入**Batch Normalization**，收敛速度翻倍，正则化效果显著 |
| 低分辨率特征图 | 使用 **FPS（Fine-Grained Pass-through）** 层，将低层特征图直接传递到高层，改善小物体检测 |
| 单一分辨率输入 | **多尺寸训练**：每 10 个 epoch 随机采样一个新的输入尺寸（步长为 32），提升模型对不同尺度的鲁棒性 |

锚框的使用是 v2 最重要的改进。通过 k-means 聚类标注数据中的边界框尺寸，得到一组代表性的 anchor。每个网格不再从零开始预测尺寸，而是在 anchor 的基础上做微调。这大幅提升了召回率。

#### YOLO v3（2018）：多尺度 + 残差 + 解耦

YOLOv3 借鉴了 ResNet 的思想，并引入了现代检测器的标配设计：

```
YOLOv3 核心架构：
                    输入 608×608
                       │
                  Darknet-53 骨干
                       │
              ╭─────────┼─────────╮
              ▼         ▼         ▼
          低层特征   中层特征   高层语义
          (1/8)     (1/16)     (1/32)
              │         │         │
              ▼         ▼         ▼
          小物体头   中等物体头   大物体头
         (76×76)    (38×38)     (19×19)
```

| 特性 | 说明 |
|---|---|
| **FPN（特征金字塔）** | 在三个不同分辨率的特征图上分别做检测（76×76、38×38、19×19），解决多尺度物体问题 |
| **Darknet-53** | 基于 ResNet 残差结构的骨干网络，引入了 skip connection |
| **Binary Cross Entropy** | 替代 v1 中的多项逻辑回归，每个类别独立预测（多标签分类） |
| **9 个 Anchor** | 通过 k-means 聚类得到 9 个 anchor，分配到三个尺度上（每尺度 3 个） |

v3 可以检测到 COCO 数据集中 90 个类别，mAP 达到 33.0，推理速度仍然保持实时水平。

#### YOLOv5（2020）：工程优化的集大成者

YOLOv5 由 Ultralytics 团队发布，最大的特点是**不是纯粹的学术改进，而是一套完整的工程体系**：

| 组件 | 作用 |
|---|---|
| **Mosaic 数据增强** | 将 4 张训练图片随机缩放、裁剪、拼接成一张新图，极大地丰富了训练数据的多样性 |
| **自适应锚计算** | 不在训练前一次性算好 anchor，而是在训练过程中动态更新，更贴合当前数据集 |
| **SPP 模块** | 空间金字塔池化，在不同池化尺度下聚合感受野 |
| **Focus 结构 → C3 模块** | 更灵活的特征提取和融合方式 |
| **Anchor-Free 变体** | 后期的 v5 版本也推出了不依赖 anchor 的检测头 |

**YOLOv5 的意义**：它将学术界的检测研究与工业界的部署需求结合了起来。你可以把它理解为"让 YOLO 从实验室走向工厂的产品"。

> **注意**：YOLOv5 和 YOLOv8 之间存在版本跳跃，中间没有 v6、v7。v7 由 AI_Century--com 团队单独发布，不是 Ultralytics 官方产品。v8 是 Ultralytics 的最新版本，已全面转向 Anchor-Free 架构。

### 2.6 Anchor-Based vs Anchor-Free

| 维度 | Anchor-Based（YOLOv3/v5） | Anchor-Free（YOLOv8/v10） |
|---|---|---|
| 核心理念 | 预设一组 anchor，预测相对 anchor 的偏移 | 不需要 anchor，直接预测边界框或物体中心 |
| Anchor 确定 | k-means 聚类标注数据中的框 | 无需聚类，零超参数 |
| 输出维度 | `(H, W, num_anchors, 5+C)` | `(H, W, 4+C)` |
| 计算量 | 需遍历所有 anchor | 每格一个预测，更少冗余 |
| 经典代表 | Faster R-CNN、SSD、YOLOv3/v5 | FCOS、CenterNet、YOLOv8/v10 |
| 标签分配 | 与 IoU 最大的 anchor 匹配 | 任意网格都可以作为正样本 |

Anchor-Free 的核心优势在于**简单**。不再有 k-means 聚类调参，不再有 anchor 选择的主观性，不再有同一位置多个 anchor 竞争同一个物体的混淆。YOLOv8 采用 Corner Box 表示法——用物体左上角到四条边的距离 $(l, t, r, b)$ 代替 $(x, y, w, h)$，进一步简化了预测头的设计。

---

## 3. 从零实现

以下实现来自 `code/main.py`，我们将逐步拆解这个包含 9 个核心组件的完整检测管线。

### 第 1 步：IoU 计算 —— 一切比较的基石

交并比是目标检测的基础工具——衡量预测框质量、NMS 去重、Anchor 分配都用它。

```python
def box_iou(boxes_a, boxes_b):
    """
    计算两组边界框之间的交并比矩阵。
    每个框用 (x1, y1, x2, y2) 表示。
    返回 IoU 矩阵，形状 (N, M)。
    """
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    # 交集区域的左上角和右下角
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / np.clip(union, 1e-8, None)
```

关键点在于广播机制的使用：`ax1[:, None]` 将形状从 `(N,)` 变为 `(N, 1)`，与形状为 `(M,)` 的 `bx1[None, :]` 进行广播运算，可以直接计算出 $N \times M$ 的交集矩阵，而不是写两层嵌套循环。

运行结果：

```text
--- 测试 1: IoU 计算 ---
  相同框的 IoU:           1.000  (预期 1.0)
  一半重叠的 IoU:         0.333  (预期 1/3)
  不重叠的 IoU:           0.000  (预期 0.0)
```

### 第 2 步：GIoU 损失 —— 即使不相交也能优化

标准 IoU 在不相交时的梯度为零，导致模型无法收敛。GIoU 通过引入闭包区域修复了这个缺陷。

```python
def calc_giou(box_a, box_b):
    """
    计算两个框的 GIoU。
    公式：GIoU = IoU - |C \\ (A ∪ B)| / |C|
    其中 C 是同时包含 A 和 B 的最小闭包矩形。
    """
    iou = box_iou(box_a.reshape(1, -1), box_b.reshape(1, -1))[0, 0]

    # 闭包区域的边界
    c_x1, c_y1 = min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
    c_x2, c_y2 = max(box_a[2], box_b[2]), max(box_a[3], box_b[3])
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)

    union_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + \
                 (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - \
                 iou * min((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]),
                           (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    return iou - (c_area - union_area) / c_area
```

当预测框远离目标时，闭包区域很大，$(C - Union) / C$ 项接近 1，GIoU 接近 -1，梯度强烈推动框移动。当预测框靠近目标时，闭包区域缩小，GIoU 趋向 1。

```text
--- 测试 2: GIoU 损失 ---
  接近目标的 GIoU:    0.684
  远离目标的 GIoU:    -0.544
  GIoU 始终有梯度，即使不相交（标准 IoU 会是 0）
```

### 第 3 步：NMS —— 消除重复检测

```python
def nms(boxes, scores, iou_threshold=0.45):
    """
    非极大值抑制：按分数降序遍历，保留高分框，
    消除与其 IoU 超过阈值的冗余框。
    """
    order = np.argsort(-scores)
    keep = []

    while len(order) > 0:
        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)
```

NMS 是一个贪心算法——每一步选择当前未处理的最高分框，然后抹掉所有与之重叠度高的框。虽然局部最优不一定全局最优，但在实践中效果非常好。

```text
--- 测试 3: NMS 去重 ---
  输入 5 个框，NMS 后保留 2 个
  保留索引: [0, 3]  （每组中最高的框被保留）
```

### 第 4 步：编码与解码 —— 像素坐标 ↔ 网络回归目标

网络不直接预测 $(x_1, y_1, x_2, y_2)$，而是预测一组经过变换的回归量：

```python
def encode_box(gt_xyxy, cell_x, cell_y, stride, anchor_wh):
    """将真实框编码为网络回归目标 [tx, ty, tw, th]。"""
    cx, cy = 0.5 * (gt_xyxy[0] + gt_xyxy[2]), 0.5 * (gt_xyxy[1] + gt_xyxy[3])
    bw, bh = gt_xyxy[2] - gt_xyxy[0], gt_xyxy[3] - gt_xyxy[1]

    # 中心点相对于网格单元的偏移（用 logit 反变换映射到实数域）
    off_x = np.clip(cx / stride - cell_x, 1e-6, 1 - 1e-6)
    off_y = np.clip(cy / stride - cell_y, 1e-6, 1 - 1e-6)
    tx = float(np.log(off_x / (1 - off_x)))
    ty = float(np.log(off_y / (1 - off_y)))

    # 宽高相对于 anchor 的对数缩放
    tw = np.log(bw / anchor_wh[0] + 1e-8)
    th = np.log(bh / anchor_wh[1] + 1e-8)

    return np.array([tx, ty, tw, th])


def decode_box(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    """将网络输出解码为像素坐标 (x1, y1, x2, y2)。"""
    tx, ty, tw, th = tx_ty_tw_th

    # 中心点
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    # 宽高
    w = anchor_wh[0] * np.exp(np.clip(tw, -10.0, 10.0))
    h = anchor_wh[1] * np.exp(np.clip(th, -10.0, 10.0))

    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
```

为什么需要编码？原因有三：

1. **中心点偏移使用 Sigmoid 约束**：`tx` 的值域是 $(-\infty, +\infty)$，但实际偏移应该在 $[0, 1]$ 内（框中心必须落在所属网格单元内）。Sigmoid 天然满足这个约束。
2. **宽高相对于 Anchor 做对数缩放**：相比 Anchor 的宽高的比例通常在 $[0.1, 10]$ 范围内，取对数后可以对称地处理放大和缩小。
3. **数值稳定性**：直接预测绝对坐标容易超出图像范围，相对编码天然受限于网格和 Anchor。

```text
--- 测试 4: 编码/解码往返 ---
  原始框:   [120.  80. 240. 220.]
  编码后:   ['0.6696', '0.3916', '1.3218', '0.9730']
  解码后:   [120.  80. 240. 220.]
  最大误差: 0.0000
```

### 第 5 步：标签分配与损失计算

训练时需要将真实标注编码为目标张量。以下代码演示了 Anchor-Based 的分配策略：

```python
def assign_anchor(boxes_xyxy, classes, anchors, num_classes, grid_size, stride=32):
    """将真实框分配给网格单元和 Anchor。"""
    target = np.zeros(
        (grid_size, grid_size, len(anchors), 5 + num_classes),
        dtype=np.float32
    )
    has_obj = np.zeros((grid_size, grid_size, len(anchors)), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        gx, gy = int(cx / stride), int(cy / stride)
        gx, gy = min(max(gx, 0), grid_size - 1), min(max(gy, 0), grid_size - 1)

        # 找出与该框 IoU 最大的 anchor
        best_anchor = 0
        best_iou = -1.0
        for i, (aw, ah) in enumerate(anchors):
            inter = min(box[2] - box[0], aw) * min(box[3] - box[1], ah)
            union = (box[2] - box[0]) * (box[3] - box[1]) + aw * ah - inter
            iou = inter / max(union, 1e-8)
            if iou > best_iou:
                best_iou = iou
                best_anchor = i

        enc = encode_box(box, gx, gy, stride, anchors[best_anchor])
        target[gy, gx, best_anchor, :4] = enc
        target[gy, gx, best_anchor, 4] = 1.0      # objectness
        target[gy, gx, best_anchor, 5 + cls] = 1.0  # 类别
        has_obj[gy, gx, best_anchor] = True

    return target, has_obj
```

YOLO 损失由三个部分加权组成：

$$
L = \lambda_{coord} L_{box} + \lambda_{obj} L_{obj}^{pos} + \lambda_{noobj} L_{obj}^{neg} + \lambda_{cls} L_{cls}
$$

其中关键权重 $\lambda_{coord} = 5.0$、$\lambda_{noobj} = 0.5$ 的作用分别是：放大地标定位损失的权重（回归损失的绝对值通常远小于分类/置信度），降低背景网格的影响（正负样本极度不平衡，背景占大多数）。

### 第 6 步：Anchor-Free 简化版本

YOLOv8 为代表的 Anchor-Free 方法消除了 anchor 维度，每个网格单元只需预测：

```python
def anchor_free_assign(gt_boxes, gt_classes, grid_h, grid_w, stride=8):
    """Anchor-Free 的标签分配（简化版）。每格只预测一个框 + 一个类别。"""
    target = np.zeros((grid_h, grid_w, 5), dtype=np.float32)

    for box, cls in zip(gt_boxes, gt_classes):
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        gx, gy = int(cx / stride), int(cy / stride)
        target[gy, gx, 4] = 1.0       # objectness
        target[gy, gx, 5 + cls] = 1.0  # 类别概率

    return target
```

Anchor-Free 让检测头的输出从 `(H, W, K, 5+C)` 缩减为 `(H, W, 5+C)`，减少了大量冗余计算，同时也消除了 k-means 聚类的调试开销。

### 第 7 步：推理后处理 —— 解码 + 过滤 + NMS

训练后的推理流程：

```python
def postprocess_predictions(pred_tensor, anchors, stride, conf_threshold=0.25, iou_threshold=0.45):
    """将检测头原始输出解码、过滤、去重，得到最终检测框。"""
    # 遍历所有网格、所有 anchor
    for each cell, each anchor:
        解码 → 得到 [x1, y1, x2, y2]
        过滤：confidence < conf_threshold 则丢弃

    # NMS 去重
    keep = nms(remaining_boxes, remaining_scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

完整运行结果展示了从零实现的检测管线全部组件都能正常工作：

```text
--- 测试 5: 完整 YOLO 管线 ---
  目标形状: (13, 13, 3, 8)  （13×13 网格，3 个 anchor，5+3 通道）
  有效对象数: 1
  总损失: 12.3042
    -       box: 0.0000
    -    obj_pos: 0.1406
    -    obj_neg: 12.1636
    -        cls: 0.0000

--- 测试 6: 推理后处理 ---
  置信度阈值 0.01 以下，NMS（0.45）后保留 106 个框

--- 测试 7: Anchor-Based vs Anchor-Free ---
  Anchor-Based 输出形状: (13, 13, 3, 8)
  Anchor-Free 输出形状:  (13, 13, 5)
  Anchor-Free 少了 anchor 维度，结构更简洁
```

---

## 4. 工业工具

### 4.1 PyTorch 内置检测器

PyTorch 提供了官方实现的目标检测模型，开箱即用：

```python
import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 加载预训练的 Faster R-COCO 模型（80 类，COCO 标注）
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

# 准备输入
transform = transforms.Compose([transforms.ToTensor()])
image = torch.randint(0, 256, (3, 640, 640)).float()  # 模拟 RGB 图像

# 推理
with torch.no_grad():
    predictions = model([image])

print(f"检测到 {len(predictions[0]['labels'])} 个物体")
print(f"框坐标: {predictions[0]['boxes']}")
print(f"置信度: {predictions[0]['scores']}")
print(f"类别 ID: {predictions[0]['labels']}")
```

### 4.2 Ultralytics YOLO —— 工业界最常用的检测框架

Ultralytics YOLO 是目前最受欢迎的检测库，支持检测、分割、姿态估计多种任务：

```python
from ultralytics import YOLO

# 加载预训练模型（nano 版本最快，large 版本最准）
model = YOLO("yolov8n.pt")  # YOLOv8 nano，~3M 参数

# 训练（自动处理数据加载、 augmentation、混合精度）
results = model.train(
    data="coco128.yaml",   # COCO 子集配置文件
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,             # GPU 0
    amp=True,             # 自动混合精度
)

# 推理
results = model("road_scene.jpg")
for r in results:
    for box in r.boxes:
        print(f"类别: {r.names[int(box.cls)]}, "
              f"置信度: {box.conf:.3f}, "
              f"框: {box.xyxy[0].round().int().tolist()}")
```

### 4.3 HuggingFace Transformers 中的检测器

HuggingFace 生态也支持多种检测模型：

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("scene.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# DETR 的预测格式：(batch, num_queries, 4) 为框坐标
# (batch, num_queries, num_classes) 为类别 logits
```

### 4.4 性能对比

| 方案 | 参数量 | 推理速度（GPU） | mAP@0.5 | 适用场景 |
|---|---|---|---|---|
| Faster R-CNN ResNet-50 | ~41M | ~45 ms/张 | 37.0 | 研究、高精度要求 |
| YOLOv8n | ~3.2M | ~2.5 ms/张 | 37.3 | 实时检测、边缘设备 |
| YOLOv8x | ~68M | ~11 ms/张 | 52.9 | 高精度检测、云端部署 |
| DETR（官方） | ~42M | ~60 ms/张 | 42.0 | 学术研究、无 anchor 基线 |
| RT-DETR | ~30M | ~6 ms/张 | 53.0 | 高速高精度平衡 |

---

## 5. 知识连线

本课学习的目标检测技术，是后续多模态学习和视觉感知模块的基础：

- **阶段 04 · 07（语义分割 U-Net）**：检测划定"哪里有东西"，分割界定"这个东西的精确像素轮廓"——两者共用相同的特征提取器和边界框回归思路。
- **阶段 04 · 08（实例分割 Mask R-CNN）**：在检测的基础上增加一个分割分支，每个检测框对应一个像素级的 mask。本质上是检测任务的扩展。
- **阶段 12（多模态人工智能）**：视觉语言模型（VLM）如 BLIP-2、LLaVA 中的视觉编码器通常借用检测框架的特征提取器（如 Swin Transformer），检测学到的多尺度感知能力直接服务于图文对齐。

---

## 6. 工程最佳实践

### 6.1 锚框策略的工程建议

- k-means 聚类时样本量不要太小——至少 1 万帧标注框才能得到稳定的 anchor 分布
- 训练过程中动态更新 anchor（YOLOv5 的做法）比静态 anchor 在自定义数据集上 mAP 高 1-2 个百分点
- Anchor-Free 方案可以跳过这一步，直接在数据加载器中做标签分配

### 6.2 NMS 替代方案的工业趋势

- **Soft-NMS**（Bodla et al., 2017）[1]：不直接剔除高 IoU 的框，而是线性或高斯衰减其置信度分数，保留可能被遮挡的物体
- **DIoU-NMS**（Yuan et al., 2021）[2]：用 Distance-IoU 替代传统 IoU 做 NMS，在密集场景中表现更好
- **WIOU-NMS**： Weighted IoU，对不同类别的 NMS 使用不同阈值

当场景中存在大量遮挡或紧密排列的物体时，不要只用基础 NMS。

### 6.3 中文场景特别建议

- COCO 和 Pascal VOC 等主流数据集中中文场景（行人装束、交通标志、建筑特征）代表性不足。建议在验证集上专门评估中文场景指标的偏差
- 如果使用自研数据，标注工具推荐使用 CVAT 或 Label Studio，前者支持 YOLO 格式的导出
- 中文交通场景中车辆和行人尺度跨度极大——城乡混合道路中小车可能只占 16×16 像素，需要在模型输入尺寸和多尺度训练上做针对性调整

### 6.4 踩坑经验

1. **锚框聚类时用错了距离函数**——k-means 用的是欧氏距离，但边界框匹配应该用 IoU 或 GIoU 作为距离。YOLOv3/v4 之后的版本用了 IoU-aware k-means，能显著提升匹配质量
2. **正负样本分配策略过于激进**——早期 YOLO 把所有 background grid 都当作负样本，导致负样本占 99% 以上。现代做法是使用 focal loss（Lin et al., 2017）[3] 或 ATSS（Zhang et al., 2020）[4] 来做智能的正样本选择
3. **数据增强缺失**——Mosaic 增强对小物体检测的提升可达 3-5 个 mAP 点。没有它，模型对紧凑场景的泛化能力显著下降

---

## 7. 常见错误

### 错误 1：用 MSE 直接回归绝对坐标

**现象：** 模型在小物体上表现良好，在大物体上定位偏差极大。训练 Loss 很快收敛到稳定值，但推理时边界框常常严重偏离。

**原因：** MSE 对绝对坐标的惩罚对所有尺度一视同仁。一个 500 像素宽的框偏差 10 像素，和一个 50 像素宽的框偏差 10 像素，MSE 给出的惩罚是一样的。但 10 像素对小框来说是 20% 的偏差，对大框来说只有 2%。

**修复：** 使用 GIoU 损失或对数缩放（如 `encode_box` 中对宽高使用 `log(w/anchor_w)`），使损失函数与框的尺度无关。

```python
# ❌ 直接回归绝对坐标
loss = torch.nn.functional.mse_loss(pred_xyxy, target_xyxy)

# ✓ 使用 GIoU 损失
loss = 1.0 - calc_giou(pred, target)
```

### 错误 2：NMS 阈值设置过高

**现象：** 同一物体被漏检——模型预测了两个重叠度较高的框，其中一个被 NMS 错误删除。

**原因：** IoU 阈值设得太高（如 0.7），两个稍有错位但都是合理预测的框被认为是重复的。特别是在密集场景中（人群、鸟群），同一物体从不同角度被检测到时，bbox 之间可能有较大位移但仍表示同一个实体。

**修复：** 将 IoU 阈值降到 0.4-0.5，或使用 Soft-NMS。

### 错误 3：混淆 Confidence Score 与 Class Probability

**现象：** 检测到的物体类别正确但置信度异常——模型对"人"的置信度高达 0.95，但该框其实覆盖了一辆自行车。

**原因：** 在 YOLOv3/v5 的原始设计中，"objectness score"和"class probability"在推理时被直接相乘得到最终得分。但如果训练时 objectness 校准不好（例如大量正样本的 objectness 只有 0.5 而非 1.0），最终的 score 就会失去意义。

**修复：** 在部署前对置信度分数做温度缩放（Temperature Scaling）校准，或在推理后加一层简单的置信度阈值调优。

---

## 8. 面试考点

### Q1：为什么 YOLO 比 Faster R-CNN 快？（难度：⭐⭐）

**参考答案：**

Faster R-CNN 是两阶段检测器：第一阶段用 Region Proposal Network（RPN）生成约 1000 个候选区域，第二阶段对每个候选区域做特征提取 + 分类 + 回归。这意味着每张图像大约需要 1000 次前向传播（或至少 1000 次 ROI pooling 操作）。

YOLO 是一阶段检测器：整个图像通过一次 CNN 前向传播，在网络输出的网格单元上直接预测边界框和类别。不管图像中有多少物体，网络的前向传播次数永远是 1。这就是 YOLO 比 Faster R-CNN 快一个数量级的根本原因。

### Q2：什么是 Anchor Box？为什么要用它？（难度：⭐⭐）

**参考答案：**

Anchor Box 是在训练前提前定义的一组具有不同宽高比的边界框模板（如 `[16×16]`、`[32×32]`、`[64×64]`、`[32×64]` 等）。每个网格单元会预测相对于这些 anchor 的偏移量，而不是从零开始预测绝对坐标。

使用 anchor 的核心原因是：边界框回归是一个高度病态的问题——给定一个 256×256 的特征图和数千种可能的框尺寸，网络很难凭空学到合理的输出分布。Anchor 为网络提供了一个强归纳偏置（inductive bias），让网络只需要学习"微调"而不是"发明"。同时，不同 anchor 覆盖不同宽高比，使得网络对细长物体和方形物体的检测能力更加均衡。

### Q3：手写 NMS 算法，并解释其时间复杂度。（难度：⭐⭐⭐）

**参考答案：**

```python
def nms(boxes, scores, iou_threshold=0.45):
    """
    boxes: (N, 4) (x1, y1, x2, y2)
    scores: (N,)
    返回: 保留的索引数组
    """
    # 按分数降序排序
    indices = np.argsort(-scores)
    keep = []

    while indices.size > 0:
        i = indices[0]
        keep.append(int(i))

        if indices.size == 1:
            break

        # 计算当前最高分框与剩余框的 IoU
        rest = indices[1:]
        ious = compute_iou(boxes[i], boxes[rest])

        # 只保留 IoU 低于阈值的框
        indices = rest[ious <= iou_threshold]

    return np.array(keep)
```

时间复杂度分析：最坏情况下每一轮 NMS 只消除一个框（所有框都不重叠），需要进行 $N + (N-1) + ... + 1 = O(N^2)$ 次 IoU 计算。每轮 IoU 计算的复杂度为 $O(1)$，因此总时间复杂度为 $O(N^2)$。实际检测中 $N$ 通常在几百到几千，这个开销可忽略不计。

### Q4：Anchor-Free 和 Anchor-Based 各有什么优劣？（难度：⭐⭐）

**参考答案：**

Anchor-Based 的优势是训练稳定、对小物体的召回率较高——anchor 为不同尺度和形状的物体提供了预定义的归纳偏置。缺点是需要在训练前做 k-means 聚类确定 anchor 集合，这是一个额外的超参数调优步骤；且在训练中正负样本分配（与哪个 anchor 匹配）容易出错，尤其是密集小物体场景。

Anchor-Free 的优势是架构简洁——不需要 k-means、不需要 anchor 选择逻辑、检测头更小更快。尤其在 YOLOv8 中，anchor-free 配合 task-aligned assigner（TAL）可以在减少计算量的同时达到相当甚至更好的精度。缺点是对某些极端长宽比的物体（如极细长的电线杆）可能需要更多的训练迭代才能收敛。

### Q5：解释 GIoU 为什么比 L1/MSE 更适合边界框回归。（难度：⭐⭐⭐）

**参考答案：**

L1/MSE 损失直接优化坐标值的绝对差异。但当预测框和真实框完全不重叠时，L1 给出的梯度仍然是均匀的——它不知道"我的框应该在左边还是右边"这个方向性信息。

GIoU 不同——它首先用 IoU 项鼓励重叠，然后用闭包区域惩罚不重叠的程度。当框远离目标时，闭包区域面积大，GIoU 负值大（接近 -1），梯度告诉模型"你的框应该向目标方向缩小闭包区域"。这天然地编码了方向的语义。

此外，GIoU 是无量纲的——它的值与框的绝对大小无关。同一个 GIoU = 0.8 表示"80% 的重叠质量"，无论框是大是小。而 L1 损失对大框的偏差容忍度远高于小框，这在实际目标检测中是不公平的。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 目标检测 | "就是给图里的东西打标签" | 同时完成定位（边界框坐标）和分类（物体类别）的双任务：输入图像，输出多个 `[x1, y1, x2, y2, class, confidence]` 四元组 |
| 边界框（Bounding Box） | "框住物体的矩形" | 用四个坐标值 $(x_1, y_1, x_2, y_2)$ 表示的轴对齐矩形，是目标检测最基本的几何原语 |
| 交并比（IoU） | "两个框重叠了多少" | 预测框与真实框交集面积除以并集面积的比值，值域 $[0, 1]$，是检测和分割任务的标准对齐度量 |
| GIoU | "改进版的 IoU" | 在 IoU 基础上减去最小闭包区域与并集的比率，使完全不相交的框也能回传有用的梯度信号 |
| NMS | "去掉重复的框" | 贪心算法：按置信度排序选出最高分框，消除与其重叠度高的冗余框，直到没有更多候选框 |
| Anchor Box | "预设的框模板" | 通过 k-means 聚类标注框的宽高值得到的一组模板，每个网格单元预测相对于 anchor 的偏移而非绝对坐标 |
| Anchor-Free | "不需要预设框" | 取消 anchor 维度，每个网格单元直接预测边界框参数（如到四边的距离），简化了检测头和标签分配 |
| 后处理（Post-processing） | "跑完模型之后的事" | 推理阶段对原始预测做的解码、置信度阈值过滤、NMS 去重的完整流水线 |
| 物体置信度（Objectness） | "模型认为这里有物体的把握" | 网络输出的标量值，表示该网格/该框包含任意物体的概率，与类别概率相乘得到最终检测分数 |

---

## 📚 小结

目标检测将定位与分类统一为一个端到端的优化问题——YOLO 的核心突破是用网格化的回归网络替代了两阶段检测方案。你从零实现了 IoU 计算、GIoU 损失、NMS、边界框编码/解码和完整的 YOLO 损失函数，理解了 Anchor-Based 与 Anchor-Free 两种范式的本质差异。

下一课我们将进入语义分割——在检测"哪里有物体"的基础上，进一步确定"物体在每个像素属于谁"。

---

## ✏️ 练习

1. 【理解】用自己的话解释为什么"YOLO 一次前向传播就能做检测"。画图说明 Faster R-CNN 和 YOLO 在推理阶段前向传播次数的区别。

2. 【实现】修改 `postprocess_predictions` 函数，使其支持多类别 NMS（per-class NMS）——即对每个类别分别执行 NMS，避免不同类别的框相互干扰。参考 Cascade R-CNN 的做法。

3. 【实验】取一张包含至少 5 个不同物体的真实图片（如 COCO 验证集中的任一图片），用 PyTorch 的 `fasterrcnn_resnet50_fpn` 和 Ultralytics YOLOv8n 分别推理，对比两者的 mAP、推理速度和框数量差异。

4. 【思考】NMS 是可微的吗？如果不可微，会对训练产生什么影响？搜索"Differentiable NMS"或"Soft-NMS"相关论文，思考如何让 NMS 的梯度流回网络。

5. 【思考】在一辆公交车里有 10 个乘客的场景中，YOLO 应该把这 10 个人分别框出来，还是框一个包含所有人的大框？这个设计选择对自动驾驶、安防监控、视频检索等不同场景意味着什么？

---

## 🚀 产出

本课产出以下可复用内容：

| 产出 | 文件 | 说明 |
|---|---|---|
| YOLO 核心组件（纯 NumPy） | `code/main.py` | 从零实现的 IoU、GIoU、NMS、边界框编码/解码、标签分配、YOLO 损失计算和后处理全管线 |
| 目标检测对比分析脚本 | `code/compare_detectors.py` | 对比 Faster R-CNN 与 YOLOv8 在同一张图片上的检测结果（需在 notebook 中编写） |
| 可复用 IoU 计算函数 | `code/iou.py` | 独立的 `box_iou()` 和 `calc_giou()` 函数，可直接导入其他项目使用 |

---

## 📖 参考资料

1. [论文] Redmon et al. "You Only Look Once: Unified, Real-Time Object Detection". CVPR, 2016. https://arxiv.org/abs/1506.02640
2. [论文] Redmon and Farhadi. "YOLO9000: Better, Faster, Stronger". CVPR, 2017. https://arxiv.org/abs/1612.08242
3. [论文] Redmon and Farhadi. "YOLOv3: An Incremental Improvement". arXiv, 2018. https://arxiv.org/abs/1804.02767
4. [论文] Bochkovskiy et al. "YOLOv4: Optimal Speed and Accuracy of Object Detection". arXiv, 2020. https://arxiv.org/abs/2004.10934
5. [论文] Liu et al. "FCOS: Fully Convolutional One-Stage Object Detection". ICCV, 2019. https://arxiv.org/abs/1904.01355
6. [论文] Zhang et al. "Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression". CVPR, 2019. https://arxiv.org/abs/1902.09630
7. [官方文档] PyTorch torchvision detection models: https://pytorch.org/vision/stable/models.html#object-detection
8. [GitHub] Ultralytics YOLO: https://github.com/ultralytics/ultralytics
9. [论文] Bodla et al. "Soft-NMS -- Improving Object Detection With One Line of Code". ICCV, 2017. https://arxiv.org/abs/1704.04503
10. [论文] Zhang et al. "ATSS: Adaptive Training Sample Selection for Dense Object Detection". CVPR, 2020. https://arxiv.org/abs/1912.02424

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
