# === main.py — 多目标跟踪从零实现 ===
# 依赖：numpy>=1.24, scipy>=1.10
# 对应课程：阶段 04 · 27（多目标跟踪）
#
# 本文件实现：
#   1. IoU 计算矩阵
#   2. 卡尔曼滤波器（匀速模型）
#   3. SORT 风格跟踪器（卡尔曼预测 + IoU 匈牙利匹配）
#   4. ByteTrack 风格的二次匹配（利用低置信度检测）
#   5. ID 切换计数与 MOTA 指标计算
#   6. 合成轨迹测试 + 可视化

import numpy as np
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# 第 1 步：交并比（IoU）计算
# ==============================================================================

def bbox_iou(boxes_a, boxes_b):
    """计算两组边界框之间的 IoU 矩阵。

    Args:
        boxes_a: (N, 4) 数组，每行 [x1, y1, x2, y2]
        boxes_b: (M, 4) 数组，每行 [x1, y1, x2, y2]

    Returns:
        (N, M) IoU 矩阵，IoU[i, j] = boxes_a[i] 与 boxes_b[j] 的交并比
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    # 提取坐标
    ax1, ay1 = boxes_a[:, 0], boxes_a[:, 1]
    ax2, ay2 = boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1 = boxes_b[:, 0], boxes_b[:, 1]
    bx2, by2 = boxes_b[:, 2], boxes_b[:, 3]

    # 计算交集区域的边界
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    # 交集面积（clip 防止负值）
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)

    # 各自面积
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    # 并集面积 = 面积A + 面积B - 交集面积
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / np.clip(union_area, 1e-8, None)


# ==============================================================================
# 第 2 步：卡尔曼滤波器（匀速运动模型）
# ==============================================================================

class KalmanFilter:
    """常速模型卡尔曼滤波器，用于预测边界框的下一帧位置。

    状态向量: [x, y, w, h, dx, dy, dw, dh]
      - (x, y) = 边界框中心坐标
      - (w, h) = 边界框宽高
      - (dx, dy, dw, dh) = 对应的速度

    匀速模型假设：下一帧位置 = 当前位置 + 速度
    """

    def __init__(self):
        # 状态维度 8（位置 4 + 速度 4），观测维度 4（仅位置）
        self.dim_state = 8
        self.dim_obs = 4

        # 状态转移矩阵 F（匀速模型：p' = p + v）
        self.F = np.eye(self.dim_state)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # 位置 += 速度

        # 观测矩阵 H（只观测位置，不观测速度）
        self.H = np.zeros((self.dim_obs, self.dim_state))
        self.H[:4, :4] = np.eye(self.dim_obs)

        # 过程噪声协方差（匀速模型的假设误差）
        self.Q = np.eye(self.dim_state) * 1.0
        # 观测噪声协方差（检测器定位误差）
        self.R = np.eye(self.dim_obs) * 10.0

        self.x = np.zeros(self.dim_state)     # 状态向量
        self.P = np.eye(self.dim_state) * 100  # 初始协方差（不确定性大）

    def init_state(self, bbox):
        """用第一个检测结果初始化状态。"""
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        # 速度初始化为 0
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        self.P = np.eye(self.dim_state) * 10.0

    def predict(self):
        """预测下一帧的状态。

        Returns:
            预测的边界框 [x1, y1, x2, y2]
        """
        # 状态预测：x' = F @ x
        self.x = self.F @ self.x
        # 协方差预测：P' = F @ P @ F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self._state_to_bbox()

    def update(self, bbox):
        """用检测结果更新状态（卡尔曼更新步骤）。

        Args:
            bbox: 检测到的边界框 [x1, y1, x2, y2]
        """
        # 将 bbox 转为观测向量
        x1, y1, x2, y2 = bbox
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

        # 卡尔曼增益：K = P @ H^T @ (H @ P @ H^T + R)^{-1}
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 残差（观测值 - 预测值）
        y = z - self.H @ self.x

        # 状态更新
        self.x = self.x + K @ y
        # 协方差更新（Joseph 形式，数值更稳定）
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ self.P

    def _state_to_bbox(self):
        """将状态向量转换为边界框 [x1, y1, x2, y2]。"""
        cx, cy, w, h = self.x[:4]
        # 保证宽高非负
        w = max(w, 1.0)
        h = max(h, 1.0)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ==============================================================================
# 第 3 步：跟踪轨迹管理
# ==============================================================================

class Track:
    """单条跟踪轨迹，包含唯一 ID、卡尔曼滤波器、存活信息。"""

    def __init__(self, track_id, bbox, frame_id):
        self.id = track_id
        self.age = 0           # 轨迹已存在帧数
        self.hits = 1          # 被匹配的总次数
        self.time_since_update = 0  # 距上次更新的帧数

        # 为该轨迹初始化一个卡尔曼滤波器
        self.kalman = KalmanFilter()
        self.kalman.init_state(bbox)

    def predict(self):
        """预测下一帧的位置。"""
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict()

    def update(self, bbox, frame_id):
        """用匹配到的检测结果更新轨迹。"""
        self.kalman.update(bbox)
        self.hits += 1
        self.time_since_update = 0


# ==============================================================================
# 第 4 步：SORT 风格跟踪器
# ==============================================================================

class SortTracker:
    """SORT（Simple Online and Realtime Tracking）风格跟踪器。

    流程：
      1. 对所有已有轨迹执行卡尔曼预测
      2. 计算预测框与检测框的 IoU 代价矩阵
      3. 匈牙利算法求解最优匹配
      4. 匹配成功的轨迹用检测结果更新
      5. 未匹配的检测创建新轨迹
      6. 超过 max_age 未更新的轨迹被删除
    """

    def __init__(self, iou_threshold=0.3, max_age=5):
        """
        Args:
            iou_threshold: IoU 低于此阈值的匹配视为无效
            max_age: 轨迹连续未匹配的最大帧数，超过则删除
        """
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame_id):
        """处理一帧的跟踪。

        Args:
            detections: 当前帧的检测框列表，每个元素为 [x1, y1, x2, y2]
            frame_id: 当前帧编号（从 0 开始）

        Returns:
            [(track_id, [x1, y1, x2, y2]), ...] 当前帧所有活跃轨迹
        """
        # === 第一步：对所有已有轨迹做卡尔曼预测 ===
        for track in self.tracks:
            track.predict()

        # === 第二步：构建代价矩阵 ===
        dets = np.array(detections, dtype=np.float64) if len(detections) > 0 else np.empty((0, 4))

        if len(self.tracks) == 0 or len(dets) == 0:
            # 没有已有轨迹或没有检测：新检测直接创建轨迹
            for d in dets:
                self.tracks.append(Track(self.next_id, d, frame_id))
                self.next_id += 1
            return self._get_outputs()

        # 获取所有轨迹的预测框
        track_pred_boxes = np.array([t.kalman._state_to_bbox() for t in self.tracks])

        # 计算 IoU 代价矩阵：代价 = 1 - IoU
        iou_matrix = bbox_iou(track_pred_boxes, dets)
        cost_matrix = 1.0 - iou_matrix

        # IoU 低于阈值的匹配代价设为极大值（不允许匹配）
        cost_matrix[iou_matrix < self.iou_threshold] = 1e6

        # === 第三步：匈牙利算法求解最优匹配 ===
        matched_tracks = set()
        matched_dets = set()

        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] < 1.0:
                    # 匹配成功：用检测结果更新轨迹
                    self.tracks[r].update(dets[c], frame_id)
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # === 第四步：未匹配的检测创建新轨迹 ===
        for i, det in enumerate(dets):
            if i not in matched_dets:
                self.tracks.append(Track(self.next_id, det, frame_id))
                self.next_id += 1

        # === 第五步：删除过期轨迹 ===
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self._get_outputs()

    def _get_outputs(self):
        """返回所有活跃轨迹的 ID 和预测框。"""
        results = []
        for track in self.tracks:
            bbox = track.kalman._state_to_bbox()
            results.append((track.id, bbox))
        return results


# ==============================================================================
# 第 5 步：ByteTrack 风格（二次匹配低置信度检测）
# ==============================================================================

class ByteTrackTracker(SortTracker):
    """ByteTrack 风格跟踪器，在 SORT 基础上增加二次匹配。

    核心改进：
      - 第一轮：高置信度检测（>= score_thresh）与轨迹正常匹配
      - 第二轮：未匹配的轨迹尝试与低置信度检测匹配（更宽松的 IoU 阈值）
      - 低置信度检测可能来自被遮挡的物体——不应直接丢弃
    """

    def __init__(self, iou_threshold=0.3, max_age=5, score_thresh=0.5):
        super().__init__(iou_threshold=iou_threshold, max_age=max_age)
        self.score_thresh = score_thresh

    def step(self, detections_scores, frame_id):
        """处理一帧（输入为检测框 + 置信度）。

        Args:
            detections_scores: [(bbox, score), ...] 检测框与置信度
            frame_id: 当前帧编号

        Returns:
            [(track_id, [x1, y1, x2, y2]), ...] 当前帧所有活跃轨迹
        """
        # 按置信度分为高、低两组
        high_dets = []
        low_dets = []
        for bbox, score in detections_scores:
            if score >= self.score_thresh:
                high_dets.append(bbox)
            else:
                low_dets.append(bbox)

        # 第一轮：标准 SORT 匹配（高置信度检测）
        self.tracks, matched_track_ids = self._match_phase(
            high_dets, frame_id, iou_thresh=self.iou_threshold
        )

        # 第二轮：未匹配的轨迹尝试与低置信度检测匹配
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_track_ids]
        if unmatched_tracks and low_dets:
            # 低置信度检测使用更宽松的 IoU 阈值
            self._match_second_phase(unmatched_tracks, low_dets, frame_id)

        # 低置信度检测中未匹配的不创建新轨迹（可能不可靠）
        # 删除过期轨迹
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self._get_outputs()

    def _match_phase(self, detections, frame_id, iou_thresh):
        """第一轮匹配（高置信度检测）。"""
        matched_tracks = set()
        matched_dets = set()
        dets = np.array(detections, dtype=np.float64) if detections else np.empty((0, 4))

        if len(self.tracks) == 0:
            for d in dets:
                self.tracks.append(Track(self.next_id, d, frame_id))
                self.next_id += 1
            return self.tracks, matched_tracks

        for track in self.tracks:
            track.predict()

        track_pred_boxes = np.array([t.kalman._state_to_bbox() for t in self.tracks])

        if len(dets) > 0:
            iou_matrix = bbox_iou(track_pred_boxes, dets)
            cost_matrix = 1.0 - iou_matrix
            cost_matrix[iou_matrix < iou_thresh] = 1e6

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] < 1.0:
                    self.tracks[r].update(dets[c], frame_id)
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # 未匹配的高置信度检测创建新轨迹
        for i, det in enumerate(dets):
            if i not in matched_dets:
                self.tracks.append(Track(self.next_id, det, frame_id))
                self.next_id += 1

        return self.tracks, matched_tracks

    def _match_second_phase(self, unmatched_track_indices, low_detections, frame_id):
        """第二轮匹配（低置信度检测，更宽松的 IoU 阈值）。"""
        low_dets = np.array(low_detections, dtype=np.float64)
        low_iou_thresh = self.iou_threshold * 0.5  # 放宽 50%

        track_pred_boxes = np.array([
            self.tracks[i].kalman._state_to_bbox() for i in unmatched_track_indices
        ])

        iou_matrix = bbox_iou(track_pred_boxes, low_dets)
        cost_matrix = 1.0 - iou_matrix
        cost_matrix[iou_matrix < low_iou_thresh] = 1e6

        matched_local = set()
        matched_low = set()

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < 1.0:
                global_idx = unmatched_track_indices[r]
                self.tracks[global_idx].update(low_dets[c], frame_id)
                matched_local.add(r)
                matched_low.add(c)


# ==============================================================================
# 第 6 步：评估指标 — ID 切换计数 & MOTA
# ==============================================================================

def count_id_switches(tracks_per_frame, gt_per_frame):
    """计算 ID 切换次数。

    对每一帧的跟踪结果与真实标签做 IoU 匹配，统计 ID 变化次数。

    Args:
        tracks_per_frame: 每帧的跟踪结果列表，每个元素为 [(track_id, bbox), ...]
        gt_per_frame: 每帧的真实标签列表，每个元素为 [(gt_id, bbox), ...]

    Returns:
        ID 切换总次数
    """
    prev_assignment = {}  # gt_id -> 当前匹配的 track_id
    switches = 0

    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue

        t_boxes = np.array([bbox for _, bbox in tracks])
        g_boxes = np.array([bbox for _, bbox in gts])
        iou_matrix = bbox_iou(g_boxes, t_boxes)

        for gt_idx, (gt_id, _) in enumerate(gts):
            best_j = int(iou_matrix[gt_idx].argmax())
            if iou_matrix[gt_idx, best_j] > 0.5:
                current_track_id = tracks[best_j][0]
                # 如果这个真实物体之前有匹配，且匹配结果发生了变化
                if gt_id in prev_assignment and prev_assignment[gt_id] != current_track_id:
                    switches += 1
                prev_assignment[gt_id] = current_track_id

    return switches


def compute_mota(tracks_per_frame, gt_per_frame, num_gt_objects):
    """计算 MOTA（Multi-Object Tracking Accuracy）。

    MOTA = 1 - (FN + FP + ID_switches) / 总GT数

    Args:
        tracks_per_frame: 每帧的跟踪结果
        gt_per_frame: 每帧的真实标签
        num_gt_objects: 真实目标总数

    Returns:
        MOTA 值（越接近 1 越好）
    """
    total_fn = 0  # 漏检（False Negatives）
    total_fp = 0  # 误检（False Positives）
    id_switches = count_id_switches(tracks_per_frame, gt_per_frame)

    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks and not gts:
            continue

        t_boxes = np.array([bbox for _, bbox in tracks]) if tracks else np.empty((0, 4))
        g_boxes = np.array([bbox for _, bbox in gts]) if gts else np.empty((0, 4))

        if len(gts) > 0 and len(tracks) > 0:
            iou_matrix = bbox_iou(g_boxes, t_boxes)
            # 计算未匹配的真实标签（漏检）和未匹配的检测（误检）
            matched_gts = set()
            matched_tracks = set()
            row_indices, col_indices = linear_sum_assignment(1.0 - iou_matrix)
            for r, c in zip(row_indices, col_indices):
                if iou_matrix[r, c] > 0.5:
                    matched_gts.add(r)
                    matched_tracks.add(c)
            total_fn += len(gts) - len(matched_gts)
            total_fp += len(tracks) - len(matched_tracks)
        elif len(gts) > 0:
            total_fn += len(gts)
        elif len(tracks) > 0:
            total_fp += len(tracks)

    if num_gt_objects == 0:
        return 1.0
    mota = 1.0 - (total_fn + total_fp + id_switches) / num_gt_objects
    return mota


# ==============================================================================
# 第 7 步：合成数据生成
# ==============================================================================

def synthetic_frames(num_frames=25, num_objects=3, H=240, W=320, seed=0, drop_prob=0.0):
    """生成合成的目标运动轨迹。

    每个目标做匀速直线运动，可选地随机丢失某些帧的检测。

    Args:
        num_frames: 帧数
        num_objects: 目标数量
        H, W: 画面尺寸
        seed: 随机种子
        drop_prob: 检测丢失概率（模拟遮挡/漏检）

    Returns:
        frames: 每帧的检测列表（可能有丢失）
        gt: 每帧的真实标签列表
    """
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-4, 4, size=(num_objects, 2))

    frames = []
    gt = []

    for f in range(num_objects):
        pass  # 占位，下面才是真正的循环

    frames = []
    gt = []
    for f in range(num_frames):
        frame_dets = []
        frame_gt = []
        for i in range(num_objects):
            cx = starts[i, 0] + f * velocities[i, 0]
            cy = starts[i, 1] + f * velocities[i, 1]
            x1 = max(0.0, cx - 10)
            y1 = max(0.0, cy - 10)
            x2 = min(float(W - 1), cx + 10)
            y2 = min(float(H - 1), cy + 10)
            bbox = [x1, y1, x2, y2]
            frame_gt.append((i, bbox))
            # 按概率丢失检测（模拟遮挡）
            if rng.random() >= drop_prob:
                frame_dets.append(bbox)
        frames.append(frame_dets)
        gt.append(frame_gt)

    return frames, gt


# ==============================================================================
# 第 8 步：可视化轨迹（ASCII）
# ==============================================================================

def print_frame_state(frame_id, tracks, gt_frame, W=320):
    """用 ASCII 字符在终端可视化一帧的跟踪结果。"""
    canvas = ["."] * W

    # 用数字显示真实标签
    for gt_id, bbox in gt_frame:
        cx = int((bbox[0] + bbox[2]) / 2)
        cx = max(0, min(W - 1, cx))
        canvas[cx] = str(gt_id % 10)

    # 用字母显示跟踪结果
    for track_id, bbox in tracks:
        cx = int((bbox[0] + bbox[2]) / 2)
        cx = max(0, min(W - 1, cx))
        if canvas[cx] == ".":
            canvas[cx] = chr(ord("A") + (track_id - 1) % 26)
        else:
            canvas[cx] = "*"  # 匹配成功

    print(f"  帧 {frame_id:>2d}: {''.join(canvas)}  (数字=GT, 字母=跟踪)")


# ==============================================================================
# 主程序：运行所有测试
# ==============================================================================

def main():
    print("=" * 60)
    print("多目标跟踪从零实现 — 测试")
    print("=" * 60)

    # --- 测试 1：基础 SORT 跟踪器 ---
    print("\n--- 测试 1：SORT 跟踪器（3 个目标） ---")
    tracker = SortTracker(iou_threshold=0.3, max_age=5)
    frames, gt = synthetic_frames(num_frames=20, num_objects=3, seed=42)
    tracks_per_frame = []
    for f, dets in enumerate(frames):
        tracks = tracker.step(dets, f)
        tracks_per_frame.append(tracks)
        if f < 5:
            print_frame_state(f, tracks, gt[f])

    switches = count_id_switches(tracks_per_frame, gt)
    num_gt = sum(len(g) for g in gt)
    mota = compute_mota(tracks_per_frame, gt, num_gt)
    print(f"\n  结果: 存活轨迹={len(tracker.tracks)}, ID切换={switches}, MOTA={mota:.3f}")

    # --- 测试 2：不同目标数量下的 ID 切换 ---
    print("\n--- 测试 2：不同目标数量对比 ---")
    for n_obj in [3, 10, 30]:
        tracker = SortTracker(iou_threshold=0.3, max_age=5)
        frames, gt = synthetic_frames(num_frames=25, num_objects=n_obj, seed=0)
        tracks_per_frame = []
        for f, dets in enumerate(frames):
            tracks = tracker.step(dets, f)
            tracks_per_frame.append(tracks)
        switches = count_id_switches(tracks_per_frame, gt)
        num_gt = sum(len(g) for g in gt)
        mota = compute_mota(tracks_per_frame, gt, num_gt)
        print(f"  {n_obj:>3d} 个目标: 存活轨迹={len(tracker.tracks):>3d}, "
              f"ID切换={switches:>3d}, MOTA={mota:.3f}")

    # --- 测试 3：模拟遮挡（检测丢失） ---
    print("\n--- 测试 3：模拟遮挡（20% 检测丢失率） ---")
    for n_obj in [3, 10]:
        tracker = SortTracker(iou_threshold=0.3, max_age=3)
        frames, gt = synthetic_frames(num_frames=25, num_objects=n_obj, drop_prob=0.2, seed=0)
        tracks_per_frame = []
        for f, dets in enumerate(frames):
            tracks = tracker.step(dets, f)
            tracks_per_frame.append(tracks)
        switches = count_id_switches(tracks_per_frame, gt)
        print(f"  {n_obj:>3d} 个目标 + 20%丢失: ID切换={switches:>3d}")

    # --- 测试 4：ByteTrack 二次匹配 ---
    print("\n--- 测试 4：ByteTrack 风格二次匹配 ---")
    rng = np.random.default_rng(42)
    n_obj = 5
    # 为每个检测附带置信度
    frames_bbox, gt = synthetic_frames(num_frames=20, num_objects=n_obj, drop_prob=0.15, seed=0)
    frames_with_score = []
    for frame_dets in frames_bbox:
        scored = []
        for det in frame_dets:
            # 大部分高置信度，少部分低置信度
            score = 0.9 if rng.random() > 0.2 else 0.3
            scored.append((det, score))
        frames_with_score.append(scored)

    tracker = ByteTrackTracker(iou_threshold=0.3, max_age=5, score_thresh=0.5)
    tracks_per_frame = []
    for f, det_scores in enumerate(frames_with_score):
        tracks = tracker.step(det_scores, f)
        tracks_per_frame.append(tracks)
    switches = count_id_switches(tracks_per_frame, gt)
    print(f"  {n_obj} 个目标 + ByteTrack: ID切换={switches}")

    # --- 测试 5：卡尔曼滤波预测可视化 ---
    print("\n--- 测试 5：卡尔曼滤波预测演示 ---")
    kf = KalmanFilter()
    kf.init_state([100, 100, 20, 20])
    print("  初始状态:", [f"{v:.1f}" for v in kf._state_to_bbox()])
    for i in range(5):
        pred = kf.predict()
        # 模拟真实检测（加一点噪声）
        noise = np.random.randn(4) * 2.0
        noisy_obs = [pred[0] + noise[0], pred[1] + noise[1],
                      pred[2] + noise[2], pred[3] + noise[3]]
        kf.update(noisy_obs)
        print(f"  第 {i + 1} 帧: 预测={[f'{v:.1f}' for v in pred]}, "
              f"更新后={[f'{v:.1f}' for v in kf._state_to_bbox()]}")

    print("\n" + "=" * 60)
    print("全部测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
