# loss_functions.py — 从零实现常用损失函数
# 依赖：Python 3.10+, 标准库（无需第三方库）
# 对应课程：阶段 03 · 05（损失函数）

import math
import random


# =============================================================================
# 1. 回归损失：MSE / MAE / Huber Loss
# =============================================================================

def mse(predictions, targets):
    """均方误差（MSE）：回归问题的默认损失函数。

    对大误差给予二次惩罚，对异常值敏感。

    Args:
        predictions: 模型预测值列表
        targets: 真实目标值列表

    Returns:
        均方误差标量
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        total += (p - t) ** 2
    return total / n


def mse_gradient(predictions, targets):
    """MSE 对预测值的梯度。

    梯度 = 2 * (预测值 - 目标值) / n，线性于误差。
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    return [2.0 * (p - t) / n for p, t in zip(predictions, targets)]


def mae(predictions, targets):
    """平均绝对误差（MAE）：对异常值更鲁棒的回归损失。

    梯度大小恒为 1，不受误差量级影响。
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        total += abs(p - t)
    return total / n


def mae_gradient(predictions, targets):
    """MAE 对预测值的梯度（符号函数）。"""
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    return [math.copysign(1.0, p - t) / n for p, t in zip(predictions, targets)]


def huber_loss(predictions, targets, delta=1.0):
    """Huber 损失：小误差用 MSE，大误差用 MAE。

    结合了 MSE 在小误差处的平滑性和 MAE 在大误差处的鲁棒性。
    delta 是切换两种行为的阈值。

    Args:
        delta: MSE 与 MAE 的切换阈值，默认 1.0
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        error = abs(p - t)
        if error <= delta:
            total += 0.5 * error ** 2
        else:
            total += delta * (error - 0.5 * delta)
    return total / n


def huber_gradient(predictions, targets, delta=1.0):
    """Huber 损失的梯度。

    小误差处为线性（同 MSE），大误差处为常数（同 MAE）。
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        diff = p - t
        if abs(diff) <= delta:
            grads.append(diff / n)
        else:
            grads.append(delta * math.copysign(1.0, diff) / n)
    return grads


# =============================================================================
# 2. 分类损失：二元交叉熵 / 多元交叉熵
# =============================================================================

def binary_cross_entropy(predictions, targets, eps=1e-15):
    """二元交叉熵（BCE）：二分类问题的标准损失函数。

    采用 -log(p) 惩罚错误预测，对"自信的错误"给予极大惩罚。
    eps 裁剪防止 log(0)。

    Args:
        eps: 数值稳定性裁剪值，防止 log(0) = -inf
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        total += -(t * math.log(p_clipped) + (1 - t) * math.log(1 - p_clipped))
    return total / n


def bce_gradient(predictions, targets, eps=1e-15):
    """BCE 对预测值的梯度。

    当 y=1 且 p 接近 0 时，梯度 = -1/p → -∞，产生极强的修正信号。
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        grads.append((-(t / p_clipped) + (1 - t) / (1 - p_clipped)) / n)
    return grads


def softmax(logits):
    """数值稳定的 Softmax 实现。

    先减去最大值防止 exp 溢出，再归一化。
    """
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def categorical_cross_entropy(logits, target_index, eps=1e-15):
    """多元交叉熵（CCE）：多分类问题的标准损失函数。

    Softmax 将 logits 转为概率，再对真实类别的概率取负对数。

    Args:
        logits: 模型原始输出（未归一化的分数）
        target_index: 真实类别的索引
    """
    probs = softmax(logits)
    p = max(eps, probs[target_index])
    return -math.log(p)


def cce_gradient(logits, target_index):
    """Softmax + CCE 的联合梯度。

    形式极其简洁：预测概率 - 1（真实类别）或 预测概率（其他类别）。
    这是 Softmax 和 CCE 天然配对的原因。
    """
    probs = softmax(logits)
    grads = list(probs)
    grads[target_index] -= 1.0
    return grads


def label_smoothed_cce(logits, target_index, num_classes, alpha=0.1, eps=1e-15):
    """标签平滑的多元交叉熵。

    将硬标签 [0,0,1,0] 替换为软标签 [0.025,0.025,0.925,0.025]（alpha=0.1），
    防止模型过度自信，提升泛化能力。

    Args:
        alpha: 平滑系数。0 表示不平滑，0.1 是常用值
    """
    probs = softmax(logits)
    loss = 0.0
    for i in range(num_classes):
        if i == target_index:
            smooth_target = 1.0 - alpha + alpha / num_classes
        else:
            smooth_target = alpha / num_classes
        p = max(eps, probs[i])
        loss += -smooth_target * math.log(p)
    return loss


# =============================================================================
# 3. 处理类别不平衡：Focal Loss
# =============================================================================

def focal_loss_binary(predictions, targets, gamma=2.0, alpha=0.25, eps=1e-15):
    """二元 Focal Loss：降低易分类样本的权重，聚焦于难分类样本。

    由 Lin 等人于 2017 年提出，用于解决目标检测中前景-背景极端不平衡问题。

    Args:
        gamma: 聚焦参数。0 退化为标准 BCE，2 是常用默认值
        alpha: 类别权重平衡因子，用于进一步缓解类别不平衡
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        # p_t 是模型对真实类别的预测概率
        if t == 1:
            p_t = p_clipped
            weight = alpha
        else:
            p_t = 1 - p_clipped
            weight = 1 - alpha
        # (1 - p_t)^gamma 在 p_t 接近 1（易分类）时趋近 0，有效降权
        total += -weight * ((1 - p_t) ** gamma) * math.log(p_t)
    return total / n


def focal_loss_gradient(predictions, targets, gamma=2.0, alpha=0.25, eps=1e-15):
    """Focal Loss 对预测值的梯度（推导结果）。"""
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        if t == 1:
            p_t = p_clipped
            a = alpha
        else:
            p_t = 1 - p_clipped
            a = 1 - alpha
        # 梯度的完整推导（链式法则）
        factor = a * ((1 - p_t) ** gamma)
        log_term = math.log(p_t)
        # d/dp 的解析表达式
        d_factor = factor * (gamma * log_term / (1 - p_t) - 1.0 / p_t)
        if t == 1:
            grads.append(-d_factor / n)
        else:
            grads.append(d_factor / n)
    return grads


# =============================================================================
# 4. 度量学习损失：对比损失 / Hinge Loss / Triplet Loss
# =============================================================================

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度。"""
    assert len(a) == len(b), "向量维度必须相同"
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    """InfoNCE / NT-Xent 对比损失（SimCLR / CLIP 风格）。

    拉近正样本对，推远负样本对。温度参数 tau 控制分布的锐度。
    tau 越小，对难负样本的惩罚越大。

    Args:
        temperature: 温度参数。SimCLR 默认 0.07，CLIP 使用可学习温度
    """
    sim_pos = cosine_similarity(anchor, positive) / temperature
    sim_negs = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    # 减去最大值防止数值溢出（Softmax 的数值稳定技巧）
    max_sim = max(sim_pos, max(sim_negs)) if sim_negs else sim_pos
    exp_pos = math.exp(sim_pos - max_sim)
    exp_negs = [math.exp(s - max_sim) for s in sim_negs]
    total_exp = exp_pos + sum(exp_negs)

    return -math.log(max(1e-15, exp_pos / total_exp))


def hinge_loss(predictions, targets, margin=1.0):
    """Hinge Loss（合页损失）：支持向量机（SVM）的经典损失函数。

    要求正确类别的分数比其他类别至少高出 margin。

    Args:
        margin: 间隔。默认 1.0
    """
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        # t 取值为 +1 或 -1（SVM 格式）
        total += max(0.0, margin - t * p)
    return total / n


def hinge_gradient(predictions, targets, margin=1.0):
    """Hinge Loss 的梯度（次梯度）。"""
    assert len(predictions) == len(targets), "预测值和目标值长度必须相同"
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        if t * p < margin:
            grads.append(-t / n)
        else:
            grads.append(0.0)
    return grads


def triplet_loss(anchor, positive, negative, margin=0.2):
    """三元组损失（Triplet Loss）。

    拉近 anchor 与 positive 的距离，同时推远 anchor 与 negative 的距离，
    直到 negative 比 positive 至少远 margin。

    Args:
        margin: 正负样本对之间要求的最小距离差
    """
    d_pos = math.sqrt(sum((a - p) ** 2 for a, p in zip(anchor, positive)))
    d_neg = math.sqrt(sum((a - n) ** 2 for a, n in zip(anchor, negative)))
    return max(0.0, d_pos - d_neg + margin)


# =============================================================================
# 5. 感知损失（Perceptual Loss）
# =============================================================================

def perceptual_loss(pred_features, target_features):
    """感知损失：在特征空间中计算 MSE，而非像素空间。

    用于图像生成、风格迁移等任务。比较预训练网络（如 VGG）提取的
    高层特征，而非原始像素——更符合人类感知。

    Args:
        pred_features: 模型生成图像的特征向量
        target_features: 目标图像的特征向量
    """
    return mse(pred_features, target_features)


# =============================================================================
# 6. 鸢尾花数据集上的分类对比实验
# =============================================================================

def sigmoid(x):
    """数值稳定的 Sigmoid 函数。"""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_iris_binary_data(n=100, seed=42):
    """生成二分类数据集（模拟鸢尾花问题的简化版）。

    两个特征，两个类别，近似线性可分。
    """
    random.seed(seed)
    data = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        # 两个类别分布在一条直线两侧
        label = 1.0 if x1 + x2 > 0 else 0.0
        data.append(([x1, x2], label))
    return data


class LossComparisonNetwork:
    """一个小型全连接网络，用于对比不同损失函数的训练效果。

    结构：2 输入 → 8 隐藏单元（ReLU）→ 1 输出（Sigmoid）
    """

    def __init__(self, loss_type="bce", hidden_size=8, lr=0.1):
        random.seed(0)
        self.loss_type = loss_type
        self.lr = lr
        self.hidden_size = hidden_size

        # 权重初始化
        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        """前向传播，缓存中间值供反向传播使用。"""
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))  # ReLU 激活

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        """反向传播，根据损失类型计算梯度并更新权重。"""
        # 损失对输出的梯度
        if self.loss_type == "mse":
            d_loss = 2.0 * (self.out - target)
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, self.out))
            d_loss = -(target / p) + (1 - target) / (1 - p)

        # 链式法则：损失 → sigmoid → 隐藏层 → 输入层
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def compute_loss(self, pred, target):
        """计算单样本损失值。"""
        if self.loss_type == "mse":
            return (pred - target) ** 2
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    def train(self, data, epochs=200):
        """训练指定轮次，返回每轮的损失和准确率。"""
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += self.compute_loss(pred, y)
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


# =============================================================================
# 主程序：演示所有损失函数
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第 1 步：回归损失 — MSE / MAE / Huber")
    print("=" * 60)
    reg_preds = [2.5, -0.5, 3.0, 7.2]
    reg_targets = [3.0, 0.0, 2.5, 8.0]
    print(f"  预测值: {reg_preds}")
    print(f"  目标值: {reg_targets}")
    print(f"  MSE 损失:  {mse(reg_preds, reg_targets):.6f}")
    print(f"  MAE 损失:  {mae(reg_preds, reg_targets):.6f}")
    print(f"  Huber 损失: {huber_loss(reg_preds, reg_targets, delta=1.0):.6f}")
    print(f"  MSE 梯度:  {[f'{g:.4f}' for g in mse_gradient(reg_preds, reg_targets)]}")
    print(f"  Huber 梯度: {[f'{g:.4f}' for g in huber_gradient(reg_preds, reg_targets, delta=1.0)]}")

    print("\n" + "=" * 60)
    print("第 2 步：二元交叉熵（BCE）")
    print("=" * 60)
    cls_preds = [0.9, 0.1, 0.7, 0.4]
    cls_targets = [1.0, 0.0, 1.0, 0.0]
    print(f"  预测值: {cls_preds}")
    print(f"  目标值: {cls_targets}")
    print(f"  BCE 损失: {binary_cross_entropy(cls_preds, cls_targets):.6f}")
    print(f"  BCE 梯度: {[f'{g:.4f}' for g in bce_gradient(cls_preds, cls_targets)]}")

    print("\n  BCE 在不同置信度下的表现（真实标签 = 1）:")
    for conf in [0.01, 0.1, 0.5, 0.9, 0.99]:
        ce = -(1.0 * math.log(max(1e-15, conf)))
        ms = (conf - 1.0) ** 2
        print(f"    p={conf:.2f}: BCE={ce:.4f}, MSE={ms:.4f}, 比值={ce / max(0.0001, ms):.1f}x")

    print("\n" + "=" * 60)
    print("第 3 步：多元交叉熵（CCE）+ Softmax")
    print("=" * 60)
    logits = [2.0, 1.0, 0.1, -1.0, 3.0]
    target_idx = 4
    probs = softmax(logits)
    print(f"  Logits:  {logits}")
    print(f"  Softmax: {[f'{p:.4f}' for p in probs]}")
    print(f"  目标类别: {target_idx}")
    print(f"  CCE 损失: {categorical_cross_entropy(logits, target_idx):.6f}")
    print(f"  梯度:     {[f'{g:.4f}' for g in cce_gradient(logits, target_idx)]}")

    print("\n" + "=" * 60)
    print("第 4 步：标签平滑")
    print("=" * 60)
    num_classes = 5
    hard_loss = categorical_cross_entropy(logits, target_idx)
    smooth_loss = label_smoothed_cce(logits, target_idx, num_classes, alpha=0.1)
    print(f"  硬标签损失:    {hard_loss:.6f}")
    print(f"  平滑标签损失:  {smooth_loss:.6f}")
    print(f"  平滑使损失增加 {smooth_loss - hard_loss:.6f}")
    print(f"  效果：模型不再追求概率恰好为 1.0，而是 0.91")

    print("\n" + "=" * 60)
    print("第 5 步：Focal Loss")
    print("=" * 60)
    imbalanced_preds = [0.95, 0.9, 0.6, 0.3, 0.1]
    imbalanced_targets = [1.0, 1.0, 1.0, 0.0, 0.0]
    bce_val = binary_cross_entropy(imbalanced_preds, imbalanced_targets)
    focal_val = focal_loss_binary(imbalanced_preds, imbalanced_targets, gamma=2.0)
    print(f"  预测值: {imbalanced_preds}")
    print(f"  目标值: {imbalanced_targets}")
    print(f"  BCE 损失:     {bce_val:.6f}")
    print(f"  Focal 损失:   {focal_val:.6f}")
    print(f"  Focal 降低了易分类样本的权重，聚焦于难分类样本")

    print("\n" + "=" * 60)
    print("第 6 步：对比损失（InfoNCE / NT-Xent）")
    print("=" * 60)
    random.seed(42)
    anchor = [random.gauss(0, 1) for _ in range(8)]
    positive = [a + random.gauss(0, 0.1) for a in anchor]  # 正样本：加小噪声
    negatives = [[random.gauss(0, 1) for _ in range(8)] for _ in range(7)]

    loss_val = contrastive_loss(anchor, positive, negatives, temperature=0.07)
    sim_pos = cosine_similarity(anchor, positive)
    sim_negs = [cosine_similarity(anchor, neg) for neg in negatives]
    print(f"  锚点-正样本相似度: {sim_pos:.4f}")
    print(f"  锚点-负样本相似度: {[f'{s:.4f}' for s in sim_negs]}")
    print(f"  对比损失 (tau=0.07): {loss_val:.4f}")

    loss_easy = contrastive_loss(anchor, positive, negatives, temperature=0.5)
    print(f"  对比损失 (tau=0.5):  {loss_easy:.4f}")
    print(f"  温度越低，分布越锐，对不完美分离的惩罚越大")

    print("\n" + "=" * 60)
    print("第 7 步：Hinge Loss 与 Triplet Loss")
    print("=" * 60)
    # Hinge Loss：SVM 格式（标签为 +1 / -1）
    svm_preds = [0.8, -0.5, 1.2, -0.3]
    svm_targets = [1.0, -1.0, 1.0, -1.0]
    print(f"  Hinge Loss: {hinge_loss(svm_preds, svm_targets, margin=1.0):.6f}")
    print(f"  Hinge 梯度: {[f'{g:.4f}' for g in hinge_gradient(svm_preds, svm_targets, margin=1.0)]}")

    # Triplet Loss
    tri_anchor = [1.0, 2.0]
    tri_positive = [1.1, 2.1]   # 正样本：距离很近
    tri_negative = [5.0, 5.0]   # 负样本：距离很远
    print(f"  Triplet Loss: {triplet_loss(tri_anchor, tri_positive, tri_negative, margin=0.2):.6f}")

    print("\n" + "=" * 60)
    print("第 8 步：MSE vs BCE 分类对比实验")
    print("=" * 60)
    data = make_iris_binary_data()

    for loss_type in ["mse", "bce"]:
        print(f"\n--- 使用 {loss_type.upper()} 训练 ---")
        net = LossComparisonNetwork(loss_type=loss_type, hidden_size=8, lr=0.1)
        results = net.train(data, epochs=200)
        final_loss, final_acc = results[-1]
        print(f"  最终结果: loss={final_loss:.4f}, accuracy={final_acc:.1f}%")

    print("\n=== 核心结论 ===")
    print("  交叉熵在分类任务上收敛更快，因为：")
    print("  - 预测错误时梯度极强（-p 接近 0 时梯度爆炸）")
    print("  - 预测正确时梯度趋近 0（无需继续修正）")
    print("  - MSE 在 sigmoid 饱和区梯度消失，导致学习停滞")
