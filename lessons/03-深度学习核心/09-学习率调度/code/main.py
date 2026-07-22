# main.py — 学习率调度策略：从零实现与对比实验
# 依赖：math, random（均为标准库，无需额外安装）
# 对应课程：阶段 03 · 09（学习率调度）
#
# 本文件实现以下调度策略：
#   1. 常量学习率
#   2. 阶梯衰减 (Step Decay)
#   3. 指数衰减 (Exponential Decay)
#   4. 余弦退火 (Cosine Annealing)
#   5. 余弦退火 + 预热 (Warmup + Cosine)
#   6. 单周期策略 (1cycle Policy)
#
# 并在圆环分类数据集上对比各策略的训练效果。

import math
import random


# ============================================================
# 第 1 步：实现各种学习率调度函数
# ============================================================

def constant_schedule(step, lr=0.01, **kwargs):
    """常量学习率——始终返回同一个值。"""
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    """阶梯衰减：每隔 step_size 步，学习率乘以 gamma。

    ResNet 时代的经典做法——每隔 30 个轮次衰减 10 倍。
    """
    return lr * (gamma ** (step // step_size))


def exponential_decay_schedule(step, lr=0.01, gamma=0.999, **kwargs):
    """指数衰减：每个步骤都乘以 gamma，平滑递减。

    gamma 越接近 1，衰减越慢；gamma=0.999 意味着每 1000 步衰减到约 37%。
    """
    return lr * (gamma ** step)


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    """余弦退火：从 lr_max 沿余弦曲线衰减到 lr_min。

    开始时衰减慢，中间加速，末尾再变慢——匹配"大部分学习发生在训练中期"的经验观察。
    """
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000,
                           warmup_steps=100, lr_min=1e-5, **kwargs):
    """预热 + 余弦退火：先线性升温到 lr_max，再余弦衰减。

    这是 Llama 3、GPT-3、BERT 等现代大模型的标准做法。
    预热阶段让 Adam 等自适应优化器的统计量（动量、方差）稳定下来。
    """
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    """单周期策略 (1cycle)：前半段从低升到高，后半段从高降到极低。

    Leslie Smith (2018) 的发现：高学习率阶段充当正则化，
    帮助模型探索更广阔的损失景观，之后再精细收敛。
    """
    mid = max(total_steps // 2, 1)
    if step < mid:
        # 前半段：从 lr/25 线性升到 lr
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        # 后半段：从 lr 线性降到 lr/10000
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress


# ============================================================
# 第 2 步：ASCII 可视化各调度曲线
# ============================================================

def visualize_schedule(name, schedule_fn, total_steps=500, **kwargs):
    """用 ASCII 柱状图展示调度曲线的形状。"""
    steps = list(range(0, total_steps, total_steps // 20))
    if total_steps - 1 not in steps:
        steps.append(total_steps - 1)

    lrs = [schedule_fn(s, total_steps=total_steps, **kwargs) for s in steps]
    max_lr = max(lrs) if max(lrs) > 0 else 1.0

    print(f"\n  {name}:")
    for s, lr_val in zip(steps, lrs):
        bar_len = int(lr_val / max_lr * 40)
        bar = "#" * bar_len
        print(f"    Step {s:4d}: lr={lr_val:.6f} {bar}")


# ============================================================
# 第 3 步：纯 Python 训练网络（用于对比各调度策略）
# ============================================================

def sigmoid(x):
    """Sigmoid 激活函数，带数值稳定性处理。"""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    """ReLU 激活函数。"""
    return max(0.0, x)


def relu_deriv(x):
    """ReLU 的导数。"""
    return 1.0 if x > 0 else 0.0


def make_circle_data(n=200, seed=42):
    """生成圆环分类数据集：圆内为类别 1，圆外为类别 0。"""
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def train_with_schedule(schedule_fn, schedule_name, data,
                        epochs=300, base_lr=0.05, **kwargs):
    """使用指定调度策略训练一个两层全连接网络。

    网络结构：输入(2) -> 隐藏层(8, ReLU) -> 输出(1, Sigmoid)
    损失函数：均方误差 (MSE)
    """
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)

    # He 初始化：标准差 = sqrt(2 / fan_in)
    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, target in data:
            # 根据当前步骤获取学习率
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)

            # 前向传播
            z1 = []
            h = []
            for i in range(hidden_size):
                z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                z1.append(z)
                h.append(relu(z))

            z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)

            # 反向传播 + 梯度下降
            error = out - target
            d_out = error * out * (1 - out)

            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out

            total_loss += (out - target) ** 2
            if (out >= 0.5) == (target >= 0.5):
                correct += 1
            step += 1

        avg_loss = total_loss / len(data)
        epoch_losses.append(avg_loss)

    return epoch_losses


# ============================================================
# 第 4 步：对比所有调度策略
# ============================================================

def compare_schedules(data):
    """用相同网络、相同数据、相同初始权重，对比五种调度策略。"""
    configs = [
        ("常量学习率", constant_schedule, {}),
        ("阶梯衰减", step_decay_schedule,
         {"step_size": 15000, "gamma": 0.1}),
        ("指数衰减", exponential_decay_schedule,
         {"gamma": 0.9998}),
        ("余弦退火", cosine_schedule,
         {"lr_min": 1e-5}),
        ("预热+余弦", warmup_cosine_schedule,
         {"warmup_steps": 3000, "lr_min": 1e-5}),
        ("单周期(1cycle)", one_cycle_schedule, {}),
    ]

    print(f"\n  {'策略':<18} {'起始 Loss':>10} {'中期 Loss':>10} "
          f"{'最终 Loss':>10} {'最低 Loss':>10}")
    print("  " + "-" * 62)

    for name, schedule_fn, extra_kwargs in configs:
        losses = train_with_schedule(
            schedule_fn, name, data, epochs=300, base_lr=0.05,
            **extra_kwargs
        )
        mid_idx = len(losses) // 2
        best = min(losses)
        print(f"  {name:<18} {losses[0]:>10.6f} {losses[mid_idx]:>10.6f} "
              f"{losses[-1]:>10.6f} {best:>10.6f}")


# ============================================================
# 第 5 步：学习率敏感性实验
# ============================================================

def lr_sensitivity(data):
    """展示学习率过高、过低、合适的三种失败模式。"""
    learning_rates = [1.0, 0.1, 0.05, 0.01, 0.001, 0.0001]

    print(f"\n  {'学习率':>10} {'起始 Loss':>12} {'最终 Loss':>12} {'状态':>15}")
    print("  " + "-" * 52)

    for lr in learning_rates:
        losses = train_with_schedule(
            constant_schedule, f"lr={lr}", data, epochs=100, base_lr=lr
        )
        start = losses[0]
        end = losses[-1]

        # 判断训练状态
        if math.isnan(end) or end > 1.0:
            status = "发散"
        elif end > start * 0.9:
            status = "几乎没动"
        elif end < 0.15:
            status = "收敛"
        else:
            status = "正在学习"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>12.6f} {end_str:>12} {status:>15}")


# ============================================================
# 第 6 步：预热步数影响实验
# ============================================================

def warmup_impact(data):
    """对比不同预热比例对训练效果的影响。"""
    warmup_fractions = [0.0, 0.01, 0.05, 0.10, 0.20]
    total_steps = 300 * len(data)

    print(f"\n  {'预热比例':>10} {'预热步数':>12} {'最终 Loss':>12} {'最低 Loss':>12}")
    print("  " + "-" * 50)

    for frac in warmup_fractions:
        warmup_steps = int(total_steps * frac)
        losses = train_with_schedule(
            warmup_cosine_schedule, f"warmup={frac}", data,
            epochs=300, base_lr=0.05,
            warmup_steps=warmup_steps, lr_min=1e-5
        )
        best = min(losses)
        print(f"  {frac*100:>9.0f}% {warmup_steps:>12d} "
              f"{losses[-1]:>12.6f} {best:>12.6f}")


# ============================================================
# 主程序：运行所有实验
# ============================================================

if __name__ == "__main__":
    # 生成训练数据
    data = make_circle_data()
    print("=" * 60)
    print("  学习率调度策略对比实验")
    print("=" * 60)

    # 实验 1：各调度策略的形状
    print("\n" + "=" * 60)
    print("  实验 1：各调度策略的形状")
    print("=" * 60)
    visualize_schedule("常量学习率", constant_schedule, lr=0.05)
    visualize_schedule("阶梯衰减", step_decay_schedule,
                       lr=0.05, step_size=125, gamma=0.5)
    visualize_schedule("指数衰减", exponential_decay_schedule,
                       lr=0.05, gamma=0.99)
    visualize_schedule("余弦退火", cosine_schedule, lr=0.05, lr_min=1e-5)
    visualize_schedule("预热+余弦", warmup_cosine_schedule,
                       lr=0.05, warmup_steps=50, lr_min=1e-5)
    visualize_schedule("单周期(1cycle)", one_cycle_schedule, lr=0.05)

    # 实验 2：学习率敏感性
    print("\n" + "=" * 60)
    print("  实验 2：学习率敏感性——三种失败模式")
    print("=" * 60)
    lr_sensitivity(data)

    # 实验 3：各调度策略的训练效果对比
    print("\n" + "=" * 60)
    print("  实验 3：各调度策略的训练效果对比")
    print("=" * 60)
    compare_schedules(data)

    # 实验 4：预热步数的影响
    print("\n" + "=" * 60)
    print("  实验 4：预热步数对训练稳定性的影响")
    print("=" * 60)
    warmup_impact(data)

    print("\n" + "=" * 60)
    print("  实验完成！")
    print("=" * 60)
