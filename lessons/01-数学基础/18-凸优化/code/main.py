# convex-optimization.py — 凸优化从零实现
# 依赖：numpy>=1.24, matplotlib>=3.7（可选，用于可视化）
# 对应课程：第 01 阶段 · 18（凸优化）

import math
import random


# ============================================================
# 第 1 步：凸性判定器
# ============================================================

def check_convexity(f, dim, bounds=(-5, 5), samples=2000, label=""):
    """通过随机采样验证函数是否满足凸函数定义。

    对于凸函数 f，任意两点 x, y 和 t ∈ [0,1]，满足：
        f(t*x + (1-t)*y) <= t*f(x) + (1-t)*f(y)

    如果有任何采样违反该不等式，则函数非凸。

    Args:
        f: 待检测函数
        dim: 输入维度
        bounds: 采样区间
        samples: 采样次数
        label: 输出标签

    Returns:
        (is_convex, violations): 是否凸、违反次数
    """
    violations = 0
    worst_violation = 0.0
    for _ in range(samples):
        x = [random.uniform(*bounds) for _ in range(dim)]
        y = [random.uniform(*bounds) for _ in range(dim)]
        t = random.uniform(0, 1)
        # 凸组合：两点之间的插值点
        mid = [t * xi + (1 - t) * yi for xi, yi in zip(x, y)]
        lhs = f(mid)
        rhs = t * f(x) + (1 - t) * f(y)
        gap = lhs - rhs
        if gap > 1e-10:
            violations += 1
            worst_violation = max(worst_violation, gap)
    is_convex = violations == 0
    status = "CONVEX" if is_convex else "NOT CONVEX"
    if label:
        print(f"  {label:30s}  {status:10s}  violations: {violations}/{samples}"
              + (f"  worst: {worst_violation:.6f}" if violations > 0 else ""))
    return is_convex, violations


# ============================================================
# 第 2 步：Hessian 矩阵特征值分析
# ============================================================

def hessian_eigenvalues_2d(H):
    """计算 2x2 矩阵的特征值（解析解）。

    对于 H = [[a, b], [c, d]]，特征值满足：
        λ^2 - (a+d)*λ + (ad - bc) = 0
    """
    a, b = H[0][0], H[0][1]
    c, d = H[1][0], H[1][1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        return None, None
    sqrt_disc = math.sqrt(discriminant)
    e1 = (trace + sqrt_disc) / 2
    e2 = (trace - sqrt_disc) / 2
    return e1, e2


def is_positive_semidefinite_2d(H):
    """判断 2x2 Hessian 矩阵是否半正定。

    半正定 ⟺ 所有特征值 >= 0 ⟺ 函数在该点处凸
    """
    e1, e2 = hessian_eigenvalues_2d(H)
    if e1 is None:
        return False
    return e1 >= -1e-10 and e2 >= -1e-10


# ============================================================
# 第 3 步：牛顿法
# ============================================================

def invert_2x2(H):
    """2x2 矩阵求逆。

    H = [[a, b], [c, d]]，则 H^(-1) = (1/det) * [[d, -b], [-c, a]]
    det 接近零时返回 None（矩阵奇异）。
    """
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    if abs(det) < 1e-15:
        return None
    return [
        [H[1][1] / det, -H[0][1] / det],
        [-H[1][0] / det, H[0][0] / det],
    ]


def mat_vec_2d(M, v):
    """2x2 矩阵与 2 维向量的乘法。"""
    return [
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1],
    ]


def newtons_method(grad_f, hessian_f, x0, steps=100, tol=1e-12):
    """牛顿法优化。

    每一步计算 Hessian 的逆矩阵乘以梯度，更新参数。
    对于二次函数，理论上一步收敛到精确解。

    Args:
        grad_f: 梯度函数
        hessian_f: Hessian 矩阵函数
        x0: 初始点
        steps: 最大迭代次数
        tol: 梯度范数阈值

    Returns:
        history: 参数历史
    """
    x = list(x0)
    history = [x[:]]
    for _ in range(steps):
        g = grad_f(x)
        # 梯度足够小时判定收敛
        if sum(gi ** 2 for gi in g) < tol:
            break
        H = hessian_f(x)
        H_inv = invert_2x2(H)
        if H_inv is None:
            break  # Hessian 奇异，无法求逆
        # 牛顿步：x_new = x - H^(-1) * grad
        dx = mat_vec_2d(H_inv, g)
        x = [x[0] - dx[0], x[1] - dx[1]]
        if any(math.isnan(xi) or math.isinf(xi) for xi in x):
            break
        history.append(x[:])
    return history


# ============================================================
# 第 4 步：梯度下降（用于对比）
# ============================================================

def optimize_gd(grad_f, x0, lr=0.01, steps=1000, tol=1e-12):
    """梯度下降。

    沿负梯度方向以固定步长移动。
    条件数大时会在狭长山谷中震荡。
    """
    x = list(x0)
    history = [x[:]]
    for _ in range(steps):
        g = grad_f(x)
        if sum(gi ** 2 for gi in g) < tol:
            break
        x = [xi - lr * gi for xi, gi in zip(x, g)]
        if any(math.isnan(xi) or math.isinf(xi) for xi in x):
            break
        history.append(x[:])
    return history


# ============================================================
# 第 5 步：拉格朗日乘子法
# ============================================================

def lagrange_solve(f_grad, g_val, g_grad, x0, lr=0.01,
                   lr_lambda=0.01, steps=5000):
    """使用拉格朗日乘子法求解等式约束优化问题。

    最小化 f(x)，满足 g(x) = 0。

    对 x 沿拉格朗日函数的梯度下降（最小化 L），
    对 λ 沿拉格朗日函数的梯度上升（最大化对偶）。

    Args:
        f_grad: 目标函数梯度
        g_val: 约束函数值
        g_grad: 约束函数梯度
        x0: 初始点
        lr: x 的学习率
        lr_lambda: λ 的学习率
        steps: 迭代次数

    Returns:
        history: (x, λ, g) 历史
    """
    x = list(x0)
    lam = 0.0
    history = []
    for _ in range(steps):
        fg = f_grad(x)
        gv = g_val(x)
        gg = g_grad(x)
        # x 的更新：沿 L 对 x 的梯度下降
        x = [
            xi - lr * (fgi + lam * ggi)
            for xi, fgi, ggi in zip(x, fg, gg)
        ]
        # λ 的更新：沿 L 对 λ 的梯度上升（对偶上升）
        lam = lam + lr_lambda * gv
        if any(math.isnan(xi) or math.isinf(xi) for xi in x):
            break
        history.append((x[:], lam, gv))
    return history


# ============================================================
# 演示函数
# ============================================================

def demo_convexity_checker():
    print("=" * 65)
    print("  凸性判定器")
    print("=" * 65)
    print()

    random.seed(42)

    # 凸函数组
    check_convexity(lambda x: x[0] ** 2, 1, label="f(x) = x^2")
    check_convexity(lambda x: abs(x[0]), 1, label="f(x) = |x|")
    check_convexity(lambda x: math.exp(x[0]), 1, label="f(x) = e^x")
    check_convexity(lambda x: x[0] ** 2 + x[1] ** 2, 2, label="f(x,y) = x^2 + y^2")
    check_convexity(lambda x: max(x[0], 0), 1, label="f(x) = max(0, x) [ReLU]")

    # 非凸函数组
    check_convexity(lambda x: math.sin(x[0]), 1, label="f(x) = sin(x)")
    check_convexity(lambda x: x[0] ** 3, 1, label="f(x) = x^3")
    check_convexity(lambda x: -x[0] ** 2, 1, label="f(x) = -x^2")
    check_convexity(
        lambda x: math.sin(x[0]) * math.cos(x[1]),
        2,
        label="f(x,y) = sin(x)*cos(y)"
    )
    check_convexity(
        lambda x: x[0] ** 2 - x[1] ** 2,
        2,
        label="f(x,y) = x^2 - y^2 [鞍点]"
    )

    print()
    print("  上方：预期为凸函数。下方：预期为非凸函数。")


def demo_hessian_analysis():
    print()
    print()
    print("=" * 65)
    print("  Hessian 矩阵与曲率分析")
    print("=" * 65)

    print()
    print("  f(x,y) = 5x^2 + y^2（狭长碗）")
    H1 = [[10, 0], [0, 2]]
    e1, e2 = hessian_eigenvalues_2d(H1)
    psd = is_positive_semidefinite_2d(H1)
    print(f"  Hessian: [[{H1[0][0]}, {H1[0][1]}], [{H1[1][0]}, {H1[1][1]}]]")
    print(f"  特征值: {e1:.1f}, {e2:.1f}")
    print(f"  条件数: {e1 / e2:.1f}")
    print(f"  半正定: {psd}")
    print(f"  凸函数: {psd}")

    print()
    print("  f(x,y) = x^2 - y^2（鞍点）")
    H2 = [[2, 0], [0, -2]]
    e1, e2 = hessian_eigenvalues_2d(H2)
    psd = is_positive_semidefinite_2d(H2)
    print(f"  Hessian: [[{H2[0][0]}, {H2[0][1]}], [{H2[1][0]}, {H2[1][1]}]]")
    print(f"  特征值: {e1:.1f}, {e2:.1f}")
    print(f"  半正定: {psd}")
    print(f"  鞍点: 特征值符号相反，确认是鞍点")

    print()
    print("  f(x,y) = x^2 + 3xy + y^2")
    H3 = [[2, 3], [3, 2]]
    e1, e2 = hessian_eigenvalues_2d(H3)
    psd = is_positive_semidefinite_2d(H3)
    print(f"  Hessian: [[{H3[0][0]}, {H3[0][1]}], [{H3[1][0]}, {H3[1][1]}]]")
    print(f"  特征值: {e1:.1f}, {e2:.1f}")
    print(f"  半正定: {psd}")
    print(f"  凸函数: {psd}（负特征值意味着不定）")

    print()
    print("  Rosenbrock 函数在最小值点 (1, 1)")
    Hmin = [[802, -400], [-400, 200]]
    e1, e2 = hessian_eigenvalues_2d(Hmin)
    psd = is_positive_semidefinite_2d(Hmin)
    print(f"  Hessian: [[{Hmin[0][0]}, {Hmin[0][1]}], [{Hmin[1][0]}, {Hmin[1][1]}]]")
    print(f"  特征值: {e1:.2f}, {e2:.2f}")
    print(f"  条件数: {e1 / e2:.1f}")
    print(f"  在 (1,1) 处半正定: {psd}")


def demo_newtons_method():
    print()
    print()
    print("=" * 65)
    print("  牛顿法 vs 梯度下降")
    print("=" * 65)

    def f(x):
        return 50 * x[0] ** 2 + x[1] ** 2

    def grad_f(x):
        return [100 * x[0], 2 * x[1]]

    def hessian_f(x):
        return [[100, 0], [0, 2]]

    start = [10.0, 10.0]

    print()
    print(f"  函数: f(x,y) = 50x^2 + y^2")
    print(f"  最小值在: (0, 0), f = 0")
    print(f"  起始点: ({start[0]}, {start[1]}), f = {f(start):.1f}")
    print(f"  条件数: {100 / 2:.0f}（狭长山谷）")

    newton_hist = newtons_method(grad_f, hessian_f, start, steps=50)
    gd_hist = optimize_gd(grad_f, start, lr=0.015, steps=500)

    print()
    print(f"  牛顿法: {len(newton_hist) - 1} 步收敛")
    print(f"  {'步数':>6s}  {'x':>12s}  {'y':>12s}  {'f(x,y)':>14s}")
    print(f"  {'-' * 48}")
    for i, p in enumerate(newton_hist):
        print(f"  {i:6d}  {p[0]:12.8f}  {p[1]:12.8f}  {f(p):14.8f}")

    print()
    threshold = 1e-10
    gd_converged = len(gd_hist) - 1
    for i, p in enumerate(gd_hist):
        if f(p) < threshold:
            gd_converged = i
            break

    print(f"  梯度下降 (lr=0.015): 共 {len(gd_hist) - 1} 步")
    steps_to_show = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 499]
    steps_to_show = [s for s in steps_to_show if s < len(gd_hist)]
    print(f"  {'步数':>6s}  {'x':>12s}  {'y':>12s}  {'f(x,y)':>14s}")
    print(f"  {'-' * 48}")
    for i in steps_to_show:
        p = gd_hist[i]
        print(f"  {i:6d}  {p[0]:12.8f}  {p[1]:12.8f}  {f(p):14.8f}")

    print()
    print(f"  牛顿法 {len(newton_hist) - 1} 步收敛")
    print(f"  GD 在步 {gd_converged} 达到 f < {threshold}"
          + ("（未收敛）" if gd_converged == len(gd_hist) - 1 else ""))
    print()
    print("  牛顿法对二次函数精确——一步到达。")
    print("  GD 在高条件数时表现差——在狭长山谷中震荡。")


def demo_condition_number_effect():
    print()
    print()
    print("=" * 65)
    print("  条件数对梯度下降的影响")
    print("=" * 65)
    print()

    conditions = [1, 5, 10, 50, 100]

    print(f"  最小化 f(x,y) = c*x^2 + y^2, 起点 = (10, 10)")
    print(f"  牛顿法对任何条件数都一步收敛。")
    print()
    print(f"  {'条件数':>8s}  {'GD步数':>10s}  {'牛顿步数':>14s}  {'GD最终loss':>14s}")
    print(f"  {'-' * 50}")

    for c in conditions:
        def grad_f(x, c=c):
            return [2 * c * x[0], 2 * x[1]]

        def hess_f(x, c=c):
            return [[2 * c, 0], [0, 2]]

        def f_val(x, c=c):
            return c * x[0] ** 2 + x[1] ** 2

        start = [10.0, 10.0]
        lr = 0.9 / (2 * c)

        gd_hist = optimize_gd(grad_f, start, lr=lr, steps=2000)
        newton_hist = newtons_method(grad_f, hess_f, start, steps=50)

        gd_steps = len(gd_hist) - 1
        newton_steps = len(newton_hist) - 1
        gd_final = f_val(gd_hist[-1])

        print(f"  {c:8d}  {gd_steps:10d}  {newton_steps:14d}  {gd_final:14.2e}")


def demo_lagrange_multipliers():
    print()
    print()
    print("=" * 65)
    print("  拉格朗日乘子法求解器")
    print("=" * 65)

    print()
    print("  问题: 最小化 f(x,y) = x^2 + y^2")
    print("  约束: g(x,y) = x + y - 1 = 0")
    print("  解析解: x = 0.5, y = 0.5, λ = -1")
    print()

    def f_grad(x):
        return [2 * x[0], 2 * x[1]]

    def g_val(x):
        return x[0] + x[1] - 1

    def g_grad(x):
        return [1.0, 1.0]

    history = lagrange_solve(f_grad, g_val, g_grad, [2.0, 2.0],
                             lr=0.01, lr_lambda=0.01, steps=5000)

    milestones = [0, 49, 499, 999, 2499, 4999]
    milestones = [m for m in milestones if m < len(history)]

    print(f"  {'步数':>6s}  {'x':>8s}  {'y':>8s}  {'λ':>8s}  {'g(x,y)':>10s}  {'f(x,y)':>10s}")
    print(f"  {'-' * 56}")

    for i in milestones:
        x, lam, gv = history[i]
        fv = x[0] ** 2 + x[1] ** 2
        print(f"  {i + 1:6d}  {x[0]:8.4f}  {x[1]:8.4f}  {lam:8.4f}  {gv:10.6f}  {fv:10.6f}")

    final_x, final_lam, final_g = history[-1]
    print()
    print(f"  最终解: x = {final_x[0]:.6f}, y = {final_x[1]:.6f}")
    print(f"  λ = {final_lam:.6f}")
    print(f"  约束违反: {abs(final_g):.2e}")
    print(f"  目标值: {final_x[0] ** 2 + final_x[1] ** 2:.6f}")

    print()
    print()
    print("  问题: 最小化 f(x,y) = (x-3)^2 + (y-3)^2")
    print("  约束: x + 2y = 4")
    print()

    def f_grad2(x):
        return [2 * (x[0] - 3), 2 * (x[1] - 3)]

    def g_val2(x):
        return x[0] + 2 * x[1] - 4

    def g_grad2(x):
        return [1.0, 2.0]

    history2 = lagrange_solve(f_grad2, g_val2, g_grad2, [0.0, 0.0],
                              lr=0.002, lr_lambda=0.002, steps=20000)

    final_x2, final_lam2, final_g2 = history2[-1]
    print(f"  数值解: x = {final_x2[0]:.6f}, y = {final_x2[1]:.6f}")
    print(f"  λ = {final_lam2:.6f}")
    print(f"  约束 x + 2y = {final_x2[0] + 2 * final_x2[1]:.6f}（目标: 4）")
    print(f"  目标值: {(final_x2[0] - 3) ** 2 + (final_x2[1] - 3) ** 2:.6f}")

    x_exact = 2.0
    y_exact = 1.0
    print()
    print(f"  解析解: x = 2, y = 1, λ = 2")
    print(f"  误差: {math.sqrt((final_x2[0] - x_exact) ** 2 + (final_x2[1] - y_exact) ** 2):.2e}")


def demo_regularization_geometry():
    print()
    print()
    print("=" * 65)
    print("  正则化 = 约束优化（几何视角）")
    print("=" * 65)
    print()

    print("  无约束最优解 (3, 2)")
    print()

    print("  L2 约束: x^2 + y^2 <= 1（单位圆）")
    x_l2 = [3.0 / math.sqrt(13), 2.0 / math.sqrt(13)]
    print(f"  投影解: ({x_l2[0]:.6f}, {x_l2[1]:.6f})")
    print(f"  ||w||^2 = {x_l2[0] ** 2 + x_l2[1] ** 2:.6f}")
    print(f"  两个权重都非零：权重收缩但不消除")
    print()

    print("  L1 约束: |x| + |y| <= 1（单位菱形）")
    print("  解位于菱形的角点处。")

    best_val = float('inf')
    best_x = None

    for x_cand in [i * 0.001 for i in range(1001)]:
        y_cand = 1.0 - x_cand
        val = (x_cand - 3) ** 2 + (y_cand - 2) ** 2
        if val < best_val:
            best_val = val
            best_x = [x_cand, y_cand]

    for y_cand_raw in [i * 0.001 for i in range(1001)]:
        x_cand = 1.0 - y_cand_raw
        val = (x_cand - 3) ** 2 + (y_cand_raw - 2) ** 2
        if val < best_val:
            best_val = val
            best_x = [x_cand, y_cand_raw]

    corner_vals = [
        ([1.0, 0.0], (1 - 3) ** 2 + (0 - 2) ** 2),
        ([0.0, 1.0], (0 - 3) ** 2 + (1 - 2) ** 2),
        ([-1.0, 0.0], (-1 - 3) ** 2 + (0 - 2) ** 2),
        ([0.0, -1.0], (0 - 3) ** 2 + (-1 - 2) ** 2),
    ]

    print(f"  扫描菱形边界...")
    print(f"  边上最优: ({best_x[0]:.4f}, {best_x[1]:.4f}), 目标值 = {best_val:.4f}")
    print()
    print(f"  菱形角点:")
    for pt, val in corner_vals:
        marker = " <-- 最优角点" if val == min(v for _, v in corner_vals) else ""
        print(f"    ({pt[0]:5.1f}, {pt[1]:5.1f})  目标值 = {val:.1f}{marker}")

    print()
    print("  L1 将解推向角点（坐标轴对齐）→ 稀疏性")
    print("  L2 将解推向圆上最近点 → 权重收缩")


def demo_first_vs_second_order():
    print()
    print()
    print("=" * 65)
    print("  一阶 vs 二阶：收敛速度对比")
    print("=" * 65)
    print()

    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def rosenbrock_grad(x):
        dx = -2 * (1 - x[0]) + 200 * (x[1] - x[0] ** 2) * (-2 * x[0])
        dy = 200 * (x[1] - x[0] ** 2)
        return [dx, dy]

    def rosenbrock_hessian(x):
        h00 = 2 - 400 * x[1] + 1200 * x[0] ** 2
        h01 = -400 * x[0]
        h10 = -400 * x[0]
        h11 = 200
        return [[h00, h01], [h10, h11]]

    start = [0.5, 0.5]

    print(f"  Rosenbrock 函数: f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print(f"  最小值在 (1, 1), f = 0")
    print(f"  起点: ({start[0]}, {start[1]}), f = {rosenbrock(start):.4f}")
    print()

    newton_hist = newtons_method(rosenbrock_grad, rosenbrock_hessian, start, steps=100)

    gd_hist = optimize_gd(rosenbrock_grad, start, lr=0.001, steps=10000)

    print(f"  牛顿法（{len(newton_hist) - 1} 步）:")
    print(f"  {'步数':>6s}  {'x':>10s}  {'y':>10s}  {'f(x,y)':>14s}")
    print(f"  {'-' * 44}")
    for i, p in enumerate(newton_hist[:15]):
        print(f"  {i:6d}  {p[0]:10.6f}  {p[1]:10.6f}  {rosenbrock(p):14.8f}")
    if len(newton_hist) > 15:
        p = newton_hist[-1]
        print(f"  {len(newton_hist) - 1:6d}  {p[0]:10.6f}  {p[1]:10.6f}  {rosenbrock(p):14.8f}")

    print()

    gd_threshold = 1e-6
    gd_converge_step = len(gd_hist) - 1
    for i, p in enumerate(gd_hist):
        if rosenbrock(p) < gd_threshold:
            gd_converge_step = i
            break

    print(f"  梯度下降 (lr=0.001, {len(gd_hist) - 1} 步):")
    show_steps = [0, 10, 100, 500, 1000, 2000, 5000, 9999]
    show_steps = [s for s in show_steps if s < len(gd_hist)]
    print(f"  {'步数':>6s}  {'x':>10s}  {'y':>10s}  {'f(x,y)':>14s}")
    print(f"  {'-' * 44}")
    for i in show_steps:
        p = gd_hist[i]
        print(f"  {i:6d}  {p[0]:10.6f}  {p[1]:10.6f}  {rosenbrock(p):14.8f}")

    print()
    print(f"  牛顿法 {len(newton_hist) - 1} 步收敛（f < 1e-12）")
    if gd_converge_step < len(gd_hist) - 1:
        print(f"  GD {gd_converge_step} 步收敛（f < {gd_threshold}）")
    else:
        final_gd = rosenbrock(gd_hist[-1])
        print(f"  GD 在 {len(gd_hist) - 1} 步内未收敛（最终: {final_gd:.2e}）")

    print()
    print("  牛顿法每步 O(n^3) 但二次收敛。")
    print("  GD 每步 O(n) 但线性收敛。")
    print("  小规模问题牛顿法胜，百万参数 GD 胜。")


def demo_convex_vs_nonconvex_landscape():
    print()
    print()
    print("=" * 65)
    print("  凸 vs 非凸：ASCII 曲面可视化")
    print("=" * 65)
    print()
    print("  凸函数: f(x) = x^2")
    print()

    for y_level in range(10, -1, -1):
        threshold = y_level * 2.5
        line = "  "
        for x_step in range(-20, 21):
            x = x_step * 0.25
            val = x ** 2
            if abs(val - threshold) < 1.3:
                line += "*"
            else:
                line += " "
        print(line)

    print("  " + "-" * 41)
    print("  " + " " * 18 + "x=0")
    print("  一个山谷。梯度下降总能找到谷底。")

    print()
    print("  非凸函数: f(x) = sin(3x) + 0.1*x^2")
    print()

    for y_level in range(10, -1, -1):
        threshold = -1.0 + y_level * 0.4
        line = "  "
        for x_step in range(-25, 26):
            x = x_step * 0.2
            val = math.sin(3 * x) + 0.1 * x ** 2
            if abs(val - threshold) < 0.25:
                line += "*"
            else:
                line += " "
        print(line)

    print("  " + "-" * 51)
    print("  多个山谷。梯度下降可能被困住。")


def demo_duality_intuition():
    print()
    print()
    print("=" * 65)
    print("  对偶理论：原问题与对偶问题")
    print("=" * 65)
    print()
    print("  原问题: 最小化 x^2 + y^2，满足 x + y >= 1")
    print("  改写约束为: -(x + y - 1) <= 0")
    print()
    print("  拉格朗日函数: L = x^2 + y^2 + λ(1 - x - y)")
    print("  ∂L/∂x = 2x - λ = 0  →  x = λ/2")
    print("  ∂L/∂y = 2y - λ = 0  →  y = λ/2")
    print()
    print("  对偶函数:")
    print("    d(λ) = min_{x,y} [x^2 + y^2 + λ(1 - x - y)]")
    print("         = (λ/2)^2 + (λ/2)^2 + λ(1 - λ)")
    print("         = λ - λ^2/2")
    print()
    print("  对偶问题: 最大化 λ - λ^2/2，满足 λ >= 0")
    print("  d'(λ) = 1 - λ = 0  →  λ* = 1")
    print()

    lam_star = 1.0
    x_star = lam_star / 2
    y_star = lam_star / 2
    primal_val = x_star ** 2 + y_star ** 2
    dual_val = lam_star - lam_star ** 2 / 2

    print(f"  原问题解: x = {x_star}, y = {y_star}")
    print(f"  原问题目标值: {primal_val}")
    print(f"  对偶目标值:   {dual_val}")
    print(f"  强对偶性:     原问题 = 对偶 = {primal_val}")
    print(f"  约束: x + y = {x_star + y_star} >= 1（紧约束）")
    print(f"  互补松弛性: λ * (1 - x - y) = {lam_star * (1 - x_star - y_star)}")


def print_summary():
    print()
    print()
    print("=" * 65)
    print("  总结")
    print("=" * 65)
    print()
    print("  1. 凸函数只有一个山谷。任何局部最小值都是全局最小值。")
    print("  2. Hessian 矩阵编码曲率信息。半正定 Hessian = 凸函数。")
    print("  3. 牛顿法利用曲率信息实现更快收敛。")
    print("  4. 拉格朗日乘子法处理等式约束。")
    print("  5. KKT 条件处理不等式约束。")
    print("  6. L1 正则化 = 菱形约束 = 稀疏性。")
    print("  7. L2 正则化 = 圆形约束 = 权重收缩。")
    print("  8. 对偶理论将原问题转化为有时更易求解的对偶问题。")
    print("  9. 神经网络是非凸的，但过参数化和随机噪声使 SGD 依然有效。")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    demo_convexity_checker()
    demo_hessian_analysis()
    demo_newtons_method()
    demo_condition_number_effect()
    demo_lagrange_multipliers()
    demo_regularization_geometry()
    demo_first_vs_second_order()
    demo_convex_vs_nonconvex_landscape()
    demo_duality_intuition()
    print_summary()
