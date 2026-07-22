# main.py -- 神经网络调试工具集
# 依赖：torch>=2.0, numpy
# 安装：pip install torch numpy
# 对应课程：阶段 03 · 13（神经网络调试）

import math
import copy
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 第 1 部分：NetworkDebugger -- 激活值与梯度监控器
# ============================================================

class NetworkDebugger:
    """通过 PyTorch Hook 机制监控模型每层的激活值和梯度统计量。

    每次前向传播自动记录每层输出的均值、标准差、零值比例；
    每次反向传播自动记录梯度的均值、绝对值均值、最大值。
    """

    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """为模型中的 Linear、Conv2d、ReLU、LeakyReLU 层注册前后向 Hook。"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                # 前向 Hook：记录激活值统计量
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                # 反向 Hook：记录梯度统计量
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        """创建激活值统计量的 Hook 工厂函数。"""
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        """创建梯度统计量的 Hook 工厂函数。"""
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        """记录一个训练步的损失值。"""
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        """检查损失曲线的健康状态，返回诊断标签。"""
        if len(self.loss_history) < 2:
            return "数据不足，无法诊断"

        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NaN 或 Inf -- 数值溢出"

        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "损失未下降 -- 优化失败"

        if len(recent) >= 5:
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            avg_diff = sum(diffs) / len(diffs) + 1e-10
            if max(diffs) - min(diffs) > 2 * abs(avg_diff):
                return "损失剧烈震荡 -- 学习率可能过大"

        return "正常"

    def check_activations(self):
        """检查每层激活值的健康状况。"""
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(f"死亡神经元: {name} 有 {stats['fraction_zero']:.0%} 的零值激活")
            if abs(stats["mean"]) > 10:
                issues.append(f"激活值爆炸: {name} 均值={stats['mean']:.2f}")
            if stats["std"] < 1e-6:
                issues.append(f"激活值坍缩: {name} 标准差={stats['std']:.2e}")
        return issues if issues else ["正常"]

    def check_gradients(self):
        """检查梯度流是否健康。"""
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(f"梯度消失: {name} 绝对均值={stats['abs_mean']:.2e}")
            if stats["abs_mean"] > 100:
                issues.append(f"梯度爆炸: {name} 绝对均值={stats['abs_mean']:.2e}")

        # 检查首尾层梯度比例
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / (last_mag + 1e-15) > 100:
                ratio = first_mag / (last_mag + 1e-15)
                issues.append(f"梯度比例异常: 首层/末层 = {ratio:.0f}x（梯度消失）")

        return issues if issues else ["正常"]

    def print_report(self):
        """打印完整的诊断报告。"""
        print("\n=== 网络调试器报告 ===")
        print(f"\n损失健康状态: {self.check_loss_health()}")
        if self.loss_history:
            print(f"  最近 5 步损失: {[f'{v:.4f}' for v in self.loss_history[-5:]]}")

        print("\n激活值诊断:")
        for item in self.check_activations():
            print(f"  {item}")

        print("\n梯度诊断:")
        for item in self.check_gradients():
            print(f"  {item}")

        print("\n各层激活值统计:")
        for name, stats in self.activation_stats.items():
            print(f"  {name}: 均值={stats['mean']:.4f} 标准差={stats['std']:.4f} "
                  f"零值比例={stats['fraction_zero']:.1%}")

        print("\n各层梯度统计:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}: 绝对均值={stats['abs_mean']:.2e} 最大值={stats['max']:.2e}")

    def remove_hooks(self):
        """移除所有注册的 Hook，释放资源。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# ============================================================
# 第 2 部分：过拟合单批次测试
# ============================================================

def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    """过拟合单批次测试 -- 深度学习中最重要的调试手段。

    取一个小批次（8-32 个样本），反复训练直到损失趋近于零。
    如果做不到，说明模型或训练循环有根本性缺陷。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    print("\n=== 过拟合单批次测试 ===")
    print(f"批次大小: {x_batch.shape[0]}, 训练步数: {steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        # 每 50 步打印一次进度
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                if output.shape[-1] == 1:
                    preds = (output > 0).float().squeeze()
                else:
                    preds = output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds == targets).float().mean().item()
            print(f"  步 {step:3d} | 损失: {loss.item():.6f} | 准确率: {acc:.1%}")

    # 判断是否通过
    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"\n  失败: 损失未收敛 ({final_loss:.4f})。模型或训练循环存在根本性错误。")
        return False
    print(f"\n  通过: 损失收敛到 {final_loss:.6f}")
    return True


# ============================================================
# 第 3 部分：学习率查找器
# ============================================================

def find_learning_rate(model, x_data, y_data, criterion,
                       start_lr=1e-7, end_lr=10, steps=100):
    """学习率查找器 -- 在一个轮次内指数增大学习率，记录损失变化。

    找到损失下降最快的区间，取其左侧一个数量级作为建议学习率。
    原始论文：Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)。
    """
    # 保存模型初始状态，扫描结束后恢复
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== 学习率查找器 ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        # 损失发散时停止
        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        # 按指数增长调整学习率
        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    # 恢复模型原始参数
    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  无法完成学习率扫描 -- 损失发散过快")
        return results

    # 找到损失最小的点及其前 10 个点作为建议学习率
    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(f"  扫描了 {len(results)} 步，范围 {start_lr:.0e} ~ {results[-1][0]:.0e}")
    print(f"  最小损失 {results[min_loss_idx][1]:.4f} 出现在 lr={results[min_loss_idx][0]:.2e}")
    print(f"  建议学习率: {suggested_lr:.2e}")

    return results


# ============================================================
# 第 4 部分：梯度检查器
# ============================================================

def _flat_to_multi_index(flat_idx, shape):
    """将扁平索引转换为多维索引，用于访问多维张量中的元素。"""
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    """梯度检查 -- 将反向传播计算的解析梯度与有限差分的数值梯度对比。

    如果两者差异过大（相对差 > 1e-3），说明反向传播实现有 bug。
    使用双精度（float64）减少浮点误差。
    """
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== 梯度检查 ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        # 计算解析梯度
        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        # 对每个参数的前 5 个元素做有限差分检查
        layer_max_diff = 0
        num_checks = min(5, param.numel())

        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            # f(w + eps)
            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            # f(w - eps)
            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            # 恢复原始值
            param.data[idx] = original

            # 数值梯度：中心差分公式
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            # 相对差：用两者的较大值做分母，避免数值太小导致假阳性
            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "正确" if layer_max_diff < 1e-5 else "可能有误"
        print(f"  {name}: 最大相对差={layer_max_diff:.2e} [{status}]")

    # 恢复模型精度
    model.float()

    print(f"\n  共检查 {checked} 个参数")
    if overall_max_diff < 1e-5:
        print("  通过: 梯度匹配（相对差 < 1e-5）")
    elif overall_max_diff < 1e-3:
        print("  警告: 存在小幅差异（1e-5 < 相对差 < 1e-3）")
    else:
        print("  失败: 梯度不匹配（相对差 > 1e-3），反向传播实现可能有误")
    return overall_max_diff


# ============================================================
# 第 5 部分：数据验证
# ============================================================

def validate_data(x, y, dataset_name="数据集"):
    """数据管道验证器 -- 检查输入数据和标签的基本健康状况。"""
    print(f"\n=== 数据验证: {dataset_name} ===")

    issues = []

    # 检查 NaN/Inf
    if torch.isnan(x).any():
        issues.append("输入数据中存在 NaN")
    if torch.isinf(x).any():
        issues.append("输入数据中存在 Inf")

    # 检查数据归一化
    mean_val = x.mean().item()
    std_val = x.std().item()
    print(f"  输入均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
    if abs(mean_val) > 2:
        issues.append(f"输入均值偏离 0 过远 ({mean_val:.4f})")
    if std_val < 0.01 or std_val > 10:
        issues.append(f"输入标准差异常 ({std_val:.4f})")

    # 检查标签
    print(f"  标签 dtype: {y.dtype}")
    print(f"  标签范围: [{y.min().item()}, {y.max().item()}]")
    print(f"  标签类别数: {y.unique().numel()}")
    print(f"  样本数: {len(y)}")

    if len(issues) == 0:
        print("  状态: 正常")
    else:
        for issue in issues:
            print(f"  警告: {issue}")

    return issues


# ============================================================
# 第 6 部分：可复现性设置
# ============================================================

def set_seed(seed=42):
    """设置所有随机种子以确保实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 以下设置可能影响性能，仅在调试时启用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为 {seed}")


# ============================================================
# 第 7 部分：演示 -- 故意制造错误并诊断
# ============================================================

def demo_broken_networks():
    """用故意制造的错误网络演示调试工具的诊断能力。"""
    set_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()
    criterion = nn.CrossEntropyLoss()

    # --- BUG 1: 学习率过大 ---
    print("\n" + "=" * 60)
    print("BUG 1: 学习率过大 (lr=10)")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    # --- BUG 2: 死亡 ReLU（错误初始化） ---
    print("\n" + "=" * 60)
    print("BUG 2: 错误初始化导致死亡 ReLU")
    print("=" * 60)
    model2 = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )
    # 故意用全负数初始化权重和偏置
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    # --- BUG 3: 遗漏 zero_grad（梯度累积） ---
    print("\n" + "=" * 60)
    print("BUG 3: 遗漏 zero_grad（梯度累积）")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        # 注意：这里故意没有调用 optimizer3.zero_grad()
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    # --- 正常网络（对照组） ---
    print("\n" + "=" * 60)
    print("正常网络: 正确配置（对照组）")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    # --- 过拟合单批次测试 ---
    print("\n" + "=" * 60)
    print("过拟合单批次测试（正常模型）")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    # --- 学习率查找器 ---
    print("\n" + "=" * 60)
    print("学习率查找器")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    # --- 梯度检查 ---
    print("\n" + "=" * 60)
    print("梯度检查（使用平滑模型 + MSE 损失）")
    print("=" * 60)
    torch.manual_seed(123)
    x_check = torch.randn(4, 3)
    y_check = torch.randn(4, 1)
    model_grad = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    gradient_check(model_grad, x_check, y_check, nn.MSELoss())

    # --- 数据验证 ---
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)
    validate_data(x, y, "模拟二分类数据")

    # --- 数据验证（有问题的数据） ---
    print("\n" + "=" * 60)
    print("数据验证（有问题的数据）")
    print("=" * 60)
    x_bad = torch.randn(64, 10) * 100  # 未归一化
    y_bad = torch.randn(64)  # 错误的 dtype
    validate_data(x_bad, y_bad, "有问题的模拟数据")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("神经网络调试工具集 -- 阶段 03 · 第 13 节")
    print("=" * 60)
    demo_broken_networks()
