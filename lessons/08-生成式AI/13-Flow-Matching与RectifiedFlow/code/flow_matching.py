# Flow Matching 从零实现
# 演示连续时间向量场学习、Flow Matching 训练和采样

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# 第 1 步：线性插值和目标向量场
# ============================================================================

def linear_interpolate(x_0, x_1, t):
    """
    线性插值：x_t = (1-t) * x_0 + t * x_1
    这是 Flow Matching 中最常用的概率路径。
    Args:
        x_0: 噪声 (B, D)
        x_1: 数据 (B, D)
        t: 时间步 (B,)，范围 [0, 1]
    Returns:
        x_t: 插值点 (B, D)
    """
    t = t.view(-1, 1)  # (B,) → (B, 1)
    return (1 - t) * x_0 + t * x_1


def optimal_transport_vector(x_0, x_1):
    """
    最优传输目标向量场。
    在 Flow Matching 中，从噪声到数据的最优路径就是直线——向量场 = x_1 - x_0。
    这是最优传输理论保证的：从标准正态分布到任意数据分布的最短路径就是直线。
    """
    return x_1 - x_0


# ============================================================================
# 第 2 步：Flow Matching 模型
# ============================================================================

class FlowMatchingMLP(nn.Module):
    """Flow Matching 的 MLP 模型——预测向量场 v_t(x)。"""

    def __init__(self, dim=784, hidden=512, num_layers=4):
        super().__init__()
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # 主干网络
        layers = [nn.Linear(dim + hidden, hidden), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Args:
            x: 位置 (B, D)
            t: 时间步 (B,)，范围 [0, 1]
        Returns:
            v_t: 向量场 (B, D) —— 在位置 x、时间 t 的速度
        """
        t_emb = self.time_embed(t.view(-1, 1).float())
        x_and_t = torch.cat([x, t_emb], dim=-1)
        return self.network(x_and_t)


# ============================================================================
# 第 3 步：训练循环
# ============================================================================

def train_flow_matching(model, dataloader, num_epochs=10, lr=1e-4, device="cpu"):
    """
    Flow Matching 训练循环。
    与 DDPM 训练非常相似——只是预测目标从"噪声"变成了"向量场"。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        model.train()

        for batch in dataloader:
            # 提取数据
            if isinstance(batch, (list, tuple)):
                x_1 = batch[0]
            else:
                x_1 = batch

            B = x_1.size(0)
            x_1 = x_1.to(device)
            x_1 = x_1.view(B, -1)  # 展平

            # 1. 采样噪声
            x_0 = torch.randn_like(x_1)

            # 2. 随机采样时间步 t ~ Uniform[0, 1]
            t = torch.rand(B, device=device)

            # 3. 线性插值：x_t = (1-t) * x_0 + t * x_1
            x_t = linear_interpolate(x_0, x_1, t)

            # 4. 目标向量场（最优传输方向）
            target = optimal_transport_vector(x_0, x_1)

            # 5. 模型预测向量场
            predicted = model(x_t, t)

            # 6. MSE 损失
            loss = F.mse_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return model


# ============================================================================
# 第 4 步：采样——ODE 求解
# ============================================================================

@torch.no_grad()
def sample_euler(model, num_samples=16, dim=784, num_steps=10, device="cpu"):
    """
    Euler 法求解——一阶 ODE 求解器。
    最简单、速度最快，但在少步数时精度较低。
    """
    model.eval()
    x_t = torch.randn(num_samples, dim, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        v_t = model(x_t, t)
        x_t = x_t + v_t * dt

    return x_t


@torch.no_grad()
def sample_heun(model, num_samples=16, dim=784, num_steps=10, device="cpu"):
    """
    Heun 法求解——二阶 ODE 求解器。
    比 Euler 更精确，适合少步数（2-5 步）采样。
    每步做两次前向评估——计算量倍增但步数可减半。
    """
    model.eval()
    x_t = torch.randn(num_samples, dim, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)

        # 预测步（Euler）
        v_t = model(x_t, t)
        x_pred = x_t + v_t * dt

        # 校正步（Heun）
        t_next = torch.full((num_samples,), min((i + 1) * dt, 1.0), device=device)
        v_pred = model(x_pred, t_next)

        # 取平均
        x_t = x_t + 0.5 * (v_t + v_pred) * dt

    return x_t


@torch.no_grad()
def sample_dpm_solver(model, num_samples=16, dim=784, num_steps=10, device="cpu"):
    """
    DPM-Solver 伪代码——高阶 ODE 求解器（简化版）。
    使用二阶 Runge-Kutta 风格求解。
    """
    model.eval()
    x_t = torch.randn(num_samples, dim, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_now = torch.full((num_samples,), i * dt, device=device)

        # t1 = t + dt/2, t2 = t + dt
        t_mid = torch.full((num_samples,), (i + 0.5) * dt, device=device)
        t_next = torch.full((num_samples,), (i + 1.0) * dt, device=device)

        # 中点法
        v0 = model(x_t, t_now)
        x_mid = x_t + 0.5 * dt * v0
        v_mid = model(x_mid, t_mid)
        x_t = x_t + dt * v_mid

    return x_t


# ============================================================================
# 第 5 步：Rectified Flow 的再配对
# ============================================================================

def rectified_flow_training_step(model, noise, data_samples, optimizer, device="cpu"):
    """
    Rectified Flow 的单步再配对训练。
    与 Flow Matching 相同，但使用模型自身采样配对来训练。
    """
    x_0 = noise.to(device)
    x_1 = data_samples.to(device)
    B = x_1.size(0)

    # 随机时间步
    t = torch.rand(B, device=device)

    # 插值
    x_t = linear_interpolate(x_0, x_1, t)

    # 目标向量场
    target = optimal_transport_vector(x_0, x_1)

    # 预测
    predicted = model(x_t, t)
    loss = F.mse_loss(predicted, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================================
# 主程序：演示
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 创建模型
    dim = 784  # 28×28 MNIST
    model = FlowMatchingMLP(dim=dim, hidden=256, num_layers=4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"待定参数量: {total_params:,}")

    # 模拟数据
    B = 32
    dummy_data = torch.randn(B, dim).to(device)

    # 训练一步
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x_0 = torch.randn_like(dummy_data)
    t = torch.rand(B, device=device)

    x_t = linear_interpolate(x_0, dummy_data, t)
    target = optimal_transport_vector(x_0, dummy_data)
    predicted = model(x_t, t)
    loss = F.mse_loss(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"训练一步后损失: {loss.item():.4f}")

    # 采样测试
    print("\n采样测试:")
    with torch.no_grad():
        samples_euler = sample_euler(model, num_samples=4, dim=dim,
                                      num_steps=5, device=device)
        print(f"  Euler (5步): {samples_euler.shape}")

        samples_heun = sample_heun(model, num_samples=4, dim=dim,
                                    num_steps=5, device=device)
        print(f"  Heun (5步): {samples_heun.shape}")

        samples_dpm = sample_dpm_solver(model, num_samples=4, dim=dim,
                                         num_steps=5, device=device)
        print(f"  DPM-Solver (5步): {samples_dpm.shape}")

    print("\n完成！")
