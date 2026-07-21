"""AI 调试工具集——张量检查、NaN 检测、设备检查。"""
import torch, time, math

def debug_print(name, tensor):
    """打印张量的完整状态。"""
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")


def detect_nan(model, loss, step):
    """检测 NaN 损失和梯度。"""
    if torch.isnan(loss):
        print(f"NaN 损失！步数: {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN 梯度: {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf 梯度: {name}")
        return True
    return False


def check_devices(model, *tensors):
    """检查模型和张量的设备一致性。"""
    model_device = next(model.parameters()).device
    print(f"模型设备: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"  警告: 张量 {i} 在 {t.device}，模型在 {model_device}")


class Timer:
    def __init__(self, name=""): self.name = name
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *args): print(f"[{self.name}] {time.perf_counter()-self.start:.4f}s")


def main():
    print("=== AI 调试工具集 ===\n")

    # 1. 张量检查
    t = torch.randn(32, 64)
    debug_print("输入张量", t)

    # 2. NaN 检测
    print("\nNaN 检测:")
    model = nn.Linear(64, 10)
    loss = torch.tensor(float("nan"))
    detect_nan(model, loss, 0)

    # 3. 设备检查
    print("\n设备检查:")
    x = torch.randn(16, 64)
    check_devices(model, x)

    # 4. 计时
    print("\n计时示例:")
    with Timer("矩阵乘法"):
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        _ = a @ b

    print("\n✓ 调试工具集就绪")
    return 0

if __name__ == "__main__":
    sys.exit(main())
