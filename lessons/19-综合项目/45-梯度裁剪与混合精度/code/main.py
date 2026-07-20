"""梯度裁剪与混合精度训练步骤。

运行：python3 code/main.py
"""
from __future__ import annotations
import csv, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("需要 PyTorch。安装：pip install torch") from exc

DEFAULT_MAX_NORM = 1.0
DEFAULT_DEVICE = "cpu"


@dataclass
class StepLog:
    step: int
    lr: float
    grad_l2_pre_clip: float
    grad_l2_post_clip: float
    loss: float
    skipped: bool
    skip_reason: str
    scaler_scale: float

    def to_csv_row(self) -> list[str]:
        return [str(self.step), f"{self.lr:.10f}", f"{self.grad_l2_pre_clip:.10f}",
                f"{self.grad_l2_post_clip:.10f}", f"{self.loss:.10f}",
                "1" if self.skipped else "0", self.skip_reason, f"{self.scaler_scale:.6f}"]


def has_non_finite_grad(parameters: Iterable[torch.nn.Parameter]) -> bool:
    for param in parameters:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad.detach()).all().item():
            return True
    return False


def compute_global_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = sum(float(p.grad.detach().pow(2).sum().item())
                for p in parameters if p.grad is not None)
    return math.sqrt(total)


def clip_global_l2_norm(parameters: list[torch.nn.Parameter], max_norm: float) -> tuple[float, float]:
    if max_norm <= 0:
        raise ValueError("max_norm 必须为正数")
    pre_clip = compute_global_l2_norm(parameters)
    if not math.isfinite(pre_clip) or pre_clip <= max_norm:
        return pre_clip, pre_clip
    scale = max_norm / (pre_clip + 1e-12)
    for param in parameters:
        if param.grad is not None:
            param.grad.detach().mul_(scale)
    return pre_clip, max_norm


class AmpTrainState:
    def __init__(self, model: nn.Module, lr: float = 1e-2, max_norm: float = DEFAULT_MAX_NORM,
                 device_type: str = DEFAULT_DEVICE, weight_decay: float = 0.01):
        self.model = model
        self.max_norm = max_norm
        self.device_type = device_type
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
        self.global_step = 0
        self._log: list[StepLog] = []
        self._loss_fn = nn.functional.mse_loss

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, inputs: torch.Tensor, targets: torch.Tensor,
             gradient_corruptor: Callable[[nn.Module], None] | None = None) -> StepLog:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device_type):
            loss = self._loss_fn(self.model(inputs), targets)
        if not torch.isfinite(loss).all().item():
            return self._record_skip(float(loss.detach().cpu().item()), "非有限损失", 0.0, update_scaler=False)
        self.scaler.scale(loss).backward()
        if gradient_corruptor is not None:
            gradient_corruptor(self.model)
        self.scaler.unscale_(self.optimizer)
        if has_non_finite_grad(self.model.parameters()):
            self.scaler.update()
            return self._record_skip(float(loss.detach().item()), "非有限梯度", float("inf"))
        pre_clip, post_clip = clip_global_l2_norm(list(self.model.parameters()), self.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        record = StepLog(self.global_step, self._current_lr(), pre_clip, post_clip,
                         float(loss.detach().item()), False, "", float(self.scaler.get_scale()))
        self._log.append(record)
        self.global_step += 1
        return record

    def _record_skip(self, loss_value: float, reason: str, pre_clip: float, update_scaler: bool = True) -> StepLog:
        record = StepLog(self.global_step, self._current_lr(), pre_clip, pre_clip,
                         loss_value, True, reason, float(self.scaler.get_scale()))
        self._log.append(record)
        self.global_step += 1
        if update_scaler:
            self.scaler.update()
        return record

    @property
    def log(self) -> list[StepLog]:
        return list(self._log)

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self._log if r.skipped)


def build_toy_model(in_dim: int = 16, out_dim: int = 4, seed: int = 7):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, out_dim))
    return model, torch.randn(8, in_dim), torch.randn(8, out_dim)


def inject_inf_into_first_grad(model: nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data[...] = float("inf")
            return


def run_demo() -> int:
    model, inputs, targets = build_toy_model()
    state = AmpTrainState(model=model, lr=1e-2, max_norm=1.0, device_type="cpu")
    for index in range(20):
        corruptor = inject_inf_into_first_grad if index == 5 else None
        record = state.step(inputs, targets, gradient_corruptor=corruptor)
        marker = "跳过" if record.skipped else "步骤"
        print(f"{marker} step={record.step:>3} lr={record.lr:.6f} "
              f"pre_clip={record.grad_l2_pre_clip:>10.6f} post_clip={record.grad_l2_post_clip:>10.6f} "
              f"loss={record.loss:.6f} scale={record.scaler_scale:.1f} reason={record.skip_reason or '-'}")
    print(f"\n跳过次数={state.skip_count}")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
