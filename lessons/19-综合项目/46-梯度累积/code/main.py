"""梯度累积从零实现。

运行：python3 code/main.py
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch import nn

HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "outputs"


@dataclass
class CurvePoint:
    effective_batch: int
    accum_steps: int
    micro_batch: int
    avg_loss: float
    samples_per_sec: float
    median_step_ms: float
    sync_calls: int
    steps: int


def make_model(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def equivalence_check(in_dim=32, hidden=48, out_dim=8, big_batch=16, accum_steps=4, lr=0.1, seed=7) -> dict:
    """全批次 vs 累积微批次的等价性检查。"""
    micro = big_batch // accum_steps
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(big_batch, in_dim, generator=gen)
    y = torch.randint(0, out_dim, (big_batch,), generator=gen)

    torch.manual_seed(seed)
    mf = make_model(in_dim, hidden, out_dim)
    opt_f = torch.optim.SGD(mf.parameters(), lr=lr)
    zero_grads(mf)
    nn.CrossEntropyLoss()(mf(x), y).backward()
    full_grads = [p.grad.detach().clone() for p in mf.parameters()]

    torch.manual_seed(seed)
    ma = make_model(in_dim, hidden, out_dim)
    opt_a = torch.optim.SGD(ma.parameters(), lr=lr)
    zero_grads(ma)
    for cx, cy in zip(torch.split(x, micro), torch.split(y, micro)):
        (nn.CrossEntropyLoss()(ma(cx), cy) / accum_steps).backward()
    accum_grads = [p.grad.detach().clone() for p in ma.parameters()]

    grad_diffs = [float((a - b).abs().max().item()) for a, b in zip(full_grads, accum_grads)]
    return {"max_grad_diff": max(grad_diffs)}


def sweep(micro_batch=4, accum_grid=None, num_steps=25, lr=0.05, seed=0, in_dim=64, hidden=128, out_dim=16) -> list[CurvePoint]:
    if accum_grid is None:
        accum_grid = [1, 2, 4, 8, 16]
    points = []
    for a in accum_grid:
        eff = micro_batch * a
        torch.manual_seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed)
        model = make_model(in_dim, hidden, out_dim)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        losses = []
        times = []
        sync = [0]
        total_s = 0
        start = time.perf_counter()
        for _ in range(num_steps):
            t0 = time.perf_counter()
            zero_grads(model)
            batch_loss = 0.0
            for __ in range(a):
                xb = torch.randn(micro_batch, in_dim, generator=gen)
                yb = torch.randint(0, out_dim, (micro_batch,), generator=gen)
                ls = nn.CrossEntropyLoss()(model(xb), yb) / a
                ls.backward()
                batch_loss += float(ls.detach().item()) * a
                sync[0] += 1
            opt.step()
            losses.append(batch_loss / a)
            times.append((time.perf_counter() - t0) * 1000)
            total_s += eff
        times.sort()
        points.append(CurvePoint(eff, a, micro_batch, sum(losses)/len(losses),
                                 total_s/max(time.perf_counter()-start, 1e-6),
                                 times[len(times)//2], sync[0], num_steps))
    return points


def main() -> int:
    eq = equivalence_check()
    print(f"等价性检查: max_grad_diff = {eq['max_grad_diff']:.6f}")
    assert eq["max_grad_diff"] < 1e-4, f"梯度分歧: {eq['max_grad_diff']}"
    points = sweep()
    print(f"{'有效批次':>10}  {'累积步':>5}  {'微批次':>5}  {'样本/秒':>10}  {'中位ms':>10}  {'损失':>8}")
    for p in points:
        print(f"{p.effective_batch:>10}  {p.accum_steps:>5}  {p.micro_batch:>5}  "
              f"{p.samples_per_sec:>10.1f}  {p.median_step_ms:>10.2f}  {p.avg_loss:>8.4f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "accum-curve.json"
    out_path.write_text(json.dumps({"points": [p.__dict__ for p in points]}, indent=2))
    print(f"写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
