"""分布式数据并行从零实现（gloo 后端）。

运行：python3 code/main.py
"""
from __future__ import annotations
import json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

OUT_DIR = Path(__file__).parent.parent / "outputs"


@dataclass
class RankResult:
    rank: int; world_size: int; backend: str
    final_loss: float; pre_param_sum: float; post_param_sum: float
    grad_norm_after_all_reduce: float
    fsdp_round_trip_ok: bool


def make_model(in_dim=32, hidden=16, out_dim=4) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def init_pg(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def broadcast_module(module: nn.Module, src=0):
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=src)


def all_reduce_grads_(module: nn.Module, ws: int) -> float:
    sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
        p.grad.data.div_(ws)
        sq += float(p.grad.data.pow(2).sum().item())
    return sq ** 0.5


class MinimalDDP(nn.Module):
    def __init__(self, module: nn.Module, world_size: int):
        super().__init__()
        self.module = module
        self.world_size = world_size
        if dist.is_initialized() and world_size > 1:
            broadcast_module(self.module, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self) -> float:
        if not dist.is_initialized() or self.world_size == 1:
            sq = sum(float(p.grad.data.pow(2).sum().item())
                     for p in self.module.parameters() if p.grad is not None)
            return sq ** 0.5
        return all_reduce_grads_(self.module, self.world_size)


def fsdp_round_trip(module: nn.Module, ws: int, rank: int) -> bool:
    for p in module.parameters():
        full = p.data.detach().clone().flatten()
        n = full.numel()
        per = (n + ws - 1) // ws
        padded = torch.cat([full, torch.zeros(per * ws - n, dtype=full.dtype)])
        mine = padded[rank * per:(rank + 1) * per].clone()
        buf = [torch.empty(per, dtype=full.dtype) for _ in range(ws)]
        dist.all_gather(buf, mine)
        rebuilt = torch.cat(buf)[:n].view_as(p.data)
        if not torch.allclose(rebuilt, p.data):
            return False
    return True


def rank_main(rank, ws, port, q, in_dim, hidden, out_dim, bs, steps, lr, seed):
    try:
        init_pg(rank, ws, port)
        torch.manual_seed(seed)
        ddp = MinimalDDP(make_model(in_dim, hidden, out_dim), ws)
        opt = torch.optim.SGD(ddp.parameters(), lr=lr)
        pre = sum(float(p.data.sum().item()) for p in ddp.parameters())
        loss_val = 0.0
        for _ in range(steps):
            x = torch.randn(bs, in_dim)
            y = torch.randint(0, out_dim, (bs,))
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(ddp(x), y)
            loss.backward()
            ddp.sync_grads()
            opt.step()
            loss_val = float(loss.detach().item())
        post = sum(float(p.data.sum().item()) for p in ddp.parameters())
        fsdp_ok = fsdp_round_trip(ddp.module, ws, rank)
        q.put((rank, RankResult(rank, ws, "gloo", loss_val, pre, post, 0.0, fsdp_ok)))
    except Exception as e:
        q.put((rank, str(e)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    ws = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 12355 + int(time.time()) % 1000
    procs = [ctx.Process(target=rank_main, args=(r, ws, port, q, 32, 16, 4, 8, 6, 0.05, 0))
             for r in range(ws)]
    for p in procs:
        p.start()
    results: Dict[int, RankResult] = {}
    deadline = time.time() + 60
    while len(results) < ws and time.time() < deadline:
        try:
            r, v = q.get(timeout=1.0)
            results[r] = v
        except:
            continue
    for p in procs:
        p.join(1.0)
    if len(results) < ws:
        print(f"仅收到 {len(results)}/{ws} 个结果")
        return 1
    for r, v in results.items():
        if isinstance(v, str):
            print(f"设备 {r} 失败: {v}")
            return 1
        print(f"设备 {r}: loss={v.final_loss:.4f} pre={v.pre_param_sum:.2f} post={v.post_param_sum:.2f} fsdp={v.fsdp_round_trip_ok}")
    sums = [v.post_param_sum for v in results.values()]
    spread = max(sums) - min(sums)
    fsdp_all = all(v.fsdp_round_trip_ok for v in results.values())
    print(f"\n参数跨度: {spread:.6f}  FSDP 全部通过: {fsdp_all}")
    assert spread < 1e-3, f"参数跨设备分歧: {spread}"
    assert fsdp_all, "FSDP 往返测试失败"
    print("✓ 分布式训练演示完成")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ddp-demo.json").write_text(json.dumps(
        {"schema": "ddp-demo.v1", "param_sum_spread": spread,
         "fsdp_all_ok": fsdp_all}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
