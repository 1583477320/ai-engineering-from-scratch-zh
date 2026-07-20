"""检查点保存与恢复从零实现。

运行：python3 code/main.py
"""
from __future__ import annotations
import hashlib, json, os, random, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

CHECKPOINT_SCHEMA = "ckpt.v1"

HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "outputs"


@dataclass
class TrainState:
    step: int
    epoch: int
    batch_in_epoch: int
    losses: List[float] = field(default_factory=list)


def make_model(in_dim=32, hidden=48, out_dim=8) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def capture_rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state().tolist()}
    if torch.cuda.is_available():
        state["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if state.get("python"):
        random.setstate(_to_tuple(state["python"]))
    if state.get("numpy"):
        np.random.set_state(_to_tuple(state["numpy"]))
    if state.get("torch_cpu"):
        torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state["torch_cuda"]])


def _to_tuple(obj):
    if isinstance(obj, list):
        return tuple(_to_tuple(x) for x in obj)
    return obj


def atomic_save(payload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent),
                                      prefix=path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def save_checkpoint(model, optimizer, scheduler, state: TrainState, out_path: Path,
                    schema=CHECKPOINT_SCHEMA) -> Dict[str, Any]:
    payload = {"schema": schema,
               "model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(),
               "state": {"step": state.step, "epoch": state.epoch,
                         "batch_in_epoch": state.batch_in_epoch, "losses": list(state.losses)},
               "rng": capture_rng_state(), "wall_saved_at": time.time()}
    atomic_save(payload, out_path)
    return payload


def load_checkpoint(path: Path, model, optimizer, scheduler) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"].startswith("ckpt"), f"未知 schema: {payload['schema']}"
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    s = payload["state"]
    return TrainState(step=int(s["step"]), epoch=int(s["epoch"]),
                      batch_in_epoch=int(s["batch_in_epoch"]), losses=list(s["losses"]))


def run_resume_demo(*, total_steps=30, interrupt_at=12, in_dim=16, hidden=24, out_dim=4,
                     batch_size=4, batches_per_epoch=5, seed=11, ckpt_dir) -> dict:
    # 完整训练
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m1 = make_model(in_dim, hidden, out_dim)
    o1 = torch.optim.AdamW(m1.parameters(), lr=0.01)
    s1 = torch.optim.lr_scheduler.CosineAnnealingLR(o1, T_max=total_steps)
    state1 = TrainState(step=0, epoch=0, batch_in_epoch=0)

    def train_until(model, opt, sched, state, stop_step):
        loss_fn = nn.CrossEntropyLoss()
        while state.step < stop_step:
            gen = torch.Generator(); gen.manual_seed(12345 + state.epoch)
            # 快进到当前批次
            for _ in range(state.batch_in_epoch):
                torch.randn(batch_size, in_dim, generator=gen)
                torch.randint(0, out_dim, (batch_size,), generator=gen)
            while state.batch_in_epoch < batches_per_epoch and state.step < stop_step:
                x = torch.randn(batch_size, in_dim, generator=gen)
                y = torch.randint(0, out_dim, (batch_size,), generator=gen)
                opt.zero_grad()
                loss_fn(model(x), y).backward(); opt.step(); sched.step()
                state.losses.append(float(loss.detach().item()))
                state.step += 1; state.batch_in_epoch += 1
            if state.batch_in_epoch >= batches_per_epoch:
                state.epoch += 1; state.batch_in_epoch = 0

    train_until(m1, o1, s1, state1, interrupt_at)
    save_checkpoint(m1, o1, s1, state1, ckpt_dir / "ckpt.pt")
    train_until(m1, o1, s1, state1, total_steps)
    full_losses = list(state1.losses)

    # 恢复训练
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m2 = make_model(in_dim, hidden, out_dim)
    o2 = torch.optim.AdamW(m2.parameters(), lr=0.01)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=total_steps)
    loaded = load_checkpoint(ckpt_dir / "ckpt.pt", m2, o2, s2)
    train_until(m2, o2, s2, loaded, total_steps)
    resumed_losses = list(loaded.losses)

    max_diff = max(abs(full_losses[i] - resumed_losses[i])
                   for i in range(interrupt_at, total_steps)) if total_steps > interrupt_at else 0.0
    return {"max_loss_diff": max_diff, "full_losses": full_losses, "resumed_losses": resumed_losses}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ckpt-") as scratch:
        ckpt = Path(scratch) / "ckpt"
        result = run_resume_demo(ckpt_dir=ckpt)
        print(f"恢复后最大损失差异: {result['max_loss_diff']:.6f}")
        assert result["max_loss_diff"] < 1e-4, "恢复后损失漂移！"
        print("✓ 单文件检查点恢复测试通过")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "resume-demo.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
