"""带检查点的回滚工作流——纯标准库。

四种场景：
  1. 干净运行
  2. 提交中途崩溃后重试 → 幂等防止双重执行
  3. 前置条件失败 → 工作流中止
  4. 验证失败 → 回滚触发

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass


# ── 模拟数据库 ──────────────────────────────────────────

DB = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}


def persist_transfer(txid, from_acct, to_acct, amount):
    DB[f"balance_{from_acct}"] -= amount
    DB[f"balance_{to_acct}"] += amount
    DB["last_transfer_id"] = txid


def rollback_transfer(txid, from_acct, to_acct, amount, prior_last):
    """补偿事务：恢复余额和先前的转账 ID。"""
    DB[f"balance_{from_acct}"] += amount
    DB[f"balance_{to_acct}"] -= amount
    DB["last_transfer_id"] = prior_last


# ── 检查点存储 ──────────────────────────────────────────

@dataclass
class Checkpoint:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def save(self, k: str, v: dict) -> None:
        data = self.load()
        data[k] = v
        # 原子性写入：临时文件 → fsync → 重命名
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)


# ── 工作流 ────────────────────────────────────────────────

def key(txid: str) -> str:
    return hashlib.sha256(txid.encode()).hexdigest()[:12]


def run_transfer(cp, txid, from_acct, to_acct, amount, min_balance,
                 inject_crash_after_execute=False,
                 inject_verify_fail=False) -> str:
    k = key(txid)
    record = cp.load().get(k, {"status": "new"})

    # 幂等性：任何终态都短路
    terminal = {
        "committed": "idempotent-skip",
        "verified": "ok",
        "rolled-back": "verify-fail-rolled-back",
        "aborted-precondition": "aborted-precondition",
    }
    if record["status"] in terminal:
        return terminal[record["status"]]

    # 前置条件检查
    if DB[f"balance_{from_acct}"] - amount < min_balance:
        cp.save(k, {"status": "aborted-precondition", "txid": txid})
        return "aborted-precondition"

    # 记录意图（崩溃时保留）
    prior_last = DB["last_transfer_id"]
    cp.save(k, {"status": "committed", "txid": txid,
                "from_acct": from_acct, "to_acct": to_acct,
                "amount": amount, "prior_last_transfer_id": prior_last})

    # 执行副作用
    persist_transfer(txid, from_acct, to_acct, amount)
    if inject_crash_after_execute:
        raise RuntimeError("simulated crash after execute")

    # 提交后验证
    if inject_verify_fail or DB["last_transfer_id"] != txid:
        rollback_transfer(txid, from_acct, to_acct, amount, prior_last)
        cp.save(k, {"status": "rolled-back", "txid": txid})
        return "verify-fail-rolled-back"

    cp.save(k, {"status": "verified", "txid": txid})
    return "ok"


# ── 主函数 ────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("检查点与回滚（阶段 15，第 16 课）")
    print("=" * 80)

    tmp = tempfile.mkdtemp()

    print("\n场景 1：干净运行")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp1.json"))
    out = run_transfer(cp, "tx-001", "A", "B", 100, min_balance=200)
    print(f"  result={out}  DB={DB}")

    print("\n场景 2：提交中途崩溃，重试（幂等捕获）")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp2.json"))
    try:
        run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200,
                     inject_crash_after_execute=True)
    except RuntimeError as e:
        print(f"  crash: {e}")
    out = run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200)
    print(f"  retry result={out}  DB={DB}")

    print("\n场景 3：前置条件失败（余额低于最低值）")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp3.json"))
    out = run_transfer(cp, "tx-003", "A", "B", 10_000, min_balance=200)
    print(f"  result={out}  DB={DB}")

    print("\n场景 4：验证失败 → 回滚")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp4.json"))
    balances_before = dict(DB)
    out = run_transfer(cp, "tx-004", "A", "B", 100, min_balance=200,
                       inject_verify_fail=True)
    balances_after = dict(DB)
    print(f"  result={out}  balances_before_after_equal="
          f"{balances_before == balances_after}")

    print()
    print("=" * 80)
    print("要点：幂等键 + 前置条件 + 验证 + 回滚")
    print("-" * 80)
    print("  四件，不是一件。每件捕获不同的失败类别：")
    print("  幂等键 → 崩溃时重试安全")
    print("  前置条件 → 批准和提交之间的状态漂移")
    print("  验证 → 副作用在我们以为发生时没有发生")
    print("  回滚 → 已知坏状态恢复或告警")
    print("  Article 14 操作解读：检查点可查询，回滚经过演练，")
    print("  审计跟踪在部署后存活。")


if __name__ == "__main__":
    main()
