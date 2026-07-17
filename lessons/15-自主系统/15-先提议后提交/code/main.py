"""先提议后提交 HITL 状态机——纯标准库。

四阶段：
  1. 提议：智能体将提议动作与幂等键持久化
  2. 呈现：审查者看到元数据（意图、血缘、爆炸半径、回滚计划）
  3. 提交：需要正面确认；幂等
  4. 验证：提交后重新读取目标资源

三个演示：
  - 干净的批准流程
  - 瞬态故障后的重试 → 幂等防止双重执行
  - 橡皮戳 vs 挑战-回应清单

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field


@dataclass
class Proposal:
    thread_id: str
    action: str
    payload: dict
    intent: str
    lineage: str
    blast_radius: str
    rollback: str

    def key(self) -> str:
        """基于 thread_id + action + payload 生成唯一幂等键。"""
        sig = json.dumps({"t": self.thread_id, "a": self.action,
                          "p": self.payload}, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]


@dataclass
class Store:
    """持久提议存储——JSON 文件。"""
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def all(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def save(self, key: str, record: dict) -> None:
        data = self.all()
        data[key] = record
        with open(self.path, "w") as f:
            json.dump(data, f)


# ── 模拟副作用跟踪器 ──────────────────────────────────────

SIDE_EFFECTS: list[str] = []

def execute(proposal: Proposal) -> bool:
    SIDE_EFFECTS.append(f"{proposal.action}:{json.dumps(proposal.payload)}")
    return True

def verify(proposal: Proposal) -> bool:
    """验证副作用确实发生（重新读取目标资源）。"""
    needle = f"{proposal.action}:{json.dumps(proposal.payload)}"
    return needle in SIDE_EFFECTS


# ── 四阶段状态机 ──────────────────────────────────────────

def propose(store: Store, p: Proposal) -> str:
    """阶段 1：提议——持久化动作和元数据。"""
    k = p.key()
    existing = store.all().get(k)
    if existing:
        print(f"  [propose] 幂等：记录 {k} 已存在 (status={existing['status']})")
        return k
    record = {"status": "waiting", **vars(p)}
    store.save(k, record)
    print(f"  [propose] 记录 {k} 已存储，等待审查")
    return k


def surface(store: Store, k: str) -> None:
    """阶段 2：呈现——审查者看到完整元数据。"""
    r = store.all()[k]
    print(f"  [surface] 提议 {k}")
    for name in ("intent", "lineage", "blast_radius", "rollback"):
        print(f"    {name:<14} {r[name]}")


def rubber_stamp_approve(store: Store, k: str) -> bool:
    """橡皮戳批准——无审查。"""
    rec = store.all()[k]
    rec["status"] = "approved"
    rec["ack_mode"] = "rubber_stamp"
    store.save(k, rec)
    print("  [approve:rubber-stamp] 点击批准（无清单）")
    return True


def checklist_approve(store: Store, k: str,
                      understood: bool, verified: bool,
                      rollback_ready: bool) -> bool:
    """挑战-回应批准——三个问题必须正面回答。"""
    if not (understood and verified and rollback_ready):
        print("  [approve:checklist] 拒绝（清单不完整）")
        return False
    rec = store.all()[k]
    rec["status"] = "approved"
    rec["ack_mode"] = "challenge_response"
    store.save(k, rec)
    print("  [approve:checklist] 批准（三个检查全部通过）")
    return True


def commit(store: Store, k: str) -> bool:
    """阶段 3：提交——执行副作用；幂等。"""
    data = store.all()
    rec = data[k]
    if rec["status"] == "committed":
        print(f"  [commit] 幂等：{k} 已提交，不重新执行")
        return True
    if rec["status"] != "approved":
        print(f"  [commit] 拒绝：{k} status={rec['status']}")
        return False
    p = Proposal(thread_id=rec["thread_id"], action=rec["action"],
                 payload=rec["payload"], intent=rec["intent"],
                 lineage=rec["lineage"], blast_radius=rec["blast_radius"],
                 rollback=rec["rollback"])
    execute(p)
    rec["status"] = "committed"
    store.save(k, rec)
    print(f"  [commit] 已执行; verify={verify(p)}")
    return True


# ── 演示 ──────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("先提议后提交 HITL（阶段 15，第 15 课）")
    print("=" * 80)
    tmp = tempfile.mkdtemp()
    store = Store(os.path.join(tmp, "proposals.json"))

    # ── 提议 ──────────────────────────────────────────────

    p = Proposal(
        thread_id="t-001", action="email.send",
        payload={"to": "team@example.com", "subject": "release"},
        intent="通知团队 v1.2 发布",
        lineage="发布说明页面 /releases/1.2",
        blast_radius="37 个收件人；发错邮件 = 外部尴尬",
        rollback="无带内回滚；跟进一封更正邮件",
    )

    print("\n演示 1：干净的挑战-回应批准流程")
    print("-" * 80)
    k = propose(store, p)
    surface(store, k)
    checklist_approve(store, k, understood=True, verified=True, rollback_ready=True)
    commit(store, k)

    print("\n演示 2：批准后重试；幂等防止双重执行")
    print("-" * 80)
    initial = len(SIDE_EFFECTS)
    commit(store, k)  # 重试
    commit(store, k)  # 重试
    print(f"  重试后总副作用：{len(SIDE_EFFECTS)} (之前 {initial}) -> 幂等")

    print("\n演示 3：橡皮戳 vs 挑战-回应")
    print("-" * 80)
    p2 = Proposal(
        thread_id="t-002", action="db.update",
        payload={"row": 42, "col": "status", "val": "closed"},
        intent="关闭过期 issue", lineage="过期 issue 仪表板扫描",
        blast_radius="一行数据库；1 小时备份窗口内可逆", rollback="从夜间备份恢复",
    )
    k2 = propose(store, p2)
    rubber_stamp_approve(store, k2)
    commit(store, k2)

    p3 = Proposal(
        thread_id="t-003", action="db.drop_table",
        payload={"table": "old_users"},
        intent="删除未使用的表（按清理运行手册）",
        lineage="运行手册 #RB-17",
        blast_radius="破坏性；420k 行被删除；24 小时内不可逆",
        rollback="从每周备份恢复；最多丢失 6 天数据",
    )
    k3 = propose(store, p3)
    ok = checklist_approve(store, k3, understood=True, verified=True, rollback_ready=False)
    if not ok:
        commit(store, k3)  # 拒绝 → commit 拒绝

    print()
    print("=" * 80)
    print("要点：使结构化审查成为最简路径")
    print("-" * 80)
    print("  幂等键防止重试时的双重执行。")
    print("  持久化使批准可以迟到两天仍然生效。")
    print("  挑战-回应清单是文档化的橡皮戳缓解；EU AI Act Article 14 要求。")
    print("  提交后验证关闭'认为已发生'的失败类别。")


if __name__ == "__main__":
    main()
