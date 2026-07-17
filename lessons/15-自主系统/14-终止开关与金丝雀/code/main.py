"""终止开关 + 断路器 + 金丝雀标记模拟器——纯标准库。

三种检测器：
  1. 终止开关：智能体外部的布尔值；每轮检查
  2. 断路器：N 个相同工具调用连续时跳闸
  3. 金丝雀标记：假凭据，读取触发告警

轨迹运行每种检测器。分类器和预算层来自第 10、13 课是正交的；
本课关于绊线。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── 外部状态（智能体无法写入）───────────────────────────

EXTERNAL_KILL_SWITCH = {"enabled": False}

# 注入工作空间的金丝雀标记
CANARY_FILES = {
    "~/.env.canary": "AWS_SECRET_ACCESS_KEY=canary-xxxxxxxxxxxxxxxx",
    "~/notes/fake-credentials.txt": "token = 'canary-sk-ABC-NEVER-USE'",
}


# ── 断路器 ──────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """在 N 个相同工具调用连续时跳闸。"""
    threshold: int = 5
    state: str = "closed"   # closed | open | half_open
    recent: list[str] = field(default_factory=list)

    def record(self, action_key: str) -> bool:
        """返回 True 如果动作可以继续。"""
        if self.state == "open":
            return False
        self.recent.append(action_key)
        if len(self.recent) >= self.threshold:
            tail = self.recent[-self.threshold:]
            if all(a == tail[0] for a in tail):
                self.state = "open"
                return False
        return True


# ── 金丝雀检测器 ────────────────────────────────────────

@dataclass
class Canary:
    triggered: list[tuple[int, str]] = field(default_factory=list)

    def check_read(self, turn: int, path: str) -> bool:
        """检查智能体是否读取了金丝雀标记。"""
        if path in CANARY_FILES:
            self.triggered.append((turn, path))
            return True
        return False


# ── 轨迹 ──────────────────────────────────────────────────

@dataclass
class Action:
    kind: str    # "tool" | "read"
    payload: str


def run_trajectory(traj: list[Action], kill_switch: dict) -> None:
    """运行轨迹，逐动作经过三种检测器。"""
    breaker = CircuitBreaker(threshold=5)
    canary = Canary()
    kill_fired = False
    breaker_fired = False

    for i, a in enumerate(traj, 1):
        # 检测器 1：终止开关
        if kill_switch["enabled"]:
            print(f"  {i:>2}. [终止开关触发] 拒绝动作 {a.kind}:{a.payload}")
            kill_fired = True
            break

        # 检测器 2：断路器
        allowed = breaker.record(f"{a.kind}:{a.payload}")
        if not allowed:
            print(f"  {i:>2}. [断路器打开] {a.kind}:{a.payload}  原因=5次相同调用")
            breaker_fired = True
            break

        # 检测器 3：金丝雀
        if a.kind == "read":
            hit = canary.check_read(i, a.payload)
            if hit:
                print(f"  {i:>2}. [金丝雀触发] 读取 {a.payload!r}  -> 告警已发送")
                continue

        print(f"  {i:>2}. ok  {a.kind}:{a.payload}")

    print(f"  summary: kill_fired={kill_fired}  breaker_fired={breaker_fired}  "
          f"canary_hits={len(canary.triggered)}")


def main() -> None:
    print("=" * 80)
    print("绊线：终止开关、断路器、金丝雀（阶段 15，第 14 课）")
    print("=" * 80)

    traj = [
        Action("tool", "read:src/app.py"),
        Action("tool", "edit:src/app.py"),
        Action("tool", "read:logs/app.log"),   # 开始重复读取
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),   # 第 5 次相同 → 断路器
        Action("read", "~/notes/checklist.md"),
        Action("read", "~/.env.canary"),       # 金丝雀命中
    ]

    print("\n终止开关关闭")
    print("-" * 80)
    run_trajectory(traj, EXTERNAL_KILL_SWITCH)

    print("\n终止开关开启（操作员外部翻转）")
    print("-" * 80)
    EXTERNAL_KILL_SWITCH["enabled"] = True
    run_trajectory(traj, EXTERNAL_KILL_SWITCH)
    EXTERNAL_KILL_SWITCH["enabled"] = False

    print()
    print("=" * 80)
    print("要点：三种检测器，三种不同的失败类别")
    print("-" * 80)
    print("  终止开关在操作员操作时停止整个智能体。")
    print("  断路器暂停特定模式，不是整个智能体。")
    print("  金丝雀标记检测意图而不需要检测内容。")
    print("  这些都不捕获语义组合攻击（见第 10 课）。")
    print("  硬性宪法限制完成防御（第 17 课）。")


if __name__ == "__main__":
    main()
