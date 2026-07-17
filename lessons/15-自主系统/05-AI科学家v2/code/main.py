"""AI Scientist v2 循环模拟器——纯标准库。

将研究循环建模为带可配置每阶段失败概率的状态机，
数据来自 Beel 等人（2025）对 AI Scientist 真实行

运行：python3 code/main.py [--experiment-failure 0.42] [--novelty-mislabel 0.25]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass


DEFAULT_SEED = 42


@dataclass
class LoopConfig:
    """循环配置——失败概率来自 Beel 等人数据。"""
    novelty_mislabel: float = 0.25       # 文献新颖性误标率
    experiment_failure: float = 0.42     # 实验失败率
    retry_recovery: float = 0.55         # 重试恢复率
    polish_masks_weakness: float = 0.70  # VLM 润色掩盖缺陷概率
    writeup_success: float = 0.85        # 论文撰写成功率
    internal_review_accept: float = 0.50 # 内部评审接受率


@dataclass
class Outcome:
    submitted: bool
    has_novelty_flaw: bool
    has_experiment_flaw: bool
    polished_but_flawed: bool
    polished_ok: bool
    abandoned_stage: str


def run_one(cfg: LoopConfig) -> Outcome:
    """运行一次研究循环。"""
    has_novelty_flaw = random.random() < cfg.novelty_mislabel

    # 实验执行 + 重试
    failed = random.random() < cfg.experiment_failure
    if failed:
        recovered = random.random() < cfg.retry_recovery
        if not recovered:
            return Outcome(
                submitted=False, has_novelty_flaw=has_novelty_flaw,
                has_experiment_flaw=True, polished_but_flawed=False,
                polished_ok=False, abandoned_stage="experiment",
            )
        # 重试恢复后仍有残余缺陷
        has_experiment_flaw = True
    else:
        has_experiment_flaw = False

    # VLM 图表润色——可能掩盖缺陷
    polished_hides = has_experiment_flaw and random.random() < cfg.polish_masks_weakness

    # 论文撰写
    if random.random() > cfg.writeup_success:
        return Outcome(
            submitted=False, has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False, polished_ok=False,
            abandoned_stage="writeup",
        )

    # 内部评审
    if random.random() > cfg.internal_review_accept:
        return Outcome(
            submitted=False, has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False, polished_ok=False,
            abandoned_stage="internal_review",
        )

    polished_ok = not has_experiment_flaw and not has_novelty_flaw
    polished_but_flawed = has_experiment_flaw or has_novelty_flaw
    return Outcome(
        submitted=True, has_novelty_flaw=has_novelty_flaw,
        has_experiment_flaw=has_experiment_flaw,
        polished_but_flawed=polished_but_flawed,
        polished_ok=polished_ok, abandoned_stage="",
    )


def report(n: int, cfg: LoopConfig) -> None:
    """运行 n 次试验并报告分布。"""
    outs = [run_one(cfg) for _ in range(n)]
    submitted = [o for o in outs if o.submitted]
    abandoned = [o for o in outs if not o.submitted]
    polished_ok = [o for o in submitted if o.polished_ok]
    polished_but_flawed = [o for o in submitted if o.polished_but_flawed]

    print(f"  配置")
    print(f"    新颖性误标率           : {cfg.novelty_mislabel:.2f}")
    print(f"    实验失败率             : {cfg.experiment_failure:.2f}")
    print(f"    重试恢复比例           : {cfg.retry_recovery:.2f}")
    print(f"    润色掩盖缺陷概率       : {cfg.polish_masks_weakness:.2f}")
    print(f"    论文撰写成功率         : {cfg.writeup_success:.2f}")
    print(f"    内部评审接受率         : {cfg.internal_review_accept:.2f}")

    print()
    print(f"  试验数                    : {n}")
    print(f"  投稿数                   : {len(submitted)} ({len(submitted)/n:.1%})")
    print(f"  放弃                     : {len(abandoned)} ({len(abandoned)/n:.1%})")
    by_stage = {}
    for o in abandoned:
        by_stage[o.abandoned_stage] = by_stage.get(o.abandoned_stage, 0) + 1
    for stage, count in sorted(by_stage.items()):
        print(f"    在 {stage:<18}: {count}")

    print()
    print(f"  投稿质量分布")
    print(f"    干净（新颖+有效）       : {len(polished_ok)} "
          f"({len(polished_ok)/n:.1%} of trials, "
          f"{len(polished_ok)/max(1, len(submitted)):.1%} of submissions)")
    print(f"    润色但有缺陷           : {len(polished_but_flawed)} "
          f"({len(polished_but_flawed)/n:.1%} of trials, "
          f"{len(polished_but_flawed)/max(1, len(submitted)):.1%} of submissions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-failure", type=float, default=None)
    parser.add_argument("--novelty-mislabel", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    print("=" * 70)
    print("AI SCIENTIST V2 循环模拟器（阶段 15，第 5 课）")
    print("=" * 70)

    overrides = {}
    if args.experiment_failure is not None:
        overrides["experiment_failure"] = args.experiment_failure
    if args.novelty_mislabel is not None:
        overrides["novelty_mislabel"] = args.novelty_mislabel

    label = "基线 (Beel 风格数据)" if not overrides else "基线 (覆盖参数)"
    print(f"\n{label}")
    print("-" * 70)
    report(1000, LoopConfig(**overrides))

    print("\n乐观场景（更严格的数字）")
    print("-" * 70)
    report(1000, LoopConfig(
        novelty_mislabel=0.10, experiment_failure=0.20,
        retry_recovery=0.80, polish_masks_weakness=0.40,
        writeup_success=0.92, internal_review_accept=0.60,
    ))

    print()
    print("=" * 70)
    print("要点: 投稿数超过健全研究数")
    print("-" * 70)
    print("  即使在乐观场景中，非平凡比例的投稿论文带着润色掩盖的")
    print("  缺陷。这就是'呈现质量差距'的操作含义——以及为什么")
    print("  人工审查门控位于循环和任何投稿场所之间。")


if __name__ == "__main__":
    main()
