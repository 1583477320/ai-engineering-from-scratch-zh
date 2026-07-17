"""STaR 循环模拟器——纯标准库。

玩具算术任务。"模型"通过三种策略产生推理链：
  1. 合理推理（总是正确）
  2. 捷径（分布内 40% 正确，OOD 接近零）
  3. 随机猜测

STaR 自举轮次过滤到正确答案的推理链。没有保护时，
捷径推理链会被强化因为它们在分布内看起来是正确的。

运行：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Trace:
    strategy: str       # "sound" | "shortcut" | "random"
    answer_correct: bool
    rationale_sound: bool


@dataclass
class Model:
    prob_sound: float
    prob_shortcut: float

    def sample(self, on_ood: bool) -> Trace:
        r = random.random()
        if r < self.prob_sound:
            return Trace("sound", True, True)
        elif r < self.prob_sound + self.prob_shortcut:
            ok = random.random() < (0.05 if on_ood else 0.40)
            return Trace("shortcut", ok, False)
        else:
            ok = random.random() < 0.10
            return Trace("random", ok, False)


def evaluate(model: Model, n: int, on_ood: bool) -> tuple[float, float]:
    """返回 (答案准确率, 推理合理比例)。"""
    correct = 0
    sound = 0
    for _ in range(n):
        t = model.sample(on_ood)
        if t.answer_correct:
            correct += 1
        if t.rationale_sound:
            sound += 1
    return correct / n, sound / n


def star_round(model: Model, n_samples: int = 1000) -> Model:
    """一轮 STaR：保留正确答案的推理链，重新训练。"""
    kept = []
    for _ in range(n_samples):
        t = model.sample(on_ood=False)
        if t.answer_correct:
            kept.append(t)
    if not kept:
        return model

    sound_kept = sum(1 for k in kept if k.strategy == "sound")
    shortcut_kept = sum(1 for k in kept if k.strategy == "shortcut")
    total = len(kept)

    alpha = 0.6
    new_sound = alpha * (sound_kept / total) + (1 - alpha) * model.prob_sound
    new_short = alpha * (shortcut_kept / total) + (1 - alpha) * model.prob_shortcut
    s = new_sound + new_short
    if s > 1.0:
        new_sound /= s
        new_short /= s
    return Model(new_sound, new_short)


def run_star(rounds: int, initial: Model) -> list[Model]:
    models = [initial]
    m = initial
    for _ in range(rounds):
        m = star_round(m)
        models.append(m)
    return models


def vstar_infer(model: Model, samples_per_problem: int, n_problems: int, on_ood: bool) -> float:
    """V-STaR 风格 best-of-N 推理。"""
    correct = 0
    for _ in range(n_problems):
        traces = [model.sample(on_ood) for _ in range(samples_per_problem)]
        best = None
        best_score = -1.0
        for t in traces:
            score = 0.9 if t.rationale_sound else (0.55 if t.answer_correct else 0.3)
            score += random.random() * 0.1
            if score > best_score:
                best_score = score
                best = t
        if best and best.answer_correct:
            correct += 1
    return correct / n_problems


def report_round(label: str, models: list[Model]) -> None:
    print(f"\n{label}")
    print("-" * 70)
    print(f"  {'轮':>5}  {'p(sound)':>10}  {'p(shortcut)':>12}  "
          f"{'ID 准确率':>10}  {'OOD 准确率':>10}  {'合理比例':>10}")
    for i, m in enumerate(models):
        id_acc, id_sound = evaluate(m, 500, on_ood=False)
        ood_acc, _ = evaluate(m, 500, on_ood=True)
        print(f"  {i:>5}  {m.prob_sound:>10.3f}  {m.prob_shortcut:>12.3f}  "
              f"{id_acc:>10.1%}  {ood_acc:>10.1%}  {id_sound:>10.1%}")


def vstar_report(model: Model) -> None:
    print("\nV-STaR best-of-N 推理")
    print("-" * 70)
    for n in (1, 4, 16):
        for ood in (False, True):
            acc = vstar_infer(model, n, 500, ood)
            tag = "OOD" if ood else "ID"
            print(f"  n={n:>3}  {tag:<3}  准确率 {acc:.1%}")


def main() -> None:
    random.seed(42)
    print("=" * 70)
    print("STaR、V-STaR、Quiet-STaR（阶段 15，第 2 课）")
    print("=" * 70)

    print("\n场景 A：无捷径先验（干净推理先验）")
    models_a = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.0))
    report_round("STaR 自举轮次（干净）", models_a)

    print("\n场景 B：有捷径先验（0.4 分布内命中率）")
    models_b = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.40))
    report_round("STaR 自举轮次（有捷径）", models_b)

    vstar_report(models_b[-1])

    print()
    print("=" * 70)
    print("要点：STaR 强化任何恰好得到答案的方法")
    print("-" * 70)
    print("  场景 A 在 ID 和 OOD 上都攀升。")
    print("  场景 B 在 ID 上攀升而 OOD 崩溃——捷径在训练数据中看起来正确。")
    print("  V-STaR 的验证器在推理时有帮助，但无法消除训练偏差。")


if __name__ == "__main__":
    main()
