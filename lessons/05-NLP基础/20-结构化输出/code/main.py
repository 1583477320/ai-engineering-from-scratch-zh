# 约束解码——FSM 掩码 logits 的教学实现
# 对应课程：阶段 05 · 20
# 演示：电话号码正则 \d{3}-\d{3}-\d{4} 的无约束 vs FSM 约束生成

import math, random, re

PHONE_REGEX = r"^\d{3}-\d{3}-\d{4}$"


class PhoneFSM:
    """电话号码有限状态机。状态 0-11 逐步推进，状态 12 = 接受。"""
    def __init__(self):
        self.accept_state = 12

    def valid_next(self, state):
        if state in (0, 1, 2, 4, 5, 6, 8, 9, 10, 11):
            return list("0123456789")
        if state in (3, 7): return ["-"]
        if state == 12: return []
        raise ValueError(f"未知状态 {state}")

    def transition(self, state, ch):
        return state + 1 if ch in self.valid_next(state) else None

    def is_accept(self, state):
        return state == self.accept_state


def softmax(xs):
    finite = [x for x in xs if x != float("-inf")]
    if not finite: return [0.0] * len(xs)
    m = max(finite)
    exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def sample(probs, rng):
    r, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc: return i
    return len(probs) - 1


def mask_logits(logits, valid_indices):
    """将无效 token 的 logit 设为 -∞——这是约束解码的核心操作。"""
    return [logits[i] if i in valid_indices else float("-inf") for i in range(len(logits))]


def fake_llm_logits(alphabet, rng):
    return [rng.gauss(0.0, 1.5) for _ in alphabet]


def generate_constrained(alphabet, fsm, seed):
    """FSM 约束生成——每步掩码无效字符。"""
    rng = random.Random(seed)
    idx = {ch: i for i, ch in enumerate(alphabet)}
    state, out = 0, ""
    while not fsm.is_accept(state):
        logits = fake_llm_logits(alphabet, rng)
        valid = fsm.valid_next(state)
        if not valid: break
        masked = mask_logits(logits, {idx[ch] for ch in valid})
        probs = softmax(masked)
        ch = alphabet[sample(probs, rng)]
        out += ch
        state = fsm.transition(state, ch)
    return out


def generate_unconstrained(alphabet, max_len, seed):
    """无约束生成——随机 logit 采样，无掩码。"""
    rng = random.Random(seed)
    out = ""
    for _ in range(max_len):
        probs = softmax(fake_llm_logits(alphabet, rng))
        out += alphabet[sample(probs, rng)]
    return out


def main():
    alphabet = list("0123456789-")
    fsm = PhoneFSM()
    print(f"=== 电话号码生成：20 次采样 ===")
    print(f"目标模式: {PHONE_REGEX}\n")

    print("无约束（随机 logit，无掩码）:")
    unc_ok = 0
    for s in range(20):
        out = generate_unconstrained(alphabet, max_len=12, seed=s)
        ok = bool(re.fullmatch(PHONE_REGEX, out))
        unc_ok += int(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {out}")
    print(f"  → 合法: {unc_ok}/20")

    print(f"\nFSM 约束（logit 掩码）:")
    con_ok = 0
    for s in range(20):
        out = generate_constrained(alphabet, fsm, seed=s)
        ok = bool(re.fullmatch(PHONE_REGEX, out))
        con_ok += int(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {out}")
    print(f"  → 合法: {con_ok}/20")

    print("\n注意：玩具 LLM 发出均匀随机 logits——真正的 LLM 有偏好的概率。")
    print("掩码无效 token 是唯一的差别。真实约束解码在 10 万+ 词表上做同样的掩码。")


if __name__ == "__main__":
    main()
