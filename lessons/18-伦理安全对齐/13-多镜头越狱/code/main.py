"""MSJ 攻击成功率模拟。"""


def msj_asr(shots, base=0.05, exp=1.5):
    if shots < 5:
        return 0.0
    return min(base * (shots / 5) ** exp, 0.99)


def with_defense(shots, reduction=0.3):
    return max(msj_asr(shots) * reduction, 0.01)


if __name__ == "__main__":
    print(f"{'镜头':>6}  {'ASR':>8}  {'有防御':>8}")
    print("-" * 26)
    for s in [1, 5, 10, 32, 64, 128, 256, 512]:
        print(f"{s:>6d}  {msj_asr(s):>8.1%}  {with_defense(s):>8.1%}")
