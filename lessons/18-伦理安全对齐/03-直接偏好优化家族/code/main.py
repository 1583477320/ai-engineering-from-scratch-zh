"""DPO vs BPO 偏好损失比较。"""
import math


def dpo_loss(pi_y_w, pi_y_l, pi_ref_w, pi_ref_l, beta=0.1):
    gap = beta * (math.log(pi_y_w / pi_ref_w) - math.log(pi_y_l / pi_ref_l))
    return -math.log(1.0 / (1.0 + math.exp(-gap)))


def bpo_loss(pi_y_w, pi_y_l, pi_ref_w, pi_ref_l, protect=0.01, beta=0.1):
    base = dpo_loss(pi_y_w, pi_y_l, pi_ref_w, pi_ref_l, beta)
    correction = protect * max(0, pi_ref_w - pi_y_w)
    return base + correction


if __name__ == "__main__":
    for name, fn in [("DPO", dpo_loss), ("BPO", bpo_loss)]:
        l = fn(0.6, 0.3, 0.7, 0.3)
        print(f"{name}: {l:.4f}")
