"""DP-SGD 噪声注入模拟。"""
import random
import math


def dp_sgd_step(gradient, clip_norm=1.0, noise_mult=1.0):
    grad_norm = math.sqrt(sum(g**2 for g in gradient))
    if grad_norm > clip_norm:
        gradient = [g * clip_norm / grad_norm for g in gradient]
    noisy = [g + random.gauss(0, noise_mult * clip_norm) for g in gradient]
    return noisy


def compute_epsilon(steps, noise_mult, delta=1e-5):
    return steps * (1.6 * noise_mult) ** 2, delta


if __name__ == "__main__":
    gradient = [0.5, -0.3, 0.8, -0.2]
    noisy = dp_sgd_step(gradient, clip_norm=1.0, noise_mult=1.0)
    eps, delta = compute_epsilon(steps=100, noise_mult=1.0)
    print(f"原始: {[f'{g:.2f}' for g in gradient]}")
    print(f"DP后: {[f'{g:.2f}' for g in noisy]}")
    print(f"(ε, δ) = ({eps:.2f}, {delta:.1e})")
