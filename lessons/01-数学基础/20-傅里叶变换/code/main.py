# 傅里叶变换 — 从零实现 DFT、FFT、卷积定理
# 依赖：Python 3.10+ 标准库（无需第三方库）
# 对应课程：阶段 01 · 20（傅里叶变换）

import math
from typing import List


# ============================================================
# 第 1 步：复数类
# ============================================================
# DFT 的系数是复数。为了从零演示原理，我们手写一个极简复数类，
# 而不是直接用 Python 内置的 complex。这样每一步运算都清晰可见。

class Complex:
    """极简复数类，支持加减乘和共轭操作。"""

    def __init__(self, real: float = 0.0, imag: float = 0.0):
        self.real = float(real)
        self.imag = float(imag)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real + other, self.imag)
        return Complex(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real - other, self.imag)
        return Complex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.imag * other)
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __rmul__(self, other):
        return self.__mul__(other)

    def magnitude(self) -> float:
        """复数的模 |X[k]|，表示该频率分量的振幅。"""
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self) -> float:
        """复数的辐角 angle(X[k])，表示该频率分量的相位。"""
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        """复共轭。用于 IFFT：取共轭 → FFT → 再取共轭并除以 N。"""
        return Complex(self.real, -self.imag)

    def __repr__(self):
        if abs(self.imag) < 1e-12:
            return f"{self.real:.6f}"
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.6f} {sign} {abs(self.imag):.6f}i"


def euler(theta: float) -> Complex:
    """欧拉公式 e^(iθ) = cos(θ) + i·sin(θ)。返回旋转因子（旋转子）。"""
    return Complex(math.cos(theta), math.sin(theta))


# ============================================================
# 第 2 步：离散傅里叶变换（DFT）— O(N²)
# ============================================================
# 核心思想：对每一个频率 k，计算信号与频率 k 的复指数之间的相关性。
# 如果信号中确实含有频率 k，相关性就很大；否则接近零。

def dft(x: List) -> List[Complex]:
    """离散傅里叶变换。输入长度为 N 的信号，输出 N 个复数系数。"""
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * euler(angle)
        result.append(total)
    return result


def idft(X: List) -> List[Complex]:
    """逆离散傅里叶变换。与 DFT 的唯一区别：指数符号为正，结果除以 N。"""
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            xk = X[k] if isinstance(X[k], Complex) else Complex(X[k])
            total = total + xk * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result


# ============================================================
# 第 3 步：快速傅里叶变换（FFT）— O(N log N)
# ============================================================
# Cooley-Tukey 算法：将信号拆分为偶数下标和奇数下标两部分，
# 递归计算各自的 DFT，再用旋转因子（twiddle factor）合并。

def fft(x: List) -> List[Complex]:
    """快速傅里叶变换。要求输入长度为 2 的幂，否则退化为 DFT。"""
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    # 分治：偶数下标和奇数下标
    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = euler(angle)
        t = twiddle * odd[k]
        # 蝴蝶运算：合并两半的结果
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result


def ifft(X: List) -> List[Complex]:
    """逆快速傅里叶变换。利用 FFT 实现：取共轭 → FFT → 取共轭并除以 N。"""
    N = len(X)
    conj_X = [xk.conjugate() if isinstance(xk, Complex) else Complex(xk) for xk in X]
    result = fft(conj_X)
    return [Complex(r.real / N, -r.imag / N) for r in result]


# ============================================================
# 第 4 步：频谱分析辅助函数
# ============================================================

def power_spectrum(X: List[Complex]) -> List[float]:
    """功率谱 |X[k]|²，表示每个频率上的能量。"""
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]


def magnitude_spectrum(X: List[Complex]) -> List[float]:
    """振幅谱 |X[k]|，表示每个频率分量的振幅。"""
    return [xk.magnitude() for xk in X]


def spectral_analysis(signal: List[float], sample_rate: float):
    """对信号进行频谱分析，返回正频率对应的频率值和振幅。"""
    N = len(signal)
    X = fft(signal)
    magnitudes = magnitude_spectrum(X)
    freqs = [k * sample_rate / N for k in range(N)]
    # 实信号只需前半部分（后半部分是镜像）
    return freqs[:N // 2 + 1], magnitudes[:N // 2 + 1]


# ============================================================
# 第 5 步：窗函数（减少频谱泄漏）
# ============================================================

def hann_window(N: int) -> List[float]:
    """汉宁窗：两端渐变为零，用于通用频谱分析。"""
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]


def hamming_window(N: int) -> List[float]:
    """海明窗：比汉宁窗更好地抑制旁瓣，用于音频处理。"""
    return [0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1)) for n in range(N)]


def apply_window(signal: List[float], window: List[float]) -> List[float]:
    """将窗函数逐元素乘以信号。"""
    return [s * w for s, w in zip(signal, window)]


# ============================================================
# 第 6 步：卷积
# ============================================================

def convolve_direct(x: List[float], h: List[float]) -> List[float]:
    """直接计算线性卷积 — O(N·M)。"""
    N = len(x)
    M = len(h)
    out_len = N + M - 1
    result = [0.0] * out_len
    for n in range(out_len):
        total = 0.0
        for k in range(M):
            if 0 <= n - k < N:
                total += x[n - k] * h[k]
        result[n] = total
    return result


def convolve_fft(x: List[float], h: List[float]) -> List[float]:
    """基于 FFT 的卷积 — O(N log N)。利用卷积定理：时域卷积 = 频域乘积。"""
    if len(x) == 0 or len(h) == 0:
        return []
    N = len(x) + len(h) - 1
    # 零填充到 2 的幂，以便 FFT 高效计算
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = list(x) + [0.0] * (padded_N - len(x))
    h_padded = list(h) + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    # 频域逐点相乘
    Y = [xk * hk for xk, hk in zip(X, H)]

    y = ifft(Y)
    return [y[n].real for n in range(N)]


# ============================================================
# 第 7 步：辅助工具
# ============================================================

def generate_signal(frequencies: List[float], amplitudes: List[float],
                    N: int, sample_rate: float) -> List[float]:
    """生成由多个正弦波叠加而成的信号。"""
    signal = [0.0] * N
    for freq, amp in zip(frequencies, amplitudes):
        for n in range(N):
            t = n / sample_rate
            signal[n] += amp * math.sin(2 * math.pi * freq * t)
    return signal


def positional_encoding(pos: int, d_model: int) -> List[float]:
    """生成 Transformer 正弦位置编码。"""
    pe = [0.0] * d_model
    for i in range(d_model // 2):
        freq = 1.0 / (10000 ** (2 * i / d_model))
        angle = pos * freq
        pe[2 * i] = math.sin(angle)
        pe[2 * i + 1] = math.cos(angle)
    return pe


# ============================================================
# 演示代码
# ============================================================

def demo_pure_sine():
    """演示 1：单频正弦信号的 DFT。"""
    print("=" * 65)
    print("  演示 1：单频正弦信号的 DFT")
    print("=" * 65)

    N = 32
    sample_rate = 32
    freq = 5
    signal = generate_signal([freq], [1.0], N, sample_rate)

    print(f"\n  信号：sin(2π·{freq}·t)，{N} 个采样点，采样率 {sample_rate} Hz\n")

    X = dft(signal)
    mags = magnitude_spectrum(X)

    print(f"  {'频率 bin k':<12s} {'频率 (Hz)':>12s} {'|X[k]|':>10s}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 10}")

    for k in range(N // 2 + 1):
        f_hz = k * sample_rate / N
        if mags[k] > 0.01:
            print(f"  k={k:<8d} {f_hz:>12.1f} {mags[k]:>10.4f}")

    print(f"\n  峰值出现在 k={freq}，对应 {freq} Hz。")
    print("  DFT 正确识别了信号的频率。\n")


def demo_multi_frequency():
    """演示 2：多频叠加信号的 FFT。"""
    print("=" * 65)
    print("  演示 2：多频叠加信号的 FFT")
    print("=" * 65)

    N = 64
    sample_rate = 64
    freqs = [3, 7, 15]
    amps = [1.0, 0.5, 0.3]

    signal = generate_signal(freqs, amps, N, sample_rate)

    print(f"\n  信号：{amps[0]}·sin(2π·{freqs[0]}·t) + "
          f"{amps[1]}·sin(2π·{freqs[1]}·t) + "
          f"{amps[2]}·sin(2π·{freqs[2]}·t)")
    print(f"  {N} 个采样点，采样率 {sample_rate} Hz\n")

    X = fft(signal)
    mags = magnitude_spectrum(X)

    print("  恢复的频率（振幅 > 0.5）：")
    print(f"  {'频率 (Hz)':>10s} {'|X[k]|':>10s} {'期望值 (振幅·N/2)':>20s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 20}")

    for k in range(N // 2 + 1):
        if mags[k] > 0.5:
            f_hz = k * sample_rate / N
            expected = ""
            for freq, amp in zip(freqs, amps):
                if abs(f_hz - freq) < 0.1:
                    expected = f"{amp * N / 2:.1f}"
            print(f"  {f_hz:>10.1f} {mags[k]:>10.4f} {expected:>20s}")

    print("\n  三个频率全部正确恢复。")
    print("  振幅与理论值（振幅·N/2）一致。\n")


def demo_fft_vs_dft():
    """演示 3：FFT 与 DFT 结果一致，但速度更快。"""
    print("=" * 65)
    print("  演示 3：FFT vs DFT — 结果一致，速度更快")
    print("=" * 65)

    N = 32
    # 使用固定随机种子保证可复现
    import random
    random.seed(42)
    signal = [random.gauss(0, 1) for _ in range(N)]

    X_dft = dft(signal)
    X_fft = fft(signal)

    max_error = 0.0
    for k in range(N):
        diff_real = abs(X_dft[k].real - X_fft[k].real)
        diff_imag = abs(X_dft[k].imag - X_fft[k].imag)
        max_error = max(max_error, diff_real, diff_imag)

    print(f"\n  随机信号，N = {N}")
    print(f"  DFT 与 FFT 的最大差异：{max_error:.2e}")
    print(f"  结果一致：{max_error < 1e-10}\n")

    print(f"  {'k':<6s} {'DFT |X[k]|':>14s} {'FFT |X[k]|':>14s} {'差异':>12s}")
    print(f"  {'-' * 6} {'-' * 14} {'-' * 14} {'-' * 12}")
    for k in range(8):
        d_mag = X_dft[k].magnitude()
        f_mag = X_fft[k].magnitude()
        diff = abs(d_mag - f_mag)
        print(f"  {k:<6d} {d_mag:>14.8f} {f_mag:>14.8f} {diff:>12.2e}")

    print(f"  ...（共 {N} 个系数，仅展示前 8 个）\n")
    print(f"  DFT 复杂度：O(N²) = {N * N} 次乘法")
    print(f"  FFT 复杂度：O(N·log₂N) = {int(N * math.log2(N))} 次乘法")
    print(f"  理论加速比：{N * N / (N * math.log2(N)):.1f}x")


def demo_reconstruction():
    """演示 4：DFT 与 IDFT 的完美重构。"""
    print("\n" + "=" * 65)
    print("  演示 4：完美重构 — DFT → IDFT")
    print("=" * 65)

    import random
    random.seed(99)
    N = 16
    signal = [random.gauss(0, 2) for _ in range(N)]

    X = fft(signal)
    reconstructed = ifft(X)

    max_err = max(abs(reconstructed[n].real - signal[n]) for n in range(N))

    print(f"\n  原始信号与重构信号（N={N}）：")
    print(f"  {'n':<4s} {'原始值':>12s} {'重构值':>14s} {'误差':>12s}")
    print(f"  {'-' * 4} {'-' * 12} {'-' * 14} {'-' * 12}")

    for n in range(N):
        err = abs(reconstructed[n].real - signal[n])
        print(f"  {n:<4d} {signal[n]:>12.6f} {reconstructed[n].real:>14.6f} {err:>12.2e}")

    print(f"\n  最大重构误差：{max_err:.2e}")
    print(f"  完美重构：{max_err < 1e-10}")
    print("  DFT 是保信息的变换 —— 没有信息损失。")


def demo_convolution_theorem():
    """演示 5：卷积定理 —— 时域卷积 = 频域乘积。"""
    print("\n" + "=" * 65)
    print("  演示 5：卷积定理")
    print("=" * 65)

    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = [1.0, 1.0, 1.0]

    # 直接计算 vs FFT 方法
    direct = convolve_direct(x, h)
    fft_result = convolve_fft(x, h)

    print(f"\n  信号 x = {x}")
    print(f"  滤波器 h = {h}")
    print("  线性卷积 (x * h)：\n")

    print(f"  {'n':<4s} {'直接计算':>10s} {'FFT 方法':>10s} {'差异':>12s}")
    print(f"  {'-' * 4} {'-' * 10} {'-' * 10} {'-' * 12}")

    max_err = 0.0
    for n in range(len(direct)):
        diff = abs(direct[n] - fft_result[n])
        max_err = max(max_err, diff)
        print(f"  {n:<4d} {direct[n]:>10.4f} {fft_result[n]:>10.4f} {diff:>12.2e}")

    print(f"\n  最大差异：{max_err:.2e}")
    print(f"  结果一致：{max_err < 1e-8}\n")
    print("  时域卷积 = 频域乘积")
    print(f"  直接卷积复杂度：O(N·M)，FFT 卷积复杂度：O(N·logN)")


def demo_windowing():
    """演示 6：窗函数与频谱泄漏。"""
    print("\n" + "=" * 65)
    print("  演示 6：窗函数与频谱泄漏")
    print("=" * 65)

    N = 64
    sample_rate = 64
    freq = 7.5  # 频率不在整数 bin 上，会导致频谱泄漏

    signal = [math.sin(2 * math.pi * freq * n / sample_rate) for n in range(N)]

    # 不加窗
    X_rect = fft(signal)
    mags_rect = magnitude_spectrum(X_rect)

    # 加汉宁窗
    hann = hann_window(N)
    signal_hann = apply_window(signal, hann)
    X_hann = fft(signal_hann)
    mags_hann = magnitude_spectrum(X_hann)

    # 加海明窗
    hamm = hamming_window(N)
    signal_hamm = apply_window(signal, hamm)
    X_hamm = fft(signal_hamm)
    mags_hamm = magnitude_spectrum(X_hamm)

    print(f"\n  信号：sin(2π·{freq}·t) —— 频率落在 bin 之间")
    print(f"  N = {N}，采样率 = {sample_rate} Hz")
    print(f"  频率分辨率：{sample_rate / N:.2f} Hz/bin")
    print(f"  {freq} Hz 落在 bin 7 和 bin 8 之间\n")

    print(f"  {'频率 (Hz)':>10s} {'无窗':>12s} {'汉宁窗':>12s} {'海明窗':>12s}")
    print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12}")

    for k in range(N // 2 + 1):
        f_hz = k * sample_rate / N
        if mags_rect[k] > 0.5 or (5 <= f_hz <= 11):
            print(f"  {f_hz:>10.1f} {mags_rect[k]:>12.4f} "
                  f"{mags_hann[k]:>12.4f} {mags_hamm[k]:>12.4f}")

    print("\n  不加窗时，能量泄漏到相邻的频率 bin。")
    print("  汉宁窗和海明窗将能量集中在真实频率附近。")
    print("  权衡：窗函数会展宽主峰，但抑制旁瓣。")


def demo_parseval():
    """演示 7：帕塞瓦尔定理 —— 能量守恒。"""
    print("\n" + "=" * 65)
    print("  演示 7：帕塞瓦尔定理 —— 能量守恒")
    print("=" * 65)

    import random
    random.seed(7)
    N = 32
    signal = [random.gauss(0, 1) for _ in range(N)]

    time_energy = sum(s ** 2 for s in signal)

    X = fft(signal)
    freq_energy = sum(xk.real ** 2 + xk.imag ** 2 for xk in X) / N

    print(f"\n  信号：{N} 个随机采样")
    print(f"  时域能量：Σ|x[n]|² = {time_energy:.6f}")
    print(f"  频域能量：(1/N)·Σ|X[k]|² = {freq_energy:.6f}")
    print(f"  差异：{abs(time_energy - freq_energy):.2e}")
    print(f"  能量守恒：{abs(time_energy - freq_energy) < 1e-10}")
    print("\n  傅里叶变换是正交变换，总能量在变换前后保持不变。")


def demo_positional_encoding():
    """演示 8：Transformer 位置编码的频率结构。"""
    print("\n" + "=" * 65)
    print("  演示 8：Transformer 位置编码的频率结构")
    print("=" * 65)

    d_model = 16
    max_pos = 8

    print(f"\n  d_model = {d_model}，位置 0-{max_pos - 1}\n")

    print("  每个维度对的频率：")
    for i in range(d_model // 2):
        freq = 1.0 / (10000 ** (2 * i / d_model))
        wavelength = 2 * math.pi / freq if freq > 0 else float('inf')
        print(f"    维度 ({2 * i:>2d},{2 * i + 1:>2d})：频率 = {freq:.8f}  "
              f"波长 = {wavelength:.1f}")

    print("\n  位置编码的点积：")
    print("  （点积只依赖于位置间距，不依赖绝对位置）\n")

    print(f"  {'pos_i':>6s} {'pos_j':>6s} {'间距':>6s} {'点积':>12s}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 12}")

    pairs = [(0, 0), (0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 4), (3, 7)]
    for p1, p2 in pairs:
        pe1 = positional_encoding(p1, d_model)
        pe2 = positional_encoding(p2, d_model)
        dot = sum(a * b for a, b in zip(pe1, pe2))
        print(f"  {p1:>6d} {p2:>6d} {abs(p2 - p1):>6d} {dot:>12.4f}")

    print("\n  间距相同的位置对具有相似的点积。")
    print("  这让模型通过注意力机制学习相对位置关系。")


def demo_frequency_scaling():
    """演示 9：FFT 相对于 DFT 的加速比。"""
    print("\n" + "=" * 65)
    print("  演示 9：FFT 复杂度随 N 的增长")
    print("=" * 65)

    print(f"\n  {'N':>8s} {'DFT O(N²)':>14s} {'FFT O(N·logN)':>16s} {'加速比':>10s}")
    print(f"  {'-' * 8} {'-' * 14} {'-' * 16} {'-' * 10}")

    for exp in range(3, 14):
        N = 2 ** exp
        dft_ops = N * N
        fft_ops = int(N * math.log2(N))
        speedup = dft_ops / fft_ops
        print(f"  {N:>8d} {dft_ops:>14,d} {fft_ops:>16,d} {speedup:>10.1f}x")

    print("\n  当 N = 1,048,576 时，DFT 需要 10¹² 次运算，FFT 仅需约 2×10⁷ 次。")


def write_prompt_output():
    """生成可复用的频谱分析提示词文件。"""
    output_path = "../outputs/prompt-fourier-transform-tutor.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write("name: prompt-fourier-transform-tutor\n")
        f.write("description: 傅里叶变换频谱分析引导提示词\n")
        f.write("phase: 1\n")
        f.write("lesson: 20\n")
        f.write("---\n\n")
        f.write("你是一位频谱分析专家。你帮助工程师使用傅里叶变换技术分析信号的频率成分。\n\n")
        f.write("当收到一个信号或信号描述时，按以下步骤引导分析：\n\n")
        f.write("1. **确定采样参数。**\n")
        f.write("   - 采样率 fs 是多少？这决定了最大可检测频率（奈奎斯特频率 = fs/2）。\n")
        f.write("   - 采样点数 N 是多少？这决定了频率分辨率（Δf = fs/N）。\n")
        f.write("   - 信号长度是否为 2 的幂？如果不是，建议零填充以提高 FFT 效率。\n\n")
        f.write("2. **选择窗函数。**\n")
        f.write("   - 信号在分析窗口内是否恰好周期？如果是，无需加窗。\n")
        f.write("   - 通用分析：使用汉宁窗（Hann window），在分辨率和泄漏之间取得良好平衡。\n")
        f.write("   - 音频/语音处理：使用海明窗（Hamming window）。\n")
        f.write("   - 当旁瓣抑制最为关键时：使用布莱克曼窗（Blackman window）。\n")
        f.write("   - 记住：加窗会展宽主峰，但能减少泄漏。\n\n")
        f.write("3. **计算并解释频谱。**\n")
        f.write("   - 功率谱 |X[k]|² 显示每个频率上的能量。\n")
        f.write("   - 功率谱中的峰值表示主导频率。\n")
        f.write("   - X[0] 是直流分量（信号均值 × N）。\n")
        f.write("   - 对于实信号，只需观察 bin 0 到 N/2（上半部分是镜像）。\n")
        f.write("   - bin k 对应的频率：f_k = k · fs / N。\n\n")
        f.write("4. **识别主导频率。**\n")
        f.write("   - 找到高于噪声阈值的峰值。\n")
        f.write("   - 将 bin 索引转换为 Hz：频率 = k · fs / N。\n")
        f.write("   - 检查谐波（基频整数倍处的峰值）。\n")
        f.write("   - 检查混叠频率（实际频率 = 表观频率 mod fs；若超过 fs/2，则折叠为 fs - f_apparent）。\n\n")
        f.write("5. **常见陷阱。**\n")
        f.write("   - 频谱泄漏：窗口内非整数周期导致能量分散到多个 bin。\n")
        f.write("   - 混叠：信号包含超过 fs/2 的频率时会在频谱中折叠回来。\n")
        f.write("   - 直流偏移：较大的 X[0] 会掩盖附近的低频内容。FFT 前先去除均值。\n")
        f.write("   - 零填充增加 bin 密度，但不会提高实际频率分辨率。\n")
        f.write("   - 圆周卷积 vs 线性卷积：DFT 自然给出圆周卷积。需要线性卷积时先零填充。\n\n")
        f.write("6. **卷积分析。**\n")
        f.write("   - 时域卷积 = 频域乘积。\n")
        f.write("   - 对于大卷积核，FFT 卷积更快：O(N log N) vs O(N·M)。\n")
        f.write("   - 将两个信号都零填充到长度 N + M - 1 以获得正确的线性卷积。\n")
    print(f"\n  提示词已写入 {output_path}")


def print_summary():
    """打印课程总结。"""
    print("\n" + "=" * 65)
    print("  课程总结")
    print("=" * 65)
    print("""
  1. DFT 将 N 个时域采样转换为 N 个频域系数。
  2. 每个 X[k] 测量信号与频率 k 的相关性。
  3. FFT 以 O(N·logN) 计算 DFT，比直接 DFT 快数百倍。
  4. DFT 和 IDFT 完美互逆 —— 没有信息损失。
  5. 卷积定理：时域卷积 = 频域乘积。这是 FFT 卷积高效的原因。
  6. 窗函数减少非周期信号的频谱泄漏。
  7. 帕塞瓦尔定理：能量在变换前后守恒。
  8. Transformer 位置编码使用相同的频率分解思想 ——
     每个位置获得独特的频谱特征。""")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    demo_pure_sine()
    demo_multi_frequency()
    demo_fft_vs_dft()
    demo_reconstruction()
    demo_convolution_theorem()
    demo_windowing()
    demo_parseval()
    demo_positional_encoding()
    demo_frequency_scaling()
    write_prompt_output()
    print_summary()
