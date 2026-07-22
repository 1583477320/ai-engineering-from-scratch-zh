# 复数运算的从零实现
# 依赖：Python 3.10+ 标准库（无需第三方库）
# 对应课程：阶段 01 · 19（复数）

import math


class Complex:
    """复数类：支持直角坐标形式和极坐标形式之间的转换，以及四则运算。"""

    def __init__(self, real: float, imag: float = 0.0):
        self.real = float(real)
        self.imag = float(imag)

    # === 算术运算 ===

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Complex(other)
        return Complex(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self.__add__(Complex(other))

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Complex(other)
        return Complex(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other):
        return Complex(other - self.real, -self.imag)

    def __mul__(self, other):
        """乘法：(a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        if isinstance(other, (int, float)):
            other = Complex(other)
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return Complex(real_part, imag_part)

    def __rmul__(self, other):
        return self.__mul__(Complex(other))

    def __truediv__(self, other):
        """除法：分子分母同乘分母的共轭，消去分母中的虚部"""
        if isinstance(other, (int, float)):
            other = Complex(other)
        denom = other.real ** 2 + other.imag ** 2
        if denom == 0:
            raise ZeroDivisionError("除以零复数")
        real_part = (self.real * other.real + self.imag * other.imag) / denom
        imag_part = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(real_part, imag_part)

    def __neg__(self):
        return Complex(-self.real, -self.imag)

    # === 模与辐角 ===

    def magnitude(self) -> float:
        """模（绝对值）：到原点的距离"""
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self) -> float:
        """辐角（相位）：与正实轴的夹角，范围 (-pi, pi]"""
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        """共轭：虚部取反"""
        return Complex(self.real, -self.imag)

    # === 表示 ===

    def __repr__(self):
        if abs(self.imag) < 1e-12:
            return f"{self.real:.6f}"
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.6f} {sign} {abs(self.imag):.6f}i"

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Complex(other)
        return (abs(self.real - other.real) < 1e-10 and
                abs(self.imag - other.imag) < 1e-10)


# === 极坐标转换与欧拉公式 ===

def to_polar(z: Complex):
    """直角坐标 → 极坐标 (r, theta)"""
    return z.magnitude(), z.phase()


def from_polar(r: float, theta: float) -> Complex:
    """极坐标 (r, theta) → 直角坐标"""
    return Complex(r * math.cos(theta), r * math.sin(theta))


def euler(theta: float) -> Complex:
    """欧拉公式：e^(i*theta) = cos(theta) + i*sin(theta)"""
    return Complex(math.cos(theta), math.sin(theta))


# === 旋转 ===

def rotate(point: Complex, angle: float) -> Complex:
    """将点绕原点旋转指定角度（弧度）"""
    return point * euler(angle)


# === 单位根 ===

def roots_of_unity(n: int):
    """计算 n 次单位根：e^(2*pi*i*k/n), k = 0, 1, ..., n-1"""
    return [euler(2 * math.pi * k / n) for k in range(n)]


# === 离散傅里叶变换 (DFT) ===

def dft(signal):
    """离散傅里叶变换：O(N^2) 实现"""
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            xn = signal[n] if isinstance(signal[n], Complex) else Complex(signal[n])
            total = total + xn * euler(angle)
        result.append(total)
    return result


def idft(spectrum):
    """逆离散傅里叶变换：完美重建原始信号"""
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result


# === 演示函数 ===

def demo_arithmetic():
    print("=" * 65)
    print("  复数四则运算")
    print("=" * 65)
    print()

    z1 = Complex(3, 2)
    z2 = Complex(1, 4)

    print(f"  z1 = {z1}")
    print(f"  z2 = {z2}")
    print()

    print(f"  z1 + z2  = {z1 + z2}")
    print(f"  z1 - z2  = {z1 - z2}")
    print(f"  z1 * z2  = {z1 * z2}")
    print(f"  z1 / z2  = {z1 / z2}")
    print()

    print(f"  |z1|     = {z1.magnitude():.6f}")
    print(f"  phase(z1)= {z1.phase():.6f} rad ({math.degrees(z1.phase()):.2f} deg)")
    print(f"  conj(z1) = {z1.conjugate()}")
    print()

    # 验证：z * conj(z) = |z|^2
    product = z1 * z1.conjugate()
    expected = z1.real ** 2 + z1.imag ** 2
    print(f"  z1 * conj(z1) = {product}")
    print(f"  a^2 + b^2     = {expected:.6f}")
    print(f"  验证通过: {abs(product.real - expected) < 1e-10}")
    print()

    # 验证除法：商 × 除数 = 被除数
    z3 = Complex(5, 2)
    z4 = Complex(1, -3)
    quotient = z3 / z4
    reconstructed = quotient * z4
    print(f"  除法验证: (5+2i) / (1-3i) = {quotient}")
    print(f"  重建:     result * (1-3i)  = {reconstructed}")
    print(f"  与原数一致: {abs(reconstructed.real - 5) < 1e-10 and abs(reconstructed.imag - 2) < 1e-10}")


def demo_polar_conversion():
    print()
    print()
    print("=" * 65)
    print("  极坐标形式与转换")
    print("=" * 65)
    print()

    test_cases = [
        Complex(1, 0),
        Complex(0, 1),
        Complex(-1, 0),
        Complex(0, -1),
        Complex(3, 4),
        Complex(-2, 3),
    ]

    print(f"  {'直角坐标':<25s} {'r':>8s}  {'theta (deg)':>12s}  {'还原':<25s}")
    print(f"  {'-' * 25} {'-' * 8}  {'-' * 12}  {'-' * 25}")

    for z in test_cases:
        r, theta = to_polar(z)
        z_back = from_polar(r, theta)
        print(f"  {str(z):<25s} {r:>8.4f}  {math.degrees(theta):>12.2f}  {str(z_back):<25s}")


def demo_euler_formula():
    print()
    print()
    print("=" * 65)
    print("  欧拉公式: e^(i*theta) = cos(theta) + i*sin(theta)")
    print("=" * 65)
    print()

    angles = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2,
              math.pi, 3 * math.pi / 2, 2 * math.pi]
    labels = ["0", "pi/6", "pi/4", "pi/3", "pi/2", "pi", "3pi/2", "2pi"]

    print(f"  {'theta':<8s} {'cos(theta)':>12s} {'sin(theta)':>12s} "
          f"{'e^(i*theta)':>25s} {'|e^(i*theta)|':>14s}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 25} {'-' * 14}")

    for label, theta in zip(labels, angles):
        e = euler(theta)
        print(f"  {label:<8s} {math.cos(theta):>12.6f} {math.sin(theta):>12.6f} "
              f"  {str(e):>23s} {e.magnitude():>14.10f}")

    print()
    e_pi = euler(math.pi)
    result = e_pi + Complex(1, 0)
    print(f"  欧拉恒等式: e^(i*pi) + 1 = {result}")
    print(f"  |e^(i*pi) + 1| = {result.magnitude():.2e} (应接近 0)")


def demo_rotation():
    print()
    print()
    print("=" * 65)
    print("  通过复数乘法实现旋转")
    print("=" * 65)
    print()

    point = Complex(3, 4)
    print(f"  原始点: {point}")
    print(f"  模: {point.magnitude():.4f}")
    print(f"  辐角: {math.degrees(point.phase()):.2f} deg")
    print()

    rotation_angles = [45, 90, 180, 270, 360]

    print(f"  {'旋转角度':<12s} {'结果':<30s} {'模':>10s} {'辐角 (deg)':>12s}")
    print(f"  {'-' * 12} {'-' * 30} {'-' * 10} {'-' * 12}")

    for deg in rotation_angles:
        rad = math.radians(deg)
        rotated = rotate(point, rad)
        r, theta = to_polar(rotated)
        print(f"  {deg:>3d} deg     {str(rotated):<30s} {r:>10.4f} {math.degrees(theta):>12.2f}")

    print()
    print("  旋转保持模不变。360 度后回到原点。")
    print()

    # 验证复数乘法与旋转矩阵等价
    print("  复数乘法 vs 旋转矩阵等价性验证:")
    print()

    test_angles = [math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]
    test_points = [Complex(1, 0), Complex(3, 4), Complex(-2, 5)]

    max_error = 0.0
    for theta in test_angles:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for p in test_points:
            complex_result = p * euler(theta)
            matrix_x = cos_t * p.real - sin_t * p.imag
            matrix_y = sin_t * p.real + cos_t * p.imag

            err = math.sqrt((complex_result.real - matrix_x) ** 2 +
                            (complex_result.imag - matrix_y) ** 2)
            max_error = max(max_error, err)

    print(f"  最大数值差异: {max_error:.2e}")


def demo_roots_of_unity():
    print()
    print()
    print("=" * 65)
    print("  单位根")
    print("=" * 65)
    print()

    for n in [4, 8]:
        roots = roots_of_unity(n)
        print(f"  {n} 次单位根:")
        print(f"  {'k':<4s} {'根':<30s} {'|根|':>8s}")
        print(f"  {'-' * 4} {'-' * 30} {'-' * 8}")

        total = Complex(0, 0)
        for k, root in enumerate(roots):
            total = total + root
            print(f"  {k:<4d} {str(root):<30s} {root.magnitude():>8.6f}")

        print(f"  所有根之和: {total}")
        print(f"  |sum| = {total.magnitude():.2e} (应接近 0)")
        print()

    print("  单位根之和恒为零，每个根的模恰好为 1。")


def demo_dft():
    print()
    print()
    print("=" * 65)
    print("  DFT 对简单信号的分析")
    print("=" * 65)
    print()

    N = 32
    freq1 = 3
    freq2 = 7
    amp1 = 1.0
    amp2 = 0.5

    signal = []
    for n in range(N):
        t = n / N
        val = amp1 * math.sin(2 * math.pi * freq1 * t) + amp2 * math.sin(2 * math.pi * freq2 * t)
        signal.append(val)

    print(f"  信号: {amp1}*sin(2*pi*{freq1}*t) + {amp2}*sin(2*pi*{freq2}*t)")
    print(f"  {N} 个采样点")
    print()

    spectrum = dft(signal)

    print(f"  {'频率 bin':<10s} {'|X[k]|':>10s} {'相位 (deg)':>12s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 12}")

    for k in range(N // 2 + 1):
        mag = spectrum[k].magnitude()
        if mag > 0.01:
            phase_deg = math.degrees(spectrum[k].phase())
            print(f"  k={k:<6d} {mag:>10.4f} {phase_deg:>12.2f}")

    print()
    print(f"  预期在 k={freq1} 处出现峰值 (幅值 {amp1 * N / 2:.1f})")
    print(f"  在 k={freq2} 处出现峰值 (幅值 {amp2 * N / 2:.1f})")
    print()

    # 验证逆变换完美重建
    reconstructed = idft(spectrum)
    max_err = max(abs(reconstructed[n].real - signal[n]) for n in range(N))
    print(f"  IDFT 重建误差: {max_err:.2e}")
    print(f"  完美重建: {max_err < 1e-10}")


def demo_phasor():
    print()
    print()
    print("=" * 65)
    print("  相量：用旋转复数表示信号")
    print("=" * 65)
    print()

    omega = 2 * math.pi * 3
    N = 16

    print(f"  相量: e^(i*{3}*2*pi*t)，采样 {N} 个点")
    print()

    print(f"  {'t':>6s} {'实部 (cos)':>12s} {'虚部 (sin)':>12s} {'模':>10s}")
    print(f"  {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 10}")

    for n in range(N):
        t = n / N
        phasor = euler(omega * t)
        print(f"  {t:>6.3f} {phasor.real:>12.6f} {phasor.imag:>12.6f} {phasor.magnitude():>10.6f}")

    print()
    print("  实部描绘 cos(6*pi*t)，虚部描绘 sin(6*pi*t)。")
    print("  模始终为 1——相量保持在单位圆上。")


def demo_positional_encoding():
    print()
    print()
    print("=" * 65)
    print("  Transformer 位置编码频率")
    print("=" * 65)
    print()

    d_model = 8
    max_pos = 10

    print(f"  d_model = {d_model}，展示前 {max_pos} 个位置")
    print()

    print(f"  频率 (1/10000^(2i/d)):")
    freqs = []
    for i in range(d_model // 2):
        freq = 1.0 / (10000 ** (2 * i / d_model))
        freqs.append(freq)
        print(f"    维度对 {i}: freq = {freq:.6f}")

    print()
    print(f"  PE 矩阵 (每个位置的 sin/cos 对):")
    print()

    header = "  pos"
    for i in range(d_model // 2):
        header += f"  sin_{i:d}     cos_{i:d}  "
    print(header)
    print(f"  {'-' * (5 + d_model // 2 * 20)}")

    for pos in range(max_pos):
        line = f"  {pos:>3d}"
        for i in range(d_model // 2):
            angle = pos * freqs[i]
            line += f"  {math.sin(angle):>7.4f}  {math.cos(angle):>7.4f}"
        print(line)

    print()
    print("  每个 (sin, cos) 对是 e^(i * pos * freq) 的实部和虚部。")
    print("  不同频率为每个位置生成独特的'指纹'。")


def print_summary():
    print()
    print()
    print("=" * 65)
    print("  总结")
    print("=" * 65)
    print()
    print("  1. 复数 z = a + bi 是复平面上的一个点 (a, b)。")
    print("  2. 乘法 = 旋转 + 缩放。除法 = 逆操作。")
    print("  3. 欧拉公式: e^(i*theta) = cos(theta) + i*sin(theta)。")
    print("  4. 乘以 e^(i*theta) 等价于旋转 theta 弧度。")
    print("  5. 复数乘法 = 二维旋转（与旋转矩阵等价）。")
    print("  6. DFT 将信号分解为旋转相量（单位根）。")
    print("  7. Transformer 位置编码是不同频率的复指数。")
    print("  8. RoPE 使用显式复数乘法编码位置。")
    print()


if __name__ == "__main__":
    demo_arithmetic()
    demo_polar_conversion()
    demo_euler_formula()
    demo_rotation()
    demo_roots_of_unity()
    demo_dft()
    demo_phasor()
    demo_positional_encoding()
    print_summary()
