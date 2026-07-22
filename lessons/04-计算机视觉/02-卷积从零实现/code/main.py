# main.py -- 从零实现 2D 卷积操作
# 依赖：numpy>=1.24
# 安装：pip install numpy
# 对应课程：阶段 04 · 02（卷积从零实现）

import numpy as np


# ============================================================
# 第 1 步：零填充（Zero Padding）
# ============================================================

def pad2d(x, p):
    """在输入的最后两个维度（H, W）周围填充 p 圈零。

    使用 x.shape[:-2] 使得同一函数兼容 (H, W)、(C, H, W)、(N, C, H, W)。
    """
    if p == 0:
        return x
    h, w = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out


# ============================================================
# 第 2 步：输出尺寸计算
# ============================================================

def output_size(h_in, k, p, s):
    """根据输入尺寸 H、卷积核大小 K、填充 P、步长 S 计算输出尺寸。

    公式：H_out = floor((H + 2P - K) / S) + 1
    这个公式在 CNN 架构设计中会被反复使用。
    """
    return (h_in + 2 * p - k) // s + 1


# ============================================================
# 第 3 步：嵌套循环版卷积（参考实现）
# ============================================================

def conv2d_naive(x, w, b=None, stride=1, padding=0):
    """用嵌套循环实现 2D 卷积——慢，但逻辑清晰无歧义。

    这是 torch.nn.functional.conv2d 的原理实现。
    四层嵌套循环：输出通道 → 行 → 列，加上隐式的 C_in × kh × kw 求和。

    Args:
        x: 输入张量，形状 (C_in, H, W)
        w: 卷积核，形状 (C_out, C_in, K_h, K_w)
        b: 偏置，形状 (C_out,)，可选
        stride: 步长
        padding: 填充大小

    Returns:
        输出张量，形状 (C_out, H_out, W_out)
    """
    c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w, f"输入通道数 {c_in} 与卷积核通道数 {c_in_w} 不匹配"

    x_pad = pad2d(x, padding)
    h_out = output_size(h, kh, padding, stride)
    w_out = output_size(w_in, kw, padding, stride)

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                # 确定当前窗口的左上角坐标
                hs = i * stride
                ws = j * stride
                # 提取与卷积核大小相同的输入窗口
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                # 逐元素相乘再求和，即点积操作
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out


# ============================================================
# 第 4 步：im2col 变换
# ============================================================

def im2col(x, kh, kw, stride=1, padding=0):
    """将输入的每个感受野窗口展开为矩阵的一列。

    对于 C_in=3, K=3，每列是 27 个数字。
    变换后的列矩阵形状为 (C_in * K_h * K_w, H_out * W_out)。

    这是所有快速卷积实现的核心技巧。
    """
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = output_size(h, kh, padding, stride)
    w_out = output_size(w, kw, padding, stride)

    # 每列包含一个感受野窗口的所有值
    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out


# ============================================================
# 第 5 步：im2col + 矩阵乘法快速卷积
# ============================================================

def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    """用 im2col + 矩阵乘法实现 2D 卷积。

    将四重循环替换为一次矩阵乘法：w_flat @ cols。
    GPU 上这就是 GEMM 操作，比嵌套循环快几个数量级。

    Args:
        x: 输入张量，形状 (C_in, H, W)
        w: 卷积核，形状 (C_out, C_in, K_h, K_w)
        b: 偏置，形状 (C_out,)，可选
        stride: 步长
        padding: 填充大小

    Returns:
        输出张量，形状 (C_out, H_out, W_out)
    """
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)

    # 将卷积核展平为二维：(C_out, C_in * K_h * K_w)
    w_flat = w.reshape(c_out, -1)

    # 一次矩阵乘法完成所有位置的卷积计算
    out = w_flat @ cols
    if b is not None:
        out += b[:, None]

    return out.reshape(c_out, h_out, w_out)


# ============================================================
# 第 6 步：感受野计算
# ============================================================

def receptive_field(layers):
    """计算多层堆叠后的感受野大小。

    对于步长为 1 的 L 层 K×K 卷积堆叠：
        RF = 1 + L × (K - 1)

    考虑步长时，感受野沿各层乘积增长。

    Args:
        layers: 列表，每个元素是 (kernel_size, stride) 的元组

    Returns:
        感受野大小（像素数）
    """
    rf = 1
    stride_prod = 1  # 累积步长乘积
    for k, s in layers:
        # 新增的感受野 = (K-1) × 之前所有层的累积步长
        rf = rf + (k - 1) * stride_prod
        stride_prod *= s
    return rf


# ============================================================
# 第 7 步：手设计卷积核
# ============================================================

# 五个经典卷积核——展示卷积层在训练前就能表达什么
KERNELS = {
    # 恒等核：什么都不做
    "identity": np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]], dtype=np.float32),

    # 均值模糊：3×3 窗口取平均
    "blur_3x3": np.ones((3, 3), dtype=np.float32) / 9.0,

    # 锐化：中心加权减去邻域，增强边缘对比度
    "sharpen": np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]], dtype=np.float32),

    # Sobel-X：检测垂直边缘（左暗右亮为正）
    "sobel_x": np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=np.float32),

    # Sobel-Y：检测水平边缘（上暗下亮为正）
    "sobel_y": np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]], dtype=np.float32),
}


def apply_kernel(img2d, kernel):
    """将单通道卷积核应用到 2D 图像上。

    Args:
        img2d: 二维灰度图像，形状 (H, W)
        kernel: 二维卷积核，形状 (K, K)

    Returns:
        卷积结果，形状 (H, W)
    """
    x = img2d[None].astype(np.float32)      # (1, H, W)
    w = kernel[None, None]                   # (1, 1, K, K)
    return conv2d_im2col(x, w, padding=1)[0]


# ============================================================
# 第 8 步：池化操作
# ============================================================

def max_pool2d(x, pool_size=2, stride=None):
    """最大池化：在 pool_size × pool_size 窗口中取最大值。

    最大池化保留局部最强的激活响应，常用于下采样。

    Args:
        x: 输入张量，形状 (C, H, W)
        pool_size: 池化窗口大小
        stride: 步长，默认等于 pool_size（不重叠）

    Returns:
        池化后的张量
    """
    if stride is None:
        stride = pool_size
    c, h, w = x.shape
    h_out = output_size(h, pool_size, 0, stride)
    w_out = output_size(w, pool_size, 0, stride)

    out = np.zeros((c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            window = x[:, hs:hs + pool_size, ws:ws + pool_size]
            out[:, i, j] = np.max(window, axis=(1, 2))
    return out


def avg_pool2d(x, pool_size=2, stride=None):
    """平均池化：在 pool_size × pool_size 窗口中取平均值。

    平均池化平滑局部响应，常用于全局平均池化（GAP）。

    Args:
        x: 输入张量，形状 (C, H, W)
        pool_size: 池化窗口大小
        stride: 步长，默认等于 pool_size

    Returns:
        池化后的张量
    """
    if stride is None:
        stride = pool_size
    c, h, w = x.shape
    h_out = output_size(h, pool_size, 0, stride)
    w_out = output_size(w, pool_size, 0, stride)

    out = np.zeros((c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            window = x[:, hs:hs + pool_size, ws:ws + pool_size]
            out[:, i, j] = np.mean(window, axis=(1, 2))
    return out


# ============================================================
# 辅助函数：合成测试图像
# ============================================================

def synthetic_step_image(size=16):
    """生成一个左右分界的阶跃图像——左半黑、右半白。

    用来测试边缘检测核是否正确工作。
    """
    img = np.zeros((1, size, size), dtype=np.float32)
    img[:, :, size // 2:] = 1.0
    return img


# ============================================================
# 主程序：验证与演示
# ============================================================

def test_against_naive():
    """验证 im2col 版本与嵌套循环版本输出一致。"""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (3, 16, 16)).astype(np.float32)
    w = rng.normal(0, 1, (8, 3, 3, 3)).astype(np.float32)
    b = rng.normal(0, 1, (8,)).astype(np.float32)

    y_naive = conv2d_naive(x, w, b, padding=1)
    y_im2col = conv2d_im2col(x, w, b, padding=1)
    diff = float(np.max(np.abs(y_naive - y_im2col)))
    return y_naive.shape, diff


def main():
    # --- 验证两种卷积实现的等价性 ---
    shape, diff = test_against_naive()
    print(f"卷积等价性验证: naive vs im2col   shape={shape}   max|diff|={diff:.2e}")

    # --- 在阶跃图像上测试 Sobel-X 核 ---
    x = synthetic_step_image()
    y = apply_kernel(x[0], KERNELS["sobel_x"])
    print("\nsobel_x 作用于左右阶跃图像（前 5 行）:")
    print(y[:5].round(1))

    # --- 输出尺寸速查表 ---
    print("\n输出尺寸速查表 (H=32):")
    for k, p, s in [(3, 0, 1), (3, 1, 1), (3, 1, 2), (2, 0, 2), (7, 3, 2)]:
        print(f"  K={k} P={p} S={s}  ->  H_out={output_size(32, k, p, s)}")

    # --- 感受野随深度增长 ---
    stacks = [
        [(3, 1)],
        [(3, 1), (3, 1)],
        [(3, 1), (3, 1), (3, 1)],
        [(3, 1), (3, 2), (3, 1), (3, 2)],
    ]
    print("\n感受野随深度增长:")
    for stack in stacks:
        print(f"  layers={stack}  ->  RF={receptive_field(stack)}")

    # --- 池化演示 ---
    img = synthetic_step_image(8)
    print(f"\n原始图像 (1×8×8) 中心区域:")
    print(img[0, :4, :4].round(1))
    pooled = max_pool2d(img, pool_size=2)
    print(f"最大池化后 (1×4×4):")
    print(pooled[0].round(1))


if __name__ == "__main__":
    main()
