"""
图像基础操作工具库 — 从零理解图像的张量表示

本模块实现了图像基础操作的所有核心功能：
- 合成图像生成
- HWC/CHW 布局转换
- RGB 转灰度、RGB 转 HSV
- ImageNet 标准化预处理与反处理
- 插值方法对比与局部粗糙度分析

依赖：numpy>=1.24, pillow>=9.0, torch>=2.0, torchvision>=0.15
安装：pip install numpy pillow torch torchvision

对应课程：阶段 04 · 01（图像基础）
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# 常量
# ============================================================

# ImageNet 标准化参数（在 [0, 1] 范围上计算）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# 第 1 步：合成图像生成与基本属性检查
# ============================================================

def synthetic_rgb(height=128, width=192, seed=0):
    """生成一张合成的 RGB 测试图像。

    用正弦函数和线性渐变构建三个通道，使图像既有平滑区域
    也有梯度区域，方便测试各种图像操作的正确性。

    Args:
        height: 图像高度
        width: 图像宽度
        seed: 随机种子，保证结果可复现

    Returns:
        uint8 类型的 RGB 图像，形状 (height, width, 3)
    """
    rng = np.random.default_rng(seed)
    # 创建归一化坐标网格 — 用于生成渐变纹理
    yy, xx = np.meshgrid(
        np.linspace(0, 1, height),
        np.linspace(0, 1, width),
        indexing="ij"
    )
    # R 通道：水平方向的正弦波，产生条纹纹理
    r = (np.sin(xx * 6) * 0.5 + 0.5) * 255
    # G 通道：从上到下的线性渐变
    g = yy * 255
    # B 通道：左下角亮、右上角暗的二维渐变
    b = (1 - yy) * xx * 255
    # 加入少量高斯噪声模拟真实传感器噪声
    noise = rng.normal(0, 6, (height, width, 3))
    rgb = np.stack([r, g, b], axis=-1) + noise
    return np.clip(rgb, 0, 255).astype(np.uint8)


def inspect(arr, label="image"):
    """打印图像张量的基本信息。

    一次性输出 dtype、形状、值范围、每通道均值。
    这是调试图像流水线的第一行代码。

    Args:
        arr: 图像张量，2D（灰度）或 3D（多通道）
        label: 打印时的名称
    """
    if arr.ndim == 2:
        # 灰度图
        print(f"[{label}] dtype={arr.dtype} shape={arr.shape} "
              f"min={arr.min()} max={arr.max()} mean={float(arr.mean()):.2f}")
        return
    # 多通道图：打印每通道均值
    per_channel_mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0).round(2).tolist()
    print(f"[{label}] dtype={arr.dtype} shape={arr.shape} "
          f"min={arr.min()} max={arr.max()} "
          f"per-channel mean={per_channel_mean}")


# ============================================================
# 第 2 步：HWC ↔ CHW 布局转换
# ============================================================

def hwc_to_chw(arr):
    """将 HWC 布局的图像转为 CHW 布局。

    PyTorch 和 cuDNN 使用 CHW，而 PIL 和磁盘文件使用 HWC。
    这个转换是视觉流水线中执行频率最高的操作之一。
    """
    return arr.transpose(2, 0, 1)


def chw_to_hwc(arr):
    """将 CHW 布局的图像转回 HWC 布局。"""
    return arr.transpose(1, 2, 0)


# ============================================================
# 第 3 步：颜色空间转换
# ============================================================

def rgb_to_grayscale(rgb):
    """RGB 转灰度图。

    使用 ITU-R BT.601 加权系数，模拟人眼亮度感知。
    人眼对绿色最敏感（权重 0.587），对蓝色最不敏感（权重 0.114）。

    如果用等权平均 (R+G+B)/3，绿色通道的信息会被低估，
    导致图像看起来比实际暗。

    Args:
        rgb: uint8 RGB 图像，形状 (H, W, 3)

    Returns:
        uint8 灰度图，形状 (H, W)
    """
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # 矩阵乘法一次性完成加权求和
    return (rgb.astype(np.float32) @ weights).astype(np.uint8)


def rgb_to_hsv(rgb):
    """RGB 转 HSV 颜色空间。

    HSV 将颜色信息分解为三个容易理解的分量：
    - H（色相）：0-360 度，表示颜色种类（红=0, 绿=120, 蓝=240）
    - S（饱和度）：0-1，表示颜色纯度（0=灰色, 1=纯色）
    - V（明度）：0-1，表示颜色亮度（0=黑色, 1=最亮）

    这在颜色分割和基于颜色的图像处理中非常有用——
    在 HSV 空间中，可以根据色相精确地选中某种颜色，
    而不受明度和饱和度的影响。

    Args:
        rgb: uint8 RGB 图像，形状 (H, W, 3)

    Returns:
        float32 HSV 图像，形状 (H, W, 3)，H 在 [0, 360)，S 和 V 在 [0, 1]
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    # 计算色相：根据最大值所在的通道选择不同的偏移
    h = np.zeros_like(cmax)
    mask = delta > 0  # 无色区域（灰度）的色相无意义

    # 用 argmax 避免浮点相等的边界问题
    argmax = np.argmax(rgb_f, axis=-1)
    rmax = mask & (argmax == 0)
    gmax = mask & (argmax == 1)
    bmax = mask & (argmax == 2)

    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    h = h * 60.0  # 转为角度制

    # 饱和度：色度与明度的比值
    s = np.where(cmax > 0, delta / cmax, 0)
    # 明度：三个通道中的最大值
    v = cmax

    return np.stack([h, s, v], axis=-1)


# ============================================================
# 第 4 步：ImageNet 标准化预处理与反处理
# ============================================================

def preprocess_imagenet(rgb_uint8):
    """将 uint8 RGB 图像转换为 ImageNet 模型的标准输入。

    三步操作：除以 255 → 减均值除标准差 → HWC 转 CHW。
    这是每个预训练模型前必须执行的标准流水线。

    Args:
        rgb_uint8: uint8 RGB 图像，形状 (H, W, 3)

    Returns:
        float32 标准化张量，形状 (3, H, W)
    """
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.transpose(2, 0, 1)
    return x


def deprocess_imagenet(chw_float32):
    """将标准化 CHW 张量还原为 uint8 RGB 图像。

    这是 preprocess_imagenet 的逆操作。
    用于验证预处理的可逆性，或在可视化时将标准化张量转回可显示的图像。

    Args:
        chw_float32: ImageNet 标准化张量，形状 (3, H, W)

    Returns:
        uint8 RGB 图像，形状 (H, W, 3)
    """
    x = chw_float32.transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    # 数值裁剪 + 转为 uint8
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


# ============================================================
# 第 5 步：插值方法对比
# ============================================================

def resize_compare(arr, scale=3):
    """使用三种不同插值方法缩放图像。

    比较最近邻、双线性和双三次插值的输出差异。

    Args:
        arr: uint8 图像，形状 (H, W, 3)
        scale: 缩放倍数（>1 为上采样，<1 为下采样）

    Returns:
        dict: 方法名 → 缩放后的图像 numpy 数组
    """
    target = (arr.shape[1] * scale, arr.shape[0] * scale)
    methods = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }
    results = {}
    for name, filt in methods.items():
        resized = Image.fromarray(arr).resize(target, filt)
        results[name] = np.asarray(resized)
    return results


def local_roughness(x):
    """计算图像局部粗糙度。

    粗糙度是水平和垂直方向梯度绝对值的均值。
    值越大表示图像中的边缘越多、越明显。

    这个指标常用于量化比较不同插值方法的效果：
    - 最近邻插值保留边缘 → 粗糙度最高
    - 双线性插值模糊边缘 → 粗糙度最低
    - 双三次插值居中

    Args:
        x: 输入图像，形状 (H, W) 或 (H, W, C)

    Returns:
        float: 粗糙度值
    """
    # 转为 float32 防止 uint8 溢出
    f = x.astype(np.float32)
    # 垂直方向梯度：相邻行的差值
    gy = np.diff(f, axis=0)
    # 水平方向梯度：相邻列的差值
    gx = np.diff(f, axis=1)
    return float(np.abs(gy).mean() + np.abs(gx).mean())


# ============================================================
# 主程序：演示完整流程
# ============================================================

def main():
    print("=" * 60)
    print("图像基础操作演示")
    print("=" * 60)
    print()

    # ----------------------------------------------------------------
    # 第 1 步：生成合成图像并检查属性
    # ----------------------------------------------------------------
    print("【第 1 步】生成合成图像")
    print("-" * 40)
    img = synthetic_rgb(height=128, width=192)
    inspect(img, "合成图像")
    print(f"像素 (0,0): {img[0, 0]}    # [R, G, B]")
    print()

    # ----------------------------------------------------------------
    # 第 2 步：HWC ↔ CHW 布局转换
    # ----------------------------------------------------------------
    print("【第 2 步】HWC ↔ CHW 布局转换")
    print("-" * 40)
    # 拆分三个通道
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    print(f"R 通道均值: {R.mean():.1f}")
    print(f"G 通道均值: {G.mean():.1f}")
    print(f"B 通道均值: {B.mean():.1f}")

    img_chw = hwc_to_chw(img)
    print(f"\nHWC 形状: {img.shape}")
    print(f"CHW 形状: {img_chw.shape}")
    print(f"CHW 通道 0 (R 通道) 形状: {img_chw[0].shape}")

    # 验证往返可逆性
    img_hwc_back = chw_to_hwc(img_chw)
    assert np.array_equal(img, img_hwc_back), "HWC ↔ CHW 往返转换失败"
    print("HWC ↔ CHW 往返可逆性: 验证通过")
    print()

    # ----------------------------------------------------------------
    # 第 3 步：颜色空间转换
    # ----------------------------------------------------------------
    print("【第 3 步】颜色空间转换")
    print("-" * 40)

    gray = rgb_to_grayscale(img)
    print(f"灰度图形状: {gray.shape}, 范围: [{gray.min()}, {gray.max()}]")

    hsv = rgb_to_hsv(img)
    print(f"HSV 形状: {hsv.shape}")
    print(f"色相范围: [{hsv[..., 0].min():.1f}, {hsv[..., 0].max():.1f}] 度")
    print(f"饱和度范围: [{hsv[..., 1].min():.2f}, {hsv[..., 1].max():.2f}]")
    print(f"明度范围: [{hsv[..., 2].min():.2f}, {hsv[..., 2].max():.2f}]")
    print()

    # ----------------------------------------------------------------
    # 第 4 步：ImageNet 预处理与可逆性验证
    # ----------------------------------------------------------------
    print("【第 4 步】ImageNet 预处理与可逆性验证")
    print("-" * 40)

    x = preprocess_imagenet(img)
    print(f"预处理后形状: {x.shape}     # (C, H, W)")
    print(f"预处理后 dtype: {x.dtype}")
    print(f"每通道均值: {x.mean(axis=(1, 2)).round(3).tolist()}")
    print(f"每通道标准差: {x.std(axis=(1, 2)).round(3).tolist()}")

    # 验证 preprocess ↔ deprocess 的可逆性
    roundtrip = deprocess_imagenet(x)
    max_diff = int(np.abs(roundtrip.astype(int) - img.astype(int)).max())
    print(f"\n预处理 ↔ 反处理 往返最大像素差: {max_diff}")
    assert max_diff <= 1, f"往返误差过大: {max_diff}"
    print("预处理 ↔ 反处理 可逆性: 验证通过")
    print()

    # ----------------------------------------------------------------
    # 第 5 步：三种插值方法对比
    # ----------------------------------------------------------------
    print("【第 5 步】三种插值方法对比（3 倍上采样）")
    print("-" * 40)

    results = resize_compare(img, scale=3)
    for name, out in results.items():
        roughness = local_roughness(out)
        print(f"{name:>10}  形状={out.shape}  粗糙度={roughness:.2f}")

    print()
    print("=" * 60)
    print("所有演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
