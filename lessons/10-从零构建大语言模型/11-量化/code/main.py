# 量化——从 FP32 到 INT4 的精度与压缩权衡

import numpy as np


# ============================================================================
# 第 1 步：数值格式展示
# ============================================================================

def show_float_format(value):
    """展示 FP32/FP16/BF16 的位表示。"""
    fp32 = np.float32(value)
    bits = fp32.view(np.uint32)
    s = (bits >> 31) & 1
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    print(f"  FP32: {value} -> sign={s} exp={e:3d}={format(e,'08b')} man={format(m,'023b')}")

    fp16 = np.float16(value)
    bits16 = fp16.view(np.uint16)
    s16 = (bits16 >> 15) & 1
    e16 = (bits16 >> 10) & 0x1F
    m16 = bits16 & 0x3FF
    print(f"  FP16: {float(fp16):.4f} -> sign={s16} exp={e16:2d}={format(e16,'05b')} man={format(m16,'010b')}")


# ============================================================================
# 第 2 步：对称量化和非对称量化
# ============================================================================

def quantize_symmetric(tensor, bits=8):
    """对称量化——以零点为中心。"""
    qmax = 2**(bits-1) - 1
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int32), 1.0
    scale = abs_max / qmax
    q = np.clip(np.round(tensor / scale), -qmax, qmax).astype(np.int32)
    return q, scale


def quantize_asymmetric(tensor, bits=8):
    """非对称量化——支持偏移零点。"""
    qmin, qmax = 0, 2**bits - 1
    t_min, t_max = np.min(tensor), np.max(tensor)
    if t_max == t_min:
        return np.zeros_like(tensor, dtype=np.int32), 1.0, 0
    scale = (t_max - t_min) / qmax
    zp = np.round(qmin - t_min / scale)
    zp = max(qmin, min(qmax, zp))
    q = np.clip(np.round(tensor / scale + zp), qmin, qmax).astype(np.int32)
    return q, scale, int(zp)


def dequantize(q, scale, zp=0):
    return (q.astype(np.float64) - zp) * scale


def quantize_per_channel(tensor, bits=8, axis=0):
    """逐通道量化——每通道独立缩放。"""
    qmax = 2**(bits-1) - 1
    if axis == 0:
        abs_max = np.max(np.abs(tensor), axis=1, keepdims=True)
    else:
        abs_max = np.max(np.abs(tensor), axis=0, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = abs_max / qmax
    q = np.clip(np.round(tensor / scales), -qmax, qmax).astype(np.int32)
    return q, scales


def quantization_error(orig, recon):
    """量化误差指标。"""
    mse = np.mean((orig - recon)**2)
    snr = 10 * np.log10(np.mean(orig**2) / (mse + 1e-20))
    cs = np.dot(orig.flatten(), recon.flatten()) / \
         (np.linalg.norm(orig.flatten()) * np.linalg.norm(recon.flatten()) + 1e-20)
    return {"mse": mse, "snr_db": float(snr), "cosine_sim": float(cs)}


# ============================================================================
# 第 3 步：位宽扫描
# ============================================================================

def bit_width_sweep(tensor):
    print(f"\n位宽扫描 (shape {tensor.shape}):")
    print(f"  {'Bits':>5} {'MSE':>14} {'SNR(dB)':>10} {'余弦相似':>10} {'压缩比':>10}")
    for bits in [2, 4, 8, 16]:
        q, s = quantize_per_channel(tensor, bits)
        recon = dequantize(q.squeeze(), s.squeeze()).reshape(tensor.shape)
        err = quantization_error(tensor, recon)
        ratio = 32.0 / bits
        print(f"  {bits:>5} {err['mse']:>14.8f} {err['snr_db']:>10.2f} {err['cosine_sim']:>10.6f} {ratio:>9.1f}x")


# ============================================================================
# 第 4 步：GPTQ 和 AWQ 模拟
# ============================================================================

def simulate_gptq(weight, calib_inputs, bits=4):
    """模拟 GPTQ——基于 Hessian 矩阵的量化。"""
    d_in, d_out = weight.shape
    qmax = 2**(bits-1) - 1
    H = sum(x.T @ x for x in calib_inputs) / len(calib_inputs)
    H += np.eye(d_in) * 1e-4
    q_weight = np.zeros_like(weight)
    W = weight.copy()
    for col in range(d_out):
        abs_max = np.max(np.abs(W[:, col]))
        if abs_max == 0:
            continue
        scale = abs_max / qmax
        q = np.clip(np.round(W[:, col] / scale), -qmax, qmax).astype(np.int32)
        q_weight[:, col] = q
        err = W[:, col] - q * scale
        for nc in range(col+1, min(col+3, d_out)):
            W[:, nc] += err * 0.1
    return q_weight


# ============================================================================
# 第 5 步：显存计算器
# ============================================================================

def memory_table():
    print(f"\n模型显存需求:")
    print(f"  {'模型':<8} {'FP32':>8} {'FP16':>8} {'INT8':>8} {'INT4':>8}")
    for name, params in [("7B",7), ("13B",13), ("70B",70), ("405B",405)]:
        fp32 = params * 1e9 * 4 / 1e9
        fp16 = params * 1e9 * 2 / 1e9
        int8 = params * 1e9 * 1 / 1e9
        int4 = params * 1e9 * 0.5 / 1e9
        print(f"  {name:<8} {fp32:>7.1f}G {fp16:>7.1f}G {int8:>7.1f}G {int4:>7.1f}G")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("量化演示")
    print("=" * 60)

    # 1. 数值格式
    print("\n1. 数值格式展示")
    show_float_format(3.14159)
    show_float_format(0.1)

    # 2. 量化方法对比
    print("\n2. 量化方法对比 (INT8)")
    w = np.random.randn(64, 128) * 0.02
    q_sym, s_sym = quantize_symmetric(w, 8)
    q_asym, s_asym, zp = quantize_asymmetric(w, 8)
    recon_sym = dequantize(q_sym, s_sym)
    recon_asym = dequantize(q_asym, s_asym, zp)
    e_sym = quantization_error(w, recon_sym)
    e_asym = quantization_error(w, recon_asym)
    print(f"  对称量化: MSE={e_sym['mse']:.8f}, SNR={e_sym['snr_db']:.2f}dB")
    print(f"  非对称:   MSE={e_asym['mse']:.8f}, SNR={e_asym['snr_db']:.2f}dB")

    # 3. 位宽扫描
    print("\n3. 位宽扫描")
    bit_width_sweep(np.random.randn(32, 64) * 0.02)

    # 4. GPTQ 模拟
    print("\n4. GPTQ 模拟 (INT4)")
    w = np.random.randn(128, 256) * 0.02
    c = [np.random.randn(1, 128) * 0.1 for _ in range(16)]
    q_gptq = simulate_gptq(w, c, 4)
    e_gptq = quantization_error(w, q_gptq.astype(np.float64))
    print(f"  MSE={e_gptq['mse']:.8f}, SNR={e_gptq['snr_db']:.2f}dB")

    # 5. 显存表
    memory_table()

    print("\n完成！")
