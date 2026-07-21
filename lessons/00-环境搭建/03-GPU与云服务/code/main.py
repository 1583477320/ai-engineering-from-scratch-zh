"""GPU 基准测试——CPU vs GPU 矩阵乘法。"""
import torch, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

size = 5000
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU 矩阵乘法 ({size}x{size}): {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")
    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU 矩阵乘法 ({size}x{size}): {gpu_time:.3f}s")
    print(f"加速比: {cpu_time / gpu_time:.0f}x")
else:
    print("GPU 不可用——在 CPU 上运行")

print(f"\nCUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"显存: {props.total_memory / 1e9:.1f} GB")
    print(f"可容纳最大模型(fp16): ~{(props.total_memory * 0.5) / 1e6:.0f}M 参数")
