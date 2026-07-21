"""Jupyter 笔记本常用技巧。"""
import numpy as np

def demo_timeit():
    """对比 numpy vs 列表推导的性能。"""
    print("创建 100,000 个随机数:")
    start = time.time()
    _ = [random.random() for _ in range(100_000)]
    py_time = time.time() - start
    start = time.time()
    _ = np.random.randn(100_000)
    np_time = time.time() - start
    print(f"  Python 列表: {py_time*1000:.1f}ms")
    print(f"  NumPy 数组:  {np_time*1000:.1f}ms")
    print(f"  加速比: {py_time/np_time:.0f}x")

if __name__ == "__main__":
    import time
    demo_timeit()
