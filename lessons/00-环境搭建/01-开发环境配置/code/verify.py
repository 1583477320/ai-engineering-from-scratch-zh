"""环境验证脚本——检查 Python/Node/Rust/GPU。"""
import sys, subprocess, os

def check(desc, ok):
    print(f"  {'✓' if ok else '✗'} {desc}")
    return ok

def main():
    print("=== 环境验证 ===")
    ok = True
    ok &= check("Python 3.11+", sys.version_info >= (3, 11))
    try:
        import numpy as np
        a = np.array([1,2,3])
        ok &= check("NumPy 可用", np.dot(a, a) == 14)
    except: ok &= check("NumPy 可用", False)
    try:
        import torch
        ok &= check("PyTorch 可用", True)
        ok &= check(f"CUDA 可用: {torch.cuda.is_available()}", True)
    except: ok &= check("PyTorch 可用", False)
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, text=True)
        ok &= check(f"Node.js {r.stdout.strip()}", r.returncode == 0)
    except: ok &= check("Node.js", False)
    try:
        r = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        ok &= check(f"Rust {r.stdout.strip()}", r.returncode == 0)
    except: ok &= check("Rust", False)
    print(f"\n{'✓ 全部通过' if ok else '✗ 有失败项目'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
