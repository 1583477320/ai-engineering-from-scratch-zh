"""SVD 从零实现——幂迭代+截断+伪逆。"""
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)
    return v @ M @ v, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    k = k or min(m, n)
    sigmas, us, vs = [], [], []
    A_res = A.copy().astype(float)
    for _ in range(k):
        AtA = A_res.T @ A_res
        ev, v = power_iteration(AtA, 200)
        if ev < 1e-10: break
        sigma = np.sqrt(ev)
        u = A_res @ v / sigma
        sigmas.append(sigma); us.append(u); vs.append(v)
        A_res -= sigma * np.outer(u, v)
    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))
    return U, S, V

def main():
    np.random.seed(42)
    A = np.random.randn(5, 4)
    U, S, V = svd_from_scratch(A)
    _, S_np, _ = np.linalg.svd(A, full_matrices=False)
    print(f"奇异值 (我们): {np.round(S, 4)}")
    print(f"奇异值 (NumPy): {np.round(S_np, 4)}")
    err = np.linalg.norm(A - U @ np.diag(S) @ V.T)
    print(f"重建误差: {err:.8f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
