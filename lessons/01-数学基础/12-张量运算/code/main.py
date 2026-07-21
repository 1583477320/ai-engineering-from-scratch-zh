"""张量运算——广播+einsum+注意力形状。"""
import numpy as np

def main():
    print("=== 广播 ===")
    acts=np.random.randn(4,3); bias=np.array([0.1,0.2,0.3])
    print(f"激活{acts.shape}+偏置{bias.shape}={(acts+bias).shape}")
    a=np.array([1,2,3]).reshape(-1,1); b=np.array([10,20,30,40]).reshape(1,-1)
    print(f"外积{a.shape}×{b.shape}={(a*b).shape}")
    print("\n=== Einsum ===")
    A=np.random.randn(3,4); B=np.random.randn(4,5)
    print(f"矩阵乘法: {np.einsum('ik,kj->ij',A,B).shape}")
    ba=np.random.randn(4,3,5); bb=np.random.randn(4,5,2)
    print(f"批处理: {np.einsum('bij,bjk->bik',ba,bb).shape}")
    print("\n=== 注意力形状 ===")
    B,H,T,D=2,4,8,16; X=np.random.randn(B,T,H*D); Wq=np.random.randn(H*D,H*D)*.02
    Q=np.einsum("bte,ek->btk",X,Wq).reshape(B,T,H,D).transpose(0,2,1,3)
    print(f"Q: {Q.shape}")
    scores=np.einsum("bhtd,bhsd->bhts",Q,Q)/np.sqrt(D)
    print(f"注意力: {scores.shape}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
