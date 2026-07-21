"""PCA 从零实现。"""
import numpy as np

class PCA:
    def __init__(self,n):
        self.n=n; self.components=None; self.mean=None
        self.eigenvalues=None; self.explained_ratio=None
    def fit(self,X):
        self.mean=np.mean(X,axis=0); Xc=X-self.mean
        cov=np.cov(Xc,rowvar=False)
        vals,vecs=np.linalg.eigh(cov)
        idx=np.argsort(vals)[::-1]; vals,vecs=vals[idx],vecs[:,idx]
        self.components=vecs[:,:self.n].T
        self.eigenvalues=vals[:self.n]
        self.explained_ratio=vals[:self.n]/np.sum(vals)
        return self
    def transform(self,X): return (X-self.mean)@self.components.T
    def fit_transform(self,X): self.fit(X); return self.transform(X)

def main():
    np.random.seed(42)
    t=np.random.uniform(0,2*np.pi,500)
    X=np.column_stack([3*np.cos(t)+np.random.normal(0,.2,500),
                       3*np.sin(t)+np.random.normal(0,.2,500),
                       .5*np.cos(t)+.3*np.sin(t)+np.random.normal(0,.1,500)])
    pca=PCA(2); Xr=pca.fit_transform(X)
    print(f"原始: {X.shape} → 降维: {Xr.shape}")
    print(f"方差解释: {pca.explained_ratio}")
    print(f"总方差保留: {sum(pca.explained_ratio):.4f}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
