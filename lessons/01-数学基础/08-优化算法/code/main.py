"""优化器从零实现——GD、SGD+Momentum、Adam。"""
import math

def rosenbrock(p):
    x,y=p; return (1-x)**2+100*(y-x**2)**2

def rosenbrock_grad(p):
    x,y=p; return [-2*(1-x)+200*(y-x**2)*(-2*x), 200*(y-x**2)]

class GradientDescent:
    def __init__(self,lr=0.001): self.lr=lr
    def step(self,p,g): return [pi-self.lr*gi for pi,gi in zip(p,g)]

class SGDMomentum:
    def __init__(self,lr=0.001,mom=0.9): self.lr=lr; self.beta=mom; self.v=None
    def step(self,p,g):
        if self.v is None: self.v=[0.0]*len(p)
        self.v=[self.beta*v+gi for v,gi in zip(self.v,g)]
        return [pi-self.lr*v for pi,v in zip(p,self.v)]

class Adam:
    def __init__(self,lr=0.001,b1=0.9,b2=0.999,eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m=None; self.v=None; self.t=0
    def step(self,p,g):
        if self.m is None: self.m=[0.0]*len(p); self.v=[0.0]*len(p)
        self.t+=1
        self.m=[self.b1*m+(1-self.b1)*gi for m,gi in zip(self.m,g)]
        self.v=[self.b2*v+(1-self.b2)*gi**2 for v,gi in zip(self.v,g)]
        mh=[m/(1-self.b1**self.t) for m in self.m]
        vh=[v/(1-self.b2**self.t) for v in self.v]
        return [pi-self.lr*mh/(vh**.5+1e-8) for pi,mh,vh in zip(p,mh,vh)]

def optimize(opt,f,grad,start,steps=5000):
    p=list(start)
    for _ in range(steps): p=opt.step(p,grad(p))
    return p

def main():
    s=[-1.0,1.0]
    for nm,opt in [("GD",GradientDescent(0.0005)),("SGD+M",SGDMomentum(0.0001,0.9)),("Adam",Adam(0.01))]:
        r=optimize(opt,rosenbrock,rosenbrock_grad,s)
        print(f"{nm:6s} → x={r[0]:.6f} y={r[1]:.6f} loss={rosenbrock(r):.8f}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
