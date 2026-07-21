"""自动微分引擎——Value 类+反向传播+XOR MLP。"""
import math, random

class Value:
    def __init__(self, data, children=(), op=''):
        self.data=data; self.grad=0.0; self._backward=lambda:None
        self._prev=set(children); self._op=op
    def __repr__(self): return f"V({self.data:.4f})"
    def __add__(self,o):
        o=o if isinstance(o,Value) else Value(o)
        out=Value(self.data+o.data,(self,o),'+')
        def _back(): self.grad+=out.grad; o.grad+=out.grad
        out._backward=_back; return out
    def __mul__(self,o):
        o=o if isinstance(o,Value) else Value(o)
        out=Value(self.data*o.data,(self,o),'*')
        def _back(): self.grad+=o.data*out.grad; o.grad+=self.data*out.grad
        out._backward=_back; return out
    def __neg__(self): return self*-1
    def __sub__(self,o): return self+(-o)
    def __pow__(self,n):
        out=Value(self.data**n,(self,),f'^{n}')
        def _back(): self.grad+=n*self.data**(n-1)*out.grad
        out._backward=_back; return out
    def relu(self):
        out=Value(max(0,self.data),(self,),'relu')
        def _back(): self.grad+=(1.0 if out.data>0 else 0.0)*out.grad
        out._backward=_back; return out
    def tanh(self):
        t=math.tanh(self.data); out=Value(t,(self,),'tanh')
        def _back(): self.grad+=(1-t**2)*out.grad
        out._backward=_back; return out
    def backward(self):
        topo,visited=[],set()
        def build(v):
            if v not in visited: visited.add(v); [build(c) for c in v._prev]; topo.append(v)
        build(self); self.grad=1.0
        for v in reversed(topo): v._backward()

class Neuron:
    def __init__(self,n): self.w=[Value(random.uniform(-1,1)) for _ in range(n)]; self.b=Value(0)
    def __call__(self,x): return sum((wi*xi for wi,xi in zip(self.w,x)),self.b).tanh()
    def parameters(self): return self.w+[self.b]

class MLP:
    def __init__(self,sizes): self.layers=[Layer(sizes[i],sizes[i+1]) for i in range(len(sizes)-1)]
    def __call__(self,x):
        for l in self.layers: x=l(x)
        return x[0] if len(x)==1 else x
    def parameters(self): return [p for l in self.layers for p in l.parameters()]

class Layer:
    def __init__(self,ni,no): self.neurons=[Neuron(ni) for _ in range(no)]
    def __call__(self,x): return [n(x) for n in self.neurons]
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]

def main():
    random.seed(42); model=MLP([2,4,1])
    xs=[[0,0],[0,1],[1,0],[1,1]]; ys=[-1,1,1,-1]
    for step in range(100):
        preds=[model(x) for x in xs]
        loss=sum((p-y)**2 for p,y in zip(preds,ys))
        for p in model.parameters(): p.grad=0.0
        loss.backward()
        for p in model.parameters(): p.data-=0.05*p.grad
        if step%20==0: print(f"步{step:3d} 损失={loss.data:.4f}")
    print("\n预测:")
    for x,y in zip(xs,ys): print(f"  {x} → {model(x).data:.3f} (目标 {y})")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
