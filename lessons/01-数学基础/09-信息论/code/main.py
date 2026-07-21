"""信息论——熵+交叉熵+KL散度+困惑度。"""
import math

def entropy(p,base=2):
    return sum(-x*math.log(x)/math.log(base) for x in p if x>0)

def cross_entropy(p,q,base=2):
    return sum(-pi*math.log(qi)/math.log(base) for pi,qi in zip(p,q) if pi>0)

def kl(p,q,base=2):
    return cross_entropy(p,q,base)-entropy(p,base)

def softmax(z):
    mx=max(z); e=[math.exp(x-mx) for x in z]; s=sum(e)
    return [xi/s for xi in e]

def main():
    print(f"公平硬币熵: {entropy([0.5,0.5]):.4f} bits")
    print(f"偏置硬币熵: {entropy([0.99,0.01]):.4f} bits")
    t=[0.7,0.2,0.1]
    print(f"CE(好): {cross_entropy(t,[0.6,0.25,0.15]):.4f}")
    print(f"CE(坏): {cross_entropy(t,[0.1,0.1,0.8]):.4f}")
    print(f"KL(好): {kl(t,[0.6,0.25,0.15]):.4f}")
    print(f"KL(坏): {kl(t,[0.1,0.1,0.8]):.4f}")
    p=softmax([2.0,1.0,0.1]); loss=-math.log(p[0])
    print(f"交叉熵: {loss:.4f} nats, 困惑度: {math.exp(loss):.2f}")
    return 0

if __name__=="__main__": import sys; sys.exit(main())
