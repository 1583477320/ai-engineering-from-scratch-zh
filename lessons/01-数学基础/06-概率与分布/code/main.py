"""概率与分布——PMF/PDF+softmax+交叉熵。"""
import math, random

def softmax(logits):
    mx=max(logits); shifted=[z-mx for z in logits]
    exps=[math.exp(z) for z in shifted]; s=sum(exps)
    return [e/s for e in exps]

def cross_entropy(logits, target):
    mx=max(logits); lse=mx+math.log(sum(math.exp(z-mx) for z in logits))
    return -sum(t*math.exp(z-lse) for t,z in zip(target,logits)) if isinstance(target,list) else -logits[target]+lse

def log_softmax(logits):
    mx=max(logits); lse=mx+math.log(sum(math.exp(z-mx) for z in logits))
    return [z-lse for z in logits]

def main():
    logits=[2.0,1.0,0.1]
    probs=softmax(logits)
    print(f"softmax: {[f'{p:.4f}' for p in probs]} sum={sum(probs):.4f}")
    ce=cross_entropy([2.0,1.0,0.1],0)
    print(f"交叉熵(target=0): {ce:.4f}")
    samples=[]
    for _ in range(10000):
        u1,u2=random.random(),random.random()
        samples.append(math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2))
    mu=sum(samples)/len(samples)
    var=sum((s-mu)**2 for s in samples)/len(samples)
    print(f"正态采样: 均值={mu:.4f} 方差={var:.4f}")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
