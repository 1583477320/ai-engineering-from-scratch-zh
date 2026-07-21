"""从零实现线性代数——向量、点积、投影、Gram-Schmidt。"""
import math

class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)
    def __add__(self, other):
        return Vector([a+b for a,b in zip(self.components, other.components)])
    def dot(self, other):
        return sum(a*b for a,b in zip(self.components, other.components))
    def magnitude(self):
        return sum(x**2 for x in self.components)**0.5
    def normalize(self):
        m=self.magnitude()
        return Vector([x/m for x in self.components])
    def cosine_similarity(self, other):
        return self.dot(other)/(self.magnitude()*other.magnitude())
    def __repr__(self):
        return f"Vector({self.components})"

def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v
        for u in basis:
            p = project(w, u)
            w = Vector([x - p.components[i] for i,x in enumerate(w.components)])
        if w.magnitude() > 1e-10:
            basis.append(w.normalize())
    return basis

def main():
    a = Vector([1, 2, 3]); b = Vector([4, 5, 6])
    print(f"a · b = {a.dot(b)}")
    print(f"cos sim = {a.cosine_similarity(b):.4f}")

    v1, v2, v3 = Vector([1,0,0]), Vector([1,1,0]), Vector([1,1,1])
    basis = gram_schmidt([v1, v2, v3])
    print("\nGram-Schmidt:")
    for i, u in enumerate(basis):
        print(f"  u{i+1} = {u}, |u{i+1}| = {u.magnitude():.6f}")

    print("\n正交性验证:")
    for i in range(len(basis)):
        for j in range(i+1, len(basis)):
            print(f"  u{i+1} · u{j+1} = {basis[i].dot(basis[j]):.6f}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
