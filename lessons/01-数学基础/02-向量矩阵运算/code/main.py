"""从零实现矩阵运算——逐元素、矩阵乘法、转置、行列式、逆。"""
class Matrix:
    def __init__(self, data):
        self.data=[list(r) for r in data]
        self.shape=(len(self.data),len(self.data[0]))
    def __add__(self,o):
        return Matrix([[self.data[i][j]+o.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])])
    def matmul(self,o):
        return Matrix([[sum(self.data[i][k]*o.data[k][j] for k in range(self.shape[1])) for j in range(o.shape[1])] for i in range(self.shape[0])])
    def transpose(self):
        return Matrix([[self.data[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])])
    def determinant(self):
        if self.shape==(2,2): return self.data[0][0]*self.data[1][1]-self.data[0][1]*self.data[1][0]
        det=0
        for j in range(self.shape[1]):
            minor=Matrix([[self.data[i][k] for k in range(self.shape[1]) if k!=j] for i in range(1,self.shape[0])])
            det+=((-1)**j)*self.data[0][j]*minor.determinant()
        return det
    def inverse_2x2(self):
        det=self.determinant()
        return Matrix([[self.data[1][1]/det,-self.data[0][1]/det],[-self.data[1][0]/det,self.data[0][0]/det]])
    @staticmethod
    def identity(n): return Matrix([[1 if i==j else 0 for j in range(n)] for i in range(n)])
    def __repr__(self): return f"Matrix({self.data})"

def relu_matrix(m):
    return Matrix([[max(0,v) for v in row] for row in m.data])

def main():
    A=Matrix([[1,2],[3,4]]); B=Matrix([[5,6],[7,8]])
    print(f"A @ B = {A.matmul(B).data}")
    print(f"det(A) = {A.determinant()}")
    I=A.matmul(A.inverse_2x2())
    print(f"A × A⁻¹ ≈ I: {all(abs(I.data[i][j]-(1 if i==j else 0))<1e-10 for i in range(2) for j in range(2))}")
    import random; random.seed(42)
    x=Matrix([[0.5],[0.8],[0.2]])
    W=Matrix([[random.uniform(-1,1) for _ in range(3)] for _ in range(2)])
    b=Matrix([[0.1],[0.1]])
    out=relu_matrix(W.matmul(x)+b)
    print(f"\n神经网络层: {x.shape} → {out.shape}")
    print(f"输出: {out.data}")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
