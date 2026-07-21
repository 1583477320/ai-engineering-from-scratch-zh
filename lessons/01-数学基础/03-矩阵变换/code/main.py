"""矩阵变换——旋转、缩放、剪切、特征值。"""
import math

def rotation_2d(theta):
    c,s=math.cos(theta),math.sin(theta); return [[c,-s],[s,c]]
def scaling_2d(sx,sy): return [[sx,0],[0,sy]]
def shearing_2d(kx,ky): return [[1,kx],[ky,1]]
def mat_vec_mul(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def mat_mul(A,B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(A[0]))) for j in range(len(B[0]))] for i in range(len(A))]

def eigenvalues_2x2(M):
    a,b=M[0]; c,d=M[1]; trace,det=a+d,a*d-b*c
    disc=trace**2-4*det
    if disc<0: return (complex(trace/2,(-disc)**0.5/2),complex(trace/2,-(-disc)**0.5/2))
    sd=disc**0.5; return ((trace+sd)/2,(trace-sd)/2)

def main():
    R=rotation_2d(math.pi/4); S=scaling_2d(2,3)
    print(f"旋转45° (1,0)→ {mat_vec_mul(R,[1.0,0.0])}")
    print(f"缩放(2,3) (1,1)→ {mat_vec_mul(S,[1.0,1.0])}")
    print(f"先旋转后缩放: {mat_vec_mul(S,mat_vec_mul(R,[1.0,0.0]))}")
    print(f"先缩放后旋转: {mat_vec_mul(R,mat_vec_mul(S,[1.0,0.0]))}")

    A=[[2,1],[1,2]]
    vals=eigenvalues_2x2(A)
    print(f"\n特征值: {vals[0]:.4f}, {vals[1]:.4f}")
    print(f"沿[1,1]拉伸{vals[0]:.0f}倍, [1,-1]不变")
    return 0

if __name__=="__main__":
    import sys; sys.exit(main())
