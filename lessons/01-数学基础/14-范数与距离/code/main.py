"""距离度量——L1+L2+余弦+Jaccard+编辑距离。"""
import math
def l1_dist(a,b): return sum(abs(x-y) for x,y in zip(a,b))
def l2_dist(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def cosine_sim(a,b):
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x**2 for x in a)); nb=math.sqrt(sum(x**2 for x in b))
    return dot/(na*nb) if na*nb else 0
def jaccard_sim(a,b):
    sa,sb=set(a),set(b); return len(sa&sb)/max(len(sa|sb),1)
def edit_distance(a,b):
    n,m=len(a),len(b); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=dp[i-1][j-1] if a[i-1]==b[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[n][m]

def main():
    a,b=(1,2,3),(4,0,6)
    print(f"L1={l1_dist(a,b)} L2={l2_dist(a,b):.4f}")
    print(f"余弦={cosine_sim(a,b):.4f}")
    print(f"Jaccard={jaccard_sim(['cat','dog'],['cat','bird','fish']):.3f}")
    print(f"编辑距离={edit_distance('kitten','sitting')}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
