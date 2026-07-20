# 混合检索配方

## RRF 融合

```python
def rrf(ranked, k=60):
    scores = defaultdict(float)
    for mod, ranked_list in ranked.items():
        for rank, (did, _) in enumerate(ranked_list, 1):
            scores[did] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

## BM25 参数

- k1=1.5 (词频饱和)
- b=0.75 (长度归一化)

## 并行执行

BM25 和稠密并行，融合是常数时间合并。
