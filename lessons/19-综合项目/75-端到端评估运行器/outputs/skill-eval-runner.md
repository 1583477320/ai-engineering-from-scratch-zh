# 端到端评估运行器配方

## 适配器接口

```python
class ModelAdapter:
    model_id: str
    def generate(self, prompt: str) -> Generation: ...
```

## 流水线

验证 → 渲染 → 生成 → 后处理 → 评分 → 聚合 → 排行榜

## 自终止

达标(规则>随机) → exit 0；不达标 → exit 1
