# 投影层与模态对齐配方

## MLP 投影器

```
Linear(768, 1024) → GELU → Linear(1024, 512)
参数: ~1.3M
```

## 余弦对齐损失

```python
loss = 1 - cosine_similarity(normalize(img_emb), normalize(txt_emb))
```

## 冻结策略

- 视觉编码器: 冻结
- 文本嵌入表: 冻结
- 投影器: 训练
