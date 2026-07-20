# ViT 编码器配方

## Pre-LN 块

```python
x = x + attn(ln1(x))
x = x + ffn(ln2(x))
```

## ViT-Base 规格

- depth=12, heads=12, dim=768
- FFN expansion: 4x
- 总参数: ~86M

## CLS 池化

```python
return vit(front_end(x))[:, 0, :]  # CLS 位置
```
