# 交叉编码重排序器配方

## 两级流水线

```
双编码器检索 Top-N → 交叉编码器重排序 Top-K
```

## N 选择

N 至少 3×K；knee 在 N=20-50

## 交叉编码器输入

`[CLS] query_tokens [SEP] document_tokens [SEP]`
