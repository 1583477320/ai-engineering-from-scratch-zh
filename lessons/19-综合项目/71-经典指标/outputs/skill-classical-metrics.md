# 经典指标配方

## 分词器契约

所有指标共享同一个分词器：`re.findall(r"\w+", text.lower())`

## 五个指标

| 指标 | 公式 | 适用 |
|:----|:-----|:-----|
| 精确匹配 | pred.strip() == target.strip() | 算术/MCQ |
| F1 | 2PR/(P+R) | 开放生成 |
| BLEU-4 | BP×exp(mean(log p1..p4)) | 翻译/摘要 |
| ROUGE-L | 2×P×R/(P+R) via LCS | 摘要 |
| accuracy | 单目标精确匹配 | 分类 |
