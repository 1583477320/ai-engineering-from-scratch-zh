# 提示词注入检测器配方

## 三层

1. 归一化（清除零宽字符，解码 base64/rot13）
2. 子串规则（"ignore all"）
3. 正则规则（"\bignor\w*\s+(all|prior)"）

## 格式

Rule(name, category, score, substring/regex)
