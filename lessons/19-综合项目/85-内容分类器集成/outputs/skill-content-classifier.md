# 内容分类器集成配方

## 三个分类器

toxicity → 毒性关键字+否定窗口
pii → 邮箱/电话/SSN/信用卡正则
instruction-leakage → 系统提示词trigram重叠

## 策略路由

high→block / medium→redact / low→warn / none→log
