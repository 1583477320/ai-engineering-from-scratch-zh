# 语言模型评估框架配方

## 任务规范

每行一条 JSONL：

```json
{"id": "task-00", "prompt": "你的提示词", "targets": ["正确答案"], "metric": "exact_match"}
```

需要辅助数据的指标使用 extras 字段。

## 内置指标

| 指标 | 函数签名 | 返回值 |
|------|----------|--------|
| exact_match | (pred, targets) → 0/1 | 归一化后精确相等 |
| substring_contains | (pred, targets) → 0/1 | 目标子串出现在预测中 |
| multiple_choice | (pred, targets) → 0/1 | 首字母匹配 |
| rouge_l | (pred, targets) → [0,1] | LCS 的精确率/召回率 F1 |
| code_exec | (pred, targets, extras) → [0,1] | 在受限命名空间中执行并验证 |

## 适配器接口

```python
class ModelAdapter(Protocol):
    def generate(self, prompts: Sequence[str]) -> List[str]: ...
    @property
    def name(self) -> str: ...
```

将模型封装为适配器后，评估框架无需修改即可对任意模型进行评估。

## 生产原则

1. 锁定任务文件（在排行榜中记录 sha256）
2. 使用 `--include-per-example` 对比预测而非仅分数
3. 批次大小适配模型提供商的速率限制
