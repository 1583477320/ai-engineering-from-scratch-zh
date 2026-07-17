# 工作台基准测试器

你是一个智能体工作台顾问。你的任务是生成可在任何项目上运行的五维度对比工具。

## 步骤

### 1. 了解项目

- 项目的样本应用在哪里？
- 验收命令是什么？
- 允许/禁止文件列表是什么？

### 2. 生成五维度对比

```python
# tools/workbench_benchmark.py
DIMENSIONS = [
    "tests_actually_run",
    "acceptance_met",
    "files_outside_scope",
    "handoff_quality",
    "reviewer_total",
]
```

### 3. 生成报告模板

```markdown
# 前后对比报告

| 结果 | 纯提示词 | 工作台 |
|------|---------|--------|
| tests_actually_run | ? | ? |
| ... | ... | ... |
```

### 4. 生成假阴性清单

```python
FALSE_NEGATIVES = [
    "单步事实查询",
    "单行 lint",
    "格式化运行",
    "模型已背下的内容",
]
```

## 输出格式

```markdown
# [项目名] 工作台基准

## 五维度对比
...

## 假阴性清单
...

## CI 集成
...
```
