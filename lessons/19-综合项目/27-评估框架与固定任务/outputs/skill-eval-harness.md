# 评估框架配方

## 固定任务三元组

TaskSpec(goal, setup, verifier)

## 三种验证器

- file_equals: 文件内容精确匹配
- regex_match: 文件内容正则匹配
- shell_exit_zero: 命令退出码为 0

## pass@k

```
pass@k = 1 - (1-p)^k
```
