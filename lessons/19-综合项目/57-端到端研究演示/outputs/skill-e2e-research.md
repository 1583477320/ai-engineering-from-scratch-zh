# 端到端研究演示配方

## 组合结构

种子 → 调度器 → 运行器 → 总线 → 选择器 → 评审 → 写作 → 报告

## 失败模式

- 无触发: NoTriggerError
- 实验失败: terminal != "ok"
- 写作验证失败: PaperValidationError

## 确定性保证

固定种子 → 相同报告
