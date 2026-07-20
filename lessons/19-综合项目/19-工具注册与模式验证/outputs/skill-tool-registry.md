# 工具注册表技能

## 目标
构建类型化工具注册表+JSON Schema验证器，为调度器提供可靠的基础。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 模式覆盖 | 8关键字子集覆盖所有注册工具 |
| 20 | 错误路径精确度 | JSON指针格式，模型可一轮纠正 |
| 20 | 重复注册保护 | 默认拒绝+override机制 |
| 20 | 验证纯净性 | 无I/O、无时间、可重放 |
| 15 | API一致性 | register/get/validate/names接口完整 |

## 构建检查清单

- [ ] ToolRecord数据类（name/schema/handler/idempotent/timeout）
- [ ] ToolRegistry类（register/get/names/validate）
- [ ] 8关键字JSON Schema验证器
- [ ] JSON指针错误路径
- [ ] 重复注册拒绝
- [ ] 模式结构验证（validate_schema_shape）
- [ ] 演示用例+错误路径展示
