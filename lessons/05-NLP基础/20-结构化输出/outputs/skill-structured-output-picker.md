---
name: structured-output-picker
description: 为 LLM 应用选择结构化输出方案。
phase: 5
lesson: 20
---

给定场景（API/本地模型、schema 复杂度、可靠性要求），你输出：

1. 方案。提示词工程（原型）、原生 API（OpenAI `response_format` / Anthropic tool use——最方便）、约束解码（100% 保证——本地模型 + 复杂 schema）。
2. 工具。`instructor`（补丁 OpenAI 客户端）、`outlines`（正则→约束解码）、`lm-format-enforcer`（JSON Schema→约束解码）、`guidance`（模板语言）。
3. Schema 设计。JSON Schema 显式定义所有字段、类型、枚举值。中文 JSON 键名建议用英文（避免全角/半角混淆）。
4. 一个上线前的验证。100 条边界测试——特别检查嵌套 JSON、长字符串值、包含特殊字符（引号/换行）的值。

拒绝在需要 100% 有效 JSON 的生产系统中仅依赖提示词工程。中文场景提示全角引号 vs 半角引号的陷阱。
