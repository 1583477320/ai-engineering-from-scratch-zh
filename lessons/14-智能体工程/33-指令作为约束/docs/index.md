# 智能体指令作为可执行约束

> 用散文写的指令是愿望。作为约束写的指令是测试。工作台将每个规则转化为智能体在运行时可以检查、审查员在事后可以验证的东西。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 14 · 32（最小工作台）| **时间：** ~50 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 区分路由散文（建议）和操作规则（约束）
- [ ] 实现可执行的指令约束——智能体运行时验证
- [ ] 理解为什么"测试即指令"比"散文即指令"更可靠
- [ ] 设计指令编写的最佳实践

---

## 1. 问题

大多数指令是散文——"请保持回答简洁"、"尽量有帮助"。但智能体不理解"尽量"——它需要**可执行的条件**。"回答不超过 100 字"是约束，"尽量简洁"是愿望。

**约束成为可执行的测试**——智能体可以在运行时检查自己是否遵守了规则。

---

## 2. 概念

### 2.1 散文 vs 约束

| 散文（愿望） | 约束（可执行） |
|------------|--------------|
| "保持简洁" | "回答不超过 100 字" |
| "不要有偏见" | "回答必须包含正反两面论据" |
| "注意安全" | "不要输出任何联系信息" |

### 2.2 可执行约束

```python
CONSTRAINTS = [
    {"rule": "max_length", "params": {"limit": 100}, "check": lambda r: len(r) < 100},
    {"rule": "no_pii", "check": lambda r: not any(p in r for p in ["@", "http://"])},
    {"rule": "includes_citation", "check": lambda r: "[" in r and "]" in r},
]

def verify_constraints(response, constraints):
    """验证���答是否满足所有约束。"""
    violations = []
    for c in constraints:
        if not c["check"](response):
            violations.append(c["rule"])
    return violations
```

### 2.3 路由散文 vs 操作规则

| 类型 | 用途 | 示例 |
|------|------|------|
| **路由散文** | 指导行为风格 | "保持专业语气" |
| **操作规则** | 可验证的约束 | "必须包含代码示例" |

---

## 3. 从零实现

### Step 1：可执行约束引擎

```python
class ConstraintEngine:
    """可执行约束引擎。"""
    def __init__(self):
        self.constraints = []

    def add(self, name, check_fn, error_message):
        self.constraints.append({"name": name, "check": check_fn, "message": error_message})

    def verify(self, response):
        """验证回答是否满足所有约束。"""
        failures = []
        for c in self.constraints:
            if not c["check"](response):
                failures.append(c["message"])
        return failures

    def verify_and_fix(self, response, model_fn, max_attempts=3):
        """验证+修复循环。"""
        for _ in range(max_attempts):
            failures = self.verify(response)
            if not failures:
                return response
            response = model_fn(f"修复以下问题: {'; '.join(failures)}\n原始:{response}")
        return response


# 常用约束
def max_length(n):
    return lambda r: len(r) <= n, f"回答超过 {n} 字"

def no_pii(response):
    return not any(p in response for p in ["@", "http://", "1[3-9]\\d{9}"])
```

---

## 4. 工具

### 4.1 约束检查

| 方法 | 示例 |
|------|------|
| 正则 | 邮箱、电话、URL |
| 格式 | JSON Schema |
| 语义 | 情绪分析 |

---

## 5. 工程最佳实践

- **约束>建议**：可执行的约束比散文更可靠
- **自动验证**：运行时自动检查约束
- **自修复**：不满足时自动重试

---

## 6. 常见错误

### 错误 1：约束太松

**现象：** "尽量……"——不可检查。

**修复：** "至少……"或"不超过……"——可检查。

---

## 7. 面试考点

### Q1：为什么约束比散文更可靠？（难度：⭐⭐）

**参考答案：**
散文是愿望——"尽量简洁"不可检查。约束是测试——"不超过 100 字"可以自动验证。每个约束在运行时自动检查，违反时自动修复或标记。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 可执行约束 | "可验证的规则" | 可以自动检查的条件——超越散文愿望 |
| 路由散文 | "行为指导" | 建议性指令——风格、语气、态度 |
| 操作规则 | "必须遵守" | 可验证的约束——违反时需要修复 |

---

## 📚 小结

散文是愿望，约束是测试。每个指令都应该是一个可自动验证的条件。运行时检查约束，不满足时自动修复。

---

## ✏️ 练习

1. **【实现】** 为一个回复生成系统设计 5 个可执行约束
2. **【设计】** 添加自修复逻辑——违反约束时自动重写

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 约束引擎 | `code/main.py` | 可执行约束 + 自动验证 |

---

## 📖 参考资料

1. [博客] Anthropic. Building Effective Agents. 2024.
2. [论文] Microsoft. AI Agent Failure Modes. 2025.
