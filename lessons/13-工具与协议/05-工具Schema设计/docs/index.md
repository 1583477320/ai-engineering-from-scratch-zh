# 工具 Schema 设计——命名、描述、参数约束

> 一个正确的工具在模型无法判断何时使用时会静默失败。命名、描述和参数形状在 StableToolBench 和 MCPToolBench++ 等基准上驱动 10-20 个百分点的工具选择准确率差异。本课命名将一个工具从"模型偶尔选对"变为"模型可靠选对"的设计规则。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 13 · 01（工具接口）、04（结构化输出）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 使用"当 X 时使用此工具，不要用于 Y"模式编写工具描述——控制在 1024 字符以内
- [ ] 设计参数命名约定——动词+名词格式，消除歧义
- [ ] 为每个参数选择正确的约束类型——enum、default、minimum、pattern
- [ ] 实现工具 Schema linter——自动检查设计质量

---

## 1. 问题

你的工具定义正确——JSON Schema 有效，执行函数能跑。但 LLM 有时选错了工具，有时传了错误参数。问题不在执行端，而在**描述端**——模型需要理解"什么时候用这个工具"和"参数应该是什么"。

好的工具描述可以让工具选择准确率从 80% 提升到 95%。

---

## 2. 概念

### 2.1 工具描述三要素

| 要素 | 作用 | 好的示例 |
|------|------|---------|
| **名称** | 唯一标识 | `get_weather`（动词+名词） |
| **描述** | 说明使用场景 | "当用户询问当前天气时使用。返回温度和条件。" |
| **参数 schema** | 约束输入格式 | `{city: {type: "string", description: "城市名称"}}` |

### 2.2 描述模式

```
当 [条件] 时使用此工具。
不要用于 [不适用场景]。
返回 [输出格式]。
```

### 2.3 命名约定

| 模式 | 示例 | 说明 |
|------|------|------|
| 动词+名词 | `get_weather`, `search_docs` | 推荐 |
| 动作+对象 | `fetch_data`, `send_email` | 可接受 |
| 模糊名 | `helper`, `utils` | ❌ 避免 |

### 2.4 参数设计

| 原则 | 示例 |
|------|------|
| 必填参数用 required | `"required": ["city"]` |
| 枚举限制范围 | `"enum": ["celsius", "fahrenheit"]` |
| 数值约束 | `"minimum": 1, "maximum": 100` |
| 默认值 | `"default": "celsius"` |
| 描述每个参数 | `"description": "城市名称（中文或英文）"` |

---

## 3. 从零实现

### Step 1：工具 Schema Linter

```python
def lint_tool_schema(name, description, schema):
    """检查工具 Schema 设计质量。"""
    issues = []

    # 检查命名
    if not name.replace("_", "").isalpha():
        issues.append(f"工具名 '{name}' 包含非字母字符")

    # 检查描述长度
    if len(description) > 1024:
        issues.append(f"描述过长 ({len(description)} > 1024)")

    # 检查必填字段
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for field in required:
        if field not in properties:
            issues.append(f"必填字段 '{field}' 未在 properties 中定义")

    # 检查字段描述
    for field, field_schema in properties.items():
        if "description" not in field_schema:
            issues.append(f"字段 '{field}' 缺少 description")

    return issues
```

### Step 2：工具描述模板

```python
def generate_tool_description(name, purpose, restrictions, output):
    """生成结构化的工具描述。"""
    return f"""当{purpose}时使用此工具。
不要用于{restrictions}。
返回{output}。"""
```

---

## 4. 工具

### 4.1 MCPToolBench

```bash
# 测试工具选择准确率
mcp_tool_bench evaluate --tools my_tools.json --queries test_queries.json
```

### 4.2 StableToolBench

```bash
# 稳定工具基准测试
stable_tool_bench test --model gpt-4o --tools my_tools.json
```

---

## 5. 工程最佳实践

### 5.1 工具描述最佳实践

- **动词+名词命名**：`get_weather` > `weather` > `helper`
- **"当 X 时使用"模式**：明确使用场景
- **"不要用于 Y"模式**：排除不适用场景
- **参数描述**：每个参数都有明确的 description

### 5.2 踩坑经验

- **描述太短**：模型无法区分相似工具
- **描述太长**：关键信息被稀释
- **参数名歧义**：`text` vs `content` vs `input`——统一命名

---

## 7. 常见错误

### 错误 1：工具描述没有排除场景

**现象：** 模型在不适用时调用了工具。

**修复：** 添加"不要用于..."——帮助模型区分。

### 错误 2：参数缺少枚举约束

**现象：** 模型传了不在范围内的值。

**修复：** 为所有离散选项添加 `enum`。

---

## 8. 面试考点

### Q1：如何提高工具选择的准确率？（难度：⭐⭐）

**参考答案：**
(1) **名称明确**：动词+名词格式（如 `get_weather`）；(2) **描述包含排除条件**："当用户询问当前天气时使用，不要用于天气预报"；(3) **参数约束完整**：enum、required、description 都要设置；(4) **工具数量控制**：>20 个工具时分组或动态加载。

### Q2：为什么参数的 description 很重要？（难度：⭐⭐⭐）

**参考答案：**
LLM 根据参数的 description 来理解"这个参数应该填什么"。没有 description 时，模型只能根据参数名猜测——容易出错。例如 `query` 和 `text` 看起来类似，但一个用于搜索、一个用于内容提取。description 帮助模型区分这些微妙差异，减少 10-20% 的参数错误。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 工具 Schema | "工具说明书" | 用 JSON Schema 定义工具的输入格式、参数约束、使用场景 |
| 工具选择准确率 | "模型选对工具的概率" | LLM 从可用工具中选出正确工具的概率 |
| 约束约束 | "参数限制" | enum（枚举）、required（必填）、minimum/maximum（范围）等 |

---

## 📚 小结

好的工具设计 = 精确命名 + "当 X 时使用"的描述 + 完整的参数约束。动词+名词命名、排除条件、参数描述是关键。工具 Schema Linter 可以自动检查设计质量。在基准测试中，好的设计可以提升 10-20% 的工具选择准确率。

---

## ✏️ 练习

1. **【实现】** 构建工具 Schema Linter——自动检查命名、描述长度、参数完整性
2. **【实验】** 对比有/无排除条件的工具描述——测量工具选择准确率差异

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| Schema Linter | `code/main.py` | 工具 Schema 自动检查工具 |

---

## 📖 参考资料

1. [论文] StableToolBench: https://github.com/thu-coai/StableToolBench
2. [文档] JSON Schema: https://json-schema.org/draft/2020-12/json-schema-validation
3. [文档] MCP Schema: https://spec.modelcontextprotocol.io

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
