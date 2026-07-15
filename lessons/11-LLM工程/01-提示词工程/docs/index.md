# 提示词工程：技巧与模式

> 大多数人写提示词像发短信——然后奇怪为什么一个 2000 亿参数的模型给出平庸的回复。提示词工程不是小技巧。是理解你发送的每个词都是指令——并且模型字面遵循指令。写更好的指令，得到更好的输出。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-05（从零构建 LLM）| **时间：** ~90 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 使用核心提示词模式（角色、上下文、约束、输出格式）将模糊请求转换为精确指令
- [ ] 构建带有显式行为规则的系统提示词，产生一致的高质量输出
- [ ] 诊断提示词失败（幻觉、拒绝、格式违规）并用有针对性的修改修复
- [ ] 实现提示词测试框架——对照一组期望输出评估提示词变化

---

## 1. 问题

你打开 ChatGPT。输入："帮我写一封营销邮件。" 得到的是通用、臃肿、不可用的东西。再试一次——好一点但还是不对。你花了 20 分钟重写同一个请求。这不是模型的问题。是指令的问题。

同一个任务，两种写法：

**模糊提示词：**
```
为我们的新产品写一封营销邮件。
```

**工程化提示词：**
```
你是一家 B2B SaaS 公司的高级文案。为 DevFlow（一个 CI/CD 管道调试器）写一封产品发布邮件。
目标受众：A 轮创业公司的工程经理。语气：自信、技术性、不推销。长度：150 字。
包含一个具体指标（3.2 倍更快的管道调试）。以指向演示页面的单个 CTA 结束。
只输出邮件正文，不要主题行建议。
```

第一个提示词激活了训练数据中营销邮件的一般分布。第二个激活了一个窄的、高质量的子集。同一个模型、同一组参数——输出天差地别。

提示词工程不是 hack 或变通方案。它是人类意图和机器能力之间的主要接口。

---

## 2. 概念

### 2.1 提示词结构

每个 LLM API 调用有三个组成部分：

```
系统消息（System Message）：设定身份、规则、约束。跨对话轮次持续。
用户消息（User Message）：当前请求。每次轮次变化。
对话历史（Conversation History）：之前的轮次。提供上下文和连贯性。
```

### 2.2 系统提示词模式

好的系统提示词包含五个关键部分：

| 组成部分 | 做什么 | 示例 |
|---------|--------|------|
| **角色** | 设定人格和行为约束 | "你是一位资深软件工程师" |
| **上下文** | 提供必要背景 | "你正在审查一个 Python 代码库" |
| **约束** | 限制行为范围 | "只提供可运行的代码，不要解释" |
| **输出格式** | 指定结构化输出 | "用 JSON 格式回答，字段：summary, code, tests" |
| **例子** | 展示而不是告诉 | "比如：用户问 X，你应该回答 Y" |

### 2.3 用户提示词模式

**Chain-of-Thought (思维链):** "让我们一步步思考" → 模型生成中间推理步骤
**角色提示:** "从专家的角度回答这个问题" → 激活特定领域知识
**约束提示:** "在 50 字以内回答，使用简单语言" → 限制输出空间
**格式提示:** "以 JSON 格式输出，字段为..." → 结构化输出

### 2.4 常见的提示词失败模式

| 失败模式 | 症状 | 修复 |
|---------|------|------|
| **幻觉** | 输出事实错误 | 添加"如果你不确定，说不知道" |
| **格式违规** | 输出不符合指定格式 | 提供明确的模板或示例 |
| **拒绝** | 模型拒绝回答 | 添加安全前导语句或重构问题 |
| **长度偏差** | 输出太长/太短 | 明确指定长度（字/段/句） |
| **重复** | 输出重复内容 | 降低温度，或添加多样化提示 |

---

## 3. 从零实现

### Step 1：提示词工程化模板

```python
def build_system_prompt(role, context, constraints, output_format, examples=None):
    """
    构建结构化的系统提示词。
    Args:
        role: 模型扮演的角色
        context: 必要背景信息
        constraints: 行为约束列表
        output_format: 输出格式描述
        examples: 可选的示例列表
    """
    parts = [
        f"## 角色\n{role}",
        f"## 上下文\n{context}",
        f"## 约束\n" + "\n".join(f"- {c}" for c in constraints),
        f"## 输出格式\n{output_format}",
    ]
    if examples:
        parts.append(f"## 示例\n" + "\n".join(f"### 示例 {i+1}\n{e}" for i, e in enumerate(examples)))
    return "\n\n".join(parts)
```

### Step 2：提示词测试框架

```python
def evaluate_prompt(prompt_fn, test_cases):
    """
    评估提示词效果。
    Args:
        prompt_fn: 返回提示词的函数 (str -> str)
        test_cases: [(输入, 期望输出模式), ...]
    Returns:
        通过率
    """
    passed = 0
    for input_text, expected_pattern in test_cases:
        response = prompt_fn(input_text)
        if expected_pattern in response:
            passed += 1
    return passed / max(len(test_cases), 1)
```

### Step 3：设置参数

```python
import openai  # 或其他客户端

def generate_with_params(prompt, temperature=0.7, max_tokens=512, 
                         top_p=0.95, frequency_penalty=0.0):
    """控制生成参数。"""
    # temperature: 0=确定, >1=创意
    # top_p: nucleus sampling 阈值
    # frequency_penalty: 惩罚重复词
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
    )
    return response.choices[0].message.content
```

---

## 4. 工具

### 4.1 OpenAI / Anthropic API

```python
# OpenAI
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一位资深 Python 工程师。"},
        {"role": "user", "content": "审查以下代码并指出安全问题。"},
    ],
)

# Anthropic
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-5",
    system="你是一位资深 Python 工程师。",
    messages=[{"role": "user", "content": "审查以下代码并指出安全问题。"}],
    max_tokens=1024,
)
```

### 4.2 温度参数影响

| 温度 | 效果 | 适用场景 |
|------|------|---------|
| 0-0.2 | 确定性的、可重复的 | 分类、翻译 |
| 0.3-0.7 | 平衡 | 通用生成（默认） |
| 0.8-1.2 | 创意性的 | 头脑风暴、故事 |
| > 1.5 | 非常随机 | 创意探索（可能无意义） |

---

## 6. 工程最佳实践

### 6.1 提示词迭代工作流

```
1. 写初始提示词
2. 测试 5-10 个输入
3. 修复失败模式
4. 重新测试
5. 锁定生产版本
6. 定期回归测试
```

### 6.2 中文场景特别建议

- 中文提示词需要用精确的指令句："请用 JSON 格式输出"而非"输出为 JSON"
- 中文系统提示词中的角色设定要具体："你是一位北京大学计算机科学教授"比"你是一位 AI 助手"效果更好

### 6.3 踩坑经验

- **温度太高**：生成不稳定，相同输入不同输出 → 生产环境用 0-0.2
- **提示词太长**：关键指令被稀释 → 最重要的约束放前面
- **没有测试**：一次写好的提示词几乎不存在 → 迭代测试

---

## 7. 常见错误

### 错误 1：模糊请求

```python
# ❌ 模糊
prompt = "写一段代码"
# ✓ 精确
prompt = "用 Python 写一个函数，输入是一个整数列表，返回中位数。包含类型标注和 docstring。"
```

### 错误 2：认为模型可以"猜"意图

**现象：** 模型输出偏离用户意图。

**修复：** 明确指定角色、上下文、约束、输出格式。

---

## 8. 面试考点

### Q1：提示词工程的本质是什么？（难度：⭐⭐）

**参考答案：**
提示词工程不是"让模型做正确的事的技巧集"——它是"在模型的训练分布中找到高质量子集的接口"。模糊提示词激活了训练数据中所有相关数据的分布。好的提示词通过提供角色、上下文、约束和格式，将这个分布缩小到任务所需的高质量区域。

### Q2：为什么系统消息和用户消息要分开？（难度：⭐⭐）

**参考答案：**
系统消息设定持久的行为规则——角色、约束、格式——这些在对话中应该一致。用户消息是每次变化的请求。分开可以：(1) 节省 token——系统消息只需要发送一次（如果 API 支持）；(2) 明确优先级——系统消息优先于用户消息；(3) 方便维护——行为规则集中在一个地方，不需要每个请求重复。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 系统提示词 | "给模型的指令" | 设定行为规则、角色和约束的持久消息 |
| 温度 (Temperature) | "创造力的程度" | 控制生成随机性的参数——0=确定，>1=创意 |
| 思维链 (CoT) | "一步步思考" | 让模型在推理过程中生成中间步骤——提升推理准确率 |
| 少样本提示 | "给例子" | 在提示词中提供输入-输出示例——上下文内学习 |

---

## 📚 小结

提示词工程是将人类意图翻译为机器指令的实践。结构化的提示词（角色+上下文+约束+格式）比模糊请求效果好得多。温度控制生成随机性。系统消息和用户消息分离提示工程效果。提示词需要迭代测试和优化——一次写好的提示词几乎不存在。

---

## ✏️ 练习

1. **【实现】** 构建一个提示词测试框架：对 5 个测试用例运行一个提示词，计算通过率。优化提示词直到 100% 通过。
2. **【实验】** 对比温度 0.0 和温度 1.0 的生成差异——在 10 个相同输入上观察多样性。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 提示词测试框架 | `code/main.py` | 评估和优化提示词的测试工具 |

---

## 📖 参考资料

1. [论文] Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". NeurIPS, 2022.
2. [API] OpenAI Prompt Engineering Guide: https://platform.openai.com/docs/guides/prompt-engineering
3. [API] Anthropic Prompt Engineering: https://docs.anthropic.com/en/docs/prompt-engineering

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
