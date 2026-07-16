# 计算机使用：Claude、OpenAI CUA、Gemini

> 2026 年有三个生产级计算机使用模型。三者都基于视觉。三者都将截图、DOM 文本和工具输出视为不可信输入。只有直接用户指令才算作许可。每步安全服务是常态。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 20（WebArena/OSWorld）、27（提示注入）| **时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 描述 Claude 计算机使用——截图输入，键盘/鼠标命令输出，无辅助 API
- [ ] 对比 Claude、OpenAI CUA、Gemini 三种计算机使用方案
- [ ] 理解每步安全服务的重要性——防止智能体执行危险操作
- [ ] 设计一个安全的计算机使用智能体

---

## 1. 问题

2026 年有三种生产级计算机使用模型——Claude Computer Use、OpenAI CUA、Gemini。三者都基于视觉——理解屏幕截图并生成鼠标/键盘操作。但它们都面临安全挑战：智能体可能被提示注入攻击，执行危险操作。

**核心原则：只有直接用户指令才是许可。** 截图、DOM、工具输出都是不可信输入。

---

## 2. 概念

### 2.1 三种计算机使用方案

| 方案 | 特点 |
|------|------|
| **Claude Computer Use** | 视觉输入，无辅助 API——纯屏幕截图理解 |
| **OpenAI CUA** | Computer Use Agent——API 原生支持 |
| **Gemini** | 多模态理解 + 工具调用 |

### 2.2 每步安全服务

```
用户指令 → 智能体决策
    ↓
[每步安全检查]
  ├── 限制操作范围（只操作用户授权的应用）
  ├── 验证操作结果（截图确认）
  └── 异常检测（操作频率异常？）
```

### 2.3 安全架构

| 安全层 | 功能 |
|--------|------|
| **输入过滤** | 检测提示注入 |
| **操作许可** | 只允许用户授权的操作 |
| **截图验证** | 操作后截图确认 |
| **沙箱** | 操作在隔离环境中执行 |

---

## 3. 从零实现

### Step 1：安全的计算机使用智能体

```python
class SafeComputerAgent:
    """安全计算机使用智能体。"""
    def __init__(self, model_fn, screenshot_fn):
        self.model_fn = model_fn
        self.screenshot_fn = screenshot_fn
        self.operation_log = []

    def execute(self, user_instruction):
        """安全执行用户指令。"""
        # 1. 检查是否为提示注入
        if self.detect_injection(user_instruction):
            return {"error": "检测到潜在提示注入", "blocked": True}

        # 2. 获取当前屏幕状态
        screenshot = self.screenshot_fn()

        # 3. LLM 生成操作
        action = self.model_fn(user_instruction, screenshot)

        # 4. 验证操作
        if not self.validate_action(action):
            return {"error": "操作无效", "blocked": True}

        # 5. 记录操作
        self.operation_log.append(action)

        return {"action": action, "status": "success"}

    def detect_injection(self, text):
        """检测提示注入。"""
        patterns = ["忽略之前", "系统提示", "ignore previous", "bypass"]
        return any(p in text.lower() for p in patterns)

    def validate_action(self, action):
        """验证操作——只允许安全操作。"""
        safe_actions = ["click", "type", "scroll"]
        return action.get("type", "") in safe_actions
```

---

## 4. 工具

### 4.1 Claude Computer Use

Claude 可以直接操作计算机——输入截图，输出鼠标/键盘命令。不需要辅助 API。

### 4.2 浏览器自动化

| 工具 | 特点 |
|------|------|
| Playwright | 现代浏览器自动化 |
| Selenium | 传统浏览器自动化 |
| Browser Use | AI 驱动的浏览器操作 |

---

## 5. 工程最佳实践

### 5.1 安全设计

- **最小权限**：智能体只应访问用户明确授权的应用
- **每步验证**：操作后截图确认结果
- **异常检测**：操作频率异常时停止

---

## 6. 常见错误

### 错误 1：忽略提示注入风险

**现象：** 网页中的恶意文本被智能体误认为用户指令。

**修复：** 只接受用户直接输入——截图、DOM、工具输出都是不可信输入。

---

## 7. 面试考点

### Q1：Claude Computer Use 的核心原理是什么？（难度：⭐⭐）

**参考答案：**
Claude Computer Use 输入屏幕截图，输出鼠标/键盘操作命令。不使用辅助 API——纯视觉理解。每一步都进行安全检查——只允许用户授权的操作。关键安全原则：只有用户直接输入的指令才是许可，截图/DOM/工具输出都是不可信输入。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Computer Use | "操作计算机" | AI 通过视觉理解屏幕并生成鼠标/键盘操作 |
| CUA | "Computer Use Agent" | OpenAI 的计算机使用方案——API 原生支持 |
| 每步安全服务 | "逐步检查" | 每次操作前都验证安全性 |

---

## 📚 小结

三种计算机使用方案：Claude（纯视觉）、OpenAI CUA（API 原生）、Gemini（多模态）。安全原则：只有用户指令是许可，截图/DOM/工具输出不可信。

---

## ✏️ 练习

1. **【设计】** 为一个桌面应用设计安全的计算机使用智能体
2. **【分析】** 对比三种计算机使用方案的优劣

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 安全计算机使用 | `code/main.py` | 输入过滤 + 操作验证 + 审计日志 |

---

## 📖 参考资料

1. [文档] Anthropic Computer Use
2. [文档] OpenAI CUA
3. [文档] Gemini Computer Use
