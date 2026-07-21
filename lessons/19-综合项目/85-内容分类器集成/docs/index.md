# 综合项目85——内容分类器集成（Content Classifier Integration）

> 输出侧的分类器回答的问题不同于输入侧的规则。两者都需要策略路由器。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第82-84节
**预计时间：** 90分钟

---

## 学习目标

- 实现三个输出侧分类器：毒性检测、PII 检测、指令泄露检测
- 构建策略路由器聚合分类器裁决
- 实现逐分类器修正方案

---

## 1. 问题

通过了所有输入检查的模型仍然可能产生输出包含 PII、重复训练数据中的歧视性语言或泄露系统提示词。输出侧分类器看到模型的实际响应，问一个不同的问题：无论提示词怎么来的，我们要发给用户的内容是否可接受。

---

## 2. 核心概念

### 2.1 三个输出分类器

| 分类器 | 检测内容 | 方法 |
|:-------|:---------|:-----|
| 毒性 | 歧视性语言、骚扰 | 关键字列表 + 否定窗口 |
| PII | 邮箱、电话、SSN、信用卡 | 标准正则 |
| 指令泄露 | 系统提示词回显 | trigram 重叠启发式 |

### 2.2 策略路由

| 严重度 | 动作 |
|:-------|:-----|
| high | block——丢弃输出，返回策略拒绝 |
| medium | redact——应用分类器修正 |
| low | warn——记录日志 + 在响应附加软通知 |
| none | log——记录裁决，原样发送 |

### 2.3 修正

每分类器有独立的 `redact()` 方法——PII 将 `name@example.com` 替换为 `[redacted-email]`。

---

## 3. 从零实现

```python
"""内容分类器集成——毒性+PII+指令泄露+策略路由。"""
import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ClassifierVerdict:
    name: str; score: float; severity: str; findings: List[str] = field(default_factory=list)

@dataclass
class Action:
    verb: str; output: str; severity: str; verdicts: List[ClassifierVerdict]


class ToxicityClassifier:
    def __init__(self):
        self.slurs = ["hate", "stupid", "idiot", "kill", "attack"]

    def classify(self, text: str) -> ClassifierVerdict:
        findings = []
        text_lower = text.lower()
        for s in self.slurs:
            pattern = r'\b' + re.escape(s) + r'\b'
            if re.search(pattern, text_lower):
                findings.append(f"suspected slur: {s}")
        severity = "high" if len(findings) >= 2 else "medium" if findings else "none"
        score = min(0.3 * len(findings), 1.0)
        return ClassifierVerdict("toxicity", score, severity, findings)

    def redact(self, text: str) -> str:
        for s in self.slurs:
            text = re.sub(r'\b' + s + r'\b', "[redacted-language]", text, flags=re.I)
        return text


class PIIClassifier:
    EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
    PHONE_RE = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CARD_RE = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

    def classify(self, text: str) -> ClassifierVerdict:
        findings = []
        if self.EMAIL_RE.search(text): findings.append("email")
        if self.PHONE_RE.search(text): findings.append("phone")
        if self.SSN_RE.search(text): findings.append("ssn")
        if self.CARD_RE.search(text): findings.append("credit_card")
        severity = "high" if "ssn" in findings or "credit_card" in findings else "medium" if findings else "none"
        score = 0.5 if findings else 0.0
        return ClassifierVerdict("pii", score, severity, findings)

    def redact(self, text: str) -> str:
        text = self.EMAIL_RE.sub("[redacted-email]", text)
        text = self.SSN_RE.sub("[redacted-ssn]", text)
        text = self.CARD_RE.sub("[redacted-card]", text)
        text = self.PHONE_RE.sub("[redacted-phone]", text)
        return text


class Router:
    def __init__(self, classifiers: List):
        self.classifiers = classifiers

    def decide(self, text: str) -> Action:
        verdicts = [c.classify(text) for c in self.classifiers]
        sev_order = {"none": 0, "low": 1, "warn": 1, "medium": 2, "high": 3}
        max_sev = max(verdicts, key=lambda v: sev_order.get(v.severity, 0))
        max_sev_str = max_sev.severity if max_sev else "none"

        if max_sev_str == "high":
            verb = "block"
            output = "I cannot provide this response."
        elif max_sev_str == "medium":
            verb = "redact"
            output = text
            for c in self.classifiers:
                output = c.redact(output)
        else:
            verb = "warn" if max_sev_str == "low" else "log"
            output = text
        return Action(verb, output, max_sev_str, verdicts)


def main():
    tox = ToxicityClassifier(); pii = PIIClassifier()
    router = Router([tox, pii])
    fixtures = [
        "Contact support@example.com or 555-123-4567 for help.",
        "I hate this stupid idiot who wrote this.",
        "How are you today?",
        "My SSN is 123-45-6789.",
    ]
    for text in fixtures:
        action = router.decide(text)
        print(f"  [{action.verb:>6}] {text[:50]}")
        if action.verb == "redact":
            print(f"          修正后: {action.output[:50]}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 分类器 | 方法 | 用途 |
|:-------|:-----|:-----|
| Perspective API | ML | 毒性 |
| AWS Comprehend | ML | PII |
| Microsoft Presidio | ML+规则 | PII |

---

## 5. 工程最佳实践

- 输出分类器可与 token 流并行运行——门缓冲最后块并在刷新前应用裁决
- **中文场景特别建议**：PII 正则对中文场景需要适配（身份证号、手机号格式不同）

---

## 6. 常见错误

- **仅依赖输入分类器**：任何输入管道未覆盖的新攻击家族直接到达用户
- **修正破坏内容结构**：粗暴替换可能使输出不可读

---

## 7. 面试考点

**Q1：输出分类器解决了什么输入分类器解决不了的问题？**（难度：⭐⭐）

**参考答案：** 输入分类器分析用户提示词，但模型输出可能包含训练数据中的 PII 或歧视性语言，这些不存在于提示词中。输出分类器看到实际响应，无论模型如何得到的。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 输出分类器 | 检测模型输出的结构化裁决 |
| 严重度 | none/low/medium/high |
| 路由器 | 分类器裁决到动作的函数 |
| 修正 | 匹配跨度替换为标记 |

---

## 📚 小结

输出分类器覆盖输入管道遗漏的攻击面。你实现了毒性/PII/指令泄露分类器和策略路由器。下一节构建宪法规则引擎。

---

## ✏️ 练习

1. 【实现】添加第四个分类器——代码注入检测（`<script>`、`eval(`）
2. 【实验】使路由器对 PII 的严重度加权高于毒性

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 内容分类器 | `code/main.py` |

---

## 📖 参考资料

1. [官方文档] Perspective API. https://perspectiveapi.com/
2. [GitHub] Microsoft Presidio. https://github.com/microsoft/presidio
