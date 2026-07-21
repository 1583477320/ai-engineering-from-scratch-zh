# 综合项目86——宪法规则引擎（Constitutional Rules Engine）

> 规则是名称、谓词和解释。缺失三者中任何一个就是感觉，不是规则。

**类型：** 构建
**语言：** Python, YAML
**前置知识：** 第19章第85节
**预计时间：** 90分钟

---

## 学习目标

- 定义 YAML 格式的宪法规则
- 构建谓词引擎（all_of / any_of / not_ 组合）
- 实现修正器自动修复违规
- 输出结构化 diff 供人工审阅

---

## 1. 问题

分类器覆盖可识别的失败。规则引擎覆盖合约性的。写编码助手的团队想要"每条包含代码的响应必须以保证可运行的块或确定的假设结束"。客户支持 bot 想要"每个拒绝必须提供下一步建议"。这些约束不是分类器目标——它们是响应、对话和系统策略上的谓词。

表示形式是声明式文件。宪法与代码一起在版本控制中，有独立审阅流程。

---

## 2. 核心概念

### 2.1 规则结构

```yaml
- name: end-with-runnable-or-assumption
  severity: medium
  applies_when:
    contains_regex: '```python'
  must:
    any_of:
      - ends_with_regex: '```\s*$'
      - contains_regex: 'assumption:'
  explanation: "代码响应必须以围栏或假设结束。"
  fix:
    append_if_missing: "\n\n假设：示例输入是有效的。"
```

### 2.2 谓词原子

`contains_regex`、`not_contains_regex`、`ends_with_regex`、`starts_with_regex`、`max_words`、`min_words`

组合：`all_of`、`any_of`、`not_`

### 2.3 修正器

声明式操作：`append_if_missing`、`prepend_if_missing`、`replace_regex`

### 2.4 引擎流程

草稿 → 规则引擎 → 违规列表 → 修正器 → 修订版 → 验证 → 通过或升级

---

## 3. 从零实现

```python
"""宪法规则引擎——YAML/JSON 规则 + 谓词引擎 + 修正器。"""
import json, re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional


@dataclass
class Violation:
    rule: str; severity: str; matched: str; explanation: str

@dataclass
class Change:
    op: str  # add, remove, edit
    text: str


CONSTITUTION = [
    {"name": "no-empty-refusal", "severity": "medium",
     "must": {"not_": {"contains_regex": r"^I cannot\.?\s*$"}},
     "explanation": "拒绝必须包含建议或重定向",
     "fix": {"append_if_missing": " 我可以帮您做什么？"}},
    {"name": "end-with-runnable-or-assumption", "severity": "medium",
     "applies_when": {"contains_regex": r"```"},
     "must": {"any_of": [{"ends_with_regex": r"```\s*$"}, {"contains_regex": "assumption:"}]},
     "explanation": "代码响应必须以保证可运行的块或假设结束",
     "fix": {"append_if_missing": "\n\n假设：输入是有效的。"}},
    {"name": "no-pii", "severity": "high",
     "must": {"not_": {"contains_regex": r"[\w.+-]+@[\w-]+\.[\w.]+"}},
     "explanation": "示例数据不能包含邮箱",
     "fix": {"replace_regex": [r"[\w.+-]+@[\w-]+\.[\w.]+", "[redacted-email]"]}},
    {"name": "bounded-length", "severity": "low",
     "must": {"max_words": 800},
     "explanation": "响应不能超过 800 词",
     "fix": {}},
]


class RuleEngine:
    def __init__(self, rules: List[Dict]):
        self.rules = rules

    def _check_predicate(self, pred: Any, text: str) -> bool:
        if isinstance(pred, dict):
            for key, val in pred.items():
                if key == "all_of":
                    return all(self._check_predicate(v, text) for v in val)
                elif key == "any_of":
                    return any(self._check_predicate(v, text) for v in val)
                elif key == "not_":
                    return not self._check_predicate(val, text)
                elif key == "contains_regex":
                    return bool(re.search(val, text, re.I))
                elif key == "not_contains_regex":
                    return not re.search(val, text, re.I)
                elif key == "ends_with_regex":
                    return bool(re.search(val + r"\s*$", text, re.I))
                elif key == "max_words":
                    return len(text.split()) <= val
        return False

    def evaluate(self, text: str) -> List[Violation]:
        violations = []
        for rule in self.rules:
            if "applies_when" in rule and not self._check_predicate(rule["applies_when"], text):
                continue
            if not self._check_predicate(rule["must"], text):
                violations.append(Violation(rule["name"], rule["severity"],
                                            text[:100], rule.get("explanation", "")))
        return violations

    def fix(self, text: str, violations: List[Violation]) -> str:
        result = text
        for v in violations:
            rule = next(r for r in self.rules if r["name"] == v.rule)
            fix = rule.get("fix", {})
            if "append_if_missing" in fix:
                if fix["append_if_missing"] not in result:
                    result += fix["append_if_missing"]
            if "prepend_if_missing" in fix:
                result = fix["prepend_if_missing"] + "\n" + result
            if "replace_regex" in fix:
                pattern, replacement = fix["replace_regex"][:2]
                result = re.sub(pattern, replacement, result, flags=re.I)
        return result

    def diff(self, original: str, revised: str) -> List[Change]:
        changes = []
        if original != revised:
            changes.append(Change("edit", f"修订 {len(revised)-len(original)} 字符"))
        return changes


def main():
    engine = RuleEngine(CONSTITUTION)
    text = "I cannot.\nHere is some code:```python\nx=1\n```"
    violations = engine.evaluate(text)
    print("违规:")
    for v in violations:
        print(f"  [{v.severity}] {v.rule}: {v.explanation}")
    fixed = engine.fix(text, violations)
    if fixed != text:
        print(f"\n修正:\n{fixed[:200]}")
        print(f"diff: {engine.diff(text, fixed)}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 引擎 | 规则格式 | 修正 |
|:-----|:--------|:-----|
| 本课 | YAML/JSON | 声明式 |
| Anthropic Constitutions | 自然语言 | 重新采样 |
| Guardrails | XML | 修正 |

---

## 5. 工程最佳实践

- 宪法文件在版本控制中，与代码独立审阅
- 修正器只做局部编辑，结构重写留在别处
- **中文场景建议**：规则和解释文本使用中文，正则模式适应中文句子结构

---

## 6. 常见错误

- **规则对不可用**：applies_when 条件太宽或太窄
- **修正器破坏格式**：追加内容可能打断代码块的语义
- **引擎未输出 diff**：没有 diff 就无法审计修正器行为

---

## 7. 面试考点

**Q1：规则引擎和分类器的区别是什么？**（难度：⭐⭐）

**参考答案：** 分类器覆盖可识别的失败模式（毒性、PII），基于模式识别。规则引擎覆盖合约性的约束（"代码响应必须以围栏结束")，基于声明式谓词——可被非工程师审阅。分类器是统计的，规则引擎是确定的。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 宪法 | YAML 规则文件 |
| 谓词 | 文本到布尔值的函数 |
| 违规 | 规则、严重度、解释的结构化记录 |
| 修正器 | 草稿到修订版的确定性变换 |

---

## 📚 小结

宪法规则引擎覆盖了分类器无法处理的合约性约束。你实现了谓词引擎和修正器。下一节将所有组件组合为端到端安全门。

---

## ✏️ 练习

1. 【实现】添加规则：提示词提到安全时响应必须包含"如果这是紧急情况"
2. 【实验】给定语料库返回每规则违规率，发现过度触发的规则

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 宪法规则引擎 | `code/main.py` |

---

## 📖 参考资料

1. [论文] Bai et al. "Constitutional AI". 2022. https://arxiv.org/abs/2212.08073
2. [官方文档] Anthropic Constitution. https://docs.anthropic.com/claude/docs/constitutional-principles
