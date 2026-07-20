# 综合项目70——任务规范格式（Task Spec Format）

> 评估框架的质量取决于任务遵守的合约。在写任何评分函数之前，先冻结 JSONL 形状和指标词汇表。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第20-29节
**预计时间：** 90分钟

---

## 学习目标

- 定义覆盖算术、多项选择、代码执行、分类和摘要的 JSONL 任务记录规范
- 固定闭式指标名称词汇表
- 将 few-shot 示例和后处理规则指定为任务的一部分
- 实现严格验证器，拒绝格式错误的记录
- 发布包含所有分支的 10 条固定任务集

---

## 1. 问题

研究代码库积累评估脚本的速度快于积累测试。六个月后，每个 notebook 都有自己的 JSON 形状，每个指标被重新实现两次，跨运行无法比较。修复方法很无聊：选一个 schema，写一个验证器，拒绝其他一切。

规范借鉴了 BIG-bench、HELM 和 lm-eval 风格框架的思想，但字段名是自己的。每个字段只有一个所有者。

---

## 2. 核心概念

### 2.1 任务记录规范

```json
{
  "task_id": "arith_001",
  "category": "arithmetic",
  "prompt": "Compute: 17 + 24\nAnswer:",
  "targets": ["41"],
  "metric_name": "exact_match",
  "few_shot_examples": [{"prompt": "Compute: 2 + 2\nAnswer:", "completion": "4"}],
  "post_process": "strip_whitespace",
  "metadata": {"difficulty": "easy"}
}
```

### 2.2 闭式词汇表

**指标**：`exact_match`、`f1`、`bleu_4`、`rouge_l`、`accuracy`、`code_exec`
**后处理**：`none`、`strip_whitespace`、`lower`、`extract_letter`、`extract_code_block`、`extract_first_line`
**类别**：`arithmetic`、`mcq`、`code_exec`、`classification`、`summary`

### 2.3 验证规则

- `task_id` 无空格，唯一
- `category` 必须与 `metric_name` 匹配
- `targets` 非空列表
- `few_shot_examples` 最多 8 条
- 后处理规则不可组合

### 2.4 渲染

运行器在提示词前拼接 few-shot 示例。相同的代码路径适用于每个模型——唯一的方差是模型本身。

---

## 3. 从零实现

```python
"""任务规范格式——JSONL schema + 验证器。"""
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

VALID_METRICS = {"exact_match", "f1", "bleu_4", "rouge_l", "accuracy", "code_exec"}
VALID_CATEGORIES = {"arithmetic", "mcq", "code_exec", "classification", "summary"}
VALID_POST = {"none", "strip_whitespace", "lower", "extract_letter", "extract_code_block", "extract_first_line"}
CATEGORY_METRIC_MAP = {
    "arithmetic": "exact_match", "mcq": "exact_match",
    "code_exec": "code_exec", "classification": "accuracy", "summary": "rouge_l",
}

@dataclass
class TaskSpec:
    task_id: str; category: str; prompt: str; targets: List[str]
    metric_name: str; post_process: str
    few_shot_examples: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def render(self):
        parts = []
        for ex in self.few_shot_examples:
            parts.append(ex["prompt"] + " " + ex.get("completion", ""))
        parts.append(self.prompt)
        return "\n\n".join(parts)

    def post_process_output(self, output: str) -> str:
        if self.post_process == "strip_whitespace": return output.strip()
        if self.post_process == "lower": return output.lower()
        if self.post_process == "extract_letter":
            for c in output.upper():
                if c in "ABCDE": return c
            return output
        if self.post_process == "extract_code_block":
            import re
            m = re.search(r"```(?:python)?\n(.*?)\n```", output, re.DOTALL)
            return m.group(1) if m else output
        if self.post_process == "extract_first_line":
            for line in output.split("\n"):
                if line.strip(): return line.strip()
        return output

from dataclasses import dataclass, field

@dataclass
class ValidationError:
    line: int; field: str; rule: str; raw: str


def validate_task(record: dict, line_num: int, seen_ids: set) -> Tuple[Optional[TaskSpec], List[ValidationError]]:
    errors = []
    for req in ["task_id", "category", "prompt", "targets", "metric_name", "post_process"]:
        if req not in record:
            errors.append(ValidationError(line_num, req, "缺少必填字段", json.dumps(record)[:100]))
            return None, errors

    if not isinstance(record["task_id"], str) or " " in record["task_id"]:
        errors.append(ValidationError(line_num, "task_id", "必须是非空格字符串", record["task_id"]))
    if record["task_id"] in seen_ids:
        errors.append(ValidationError(line_num, "task_id", "重复", record["task_id"]))
    if record["category"] not in VALID_CATEGORIES:
        errors.append(ValidationError(line_num, "category", f"必须是 {VALID_CATEGORIES}", record["category"]))
    if record["metric_name"] not in VALID_METRICS:
        errors.append(ValidationError(line_num, "metric_name", f"必须是 {VALID_METRICS}", record["metric_name"]))
    if record["post_process"] not in VALID_POST:
        errors.append(ValidationError(line_num, "post_process", f"必须是 {VALID_POST}", record["post_process"]))
    if not isinstance(record["targets"], list) or not record["targets"]:
        errors.append(ValidationError(line_num, "targets", "必须是非空列表", str(record["targets"])))
    if not isinstance(record["prompt"], str) or not record["prompt"]:
        errors.append(ValidationError(line_num, "prompt", "必须是非空字符串", ""))

    expected = CATEGORY_METRIC_MAP.get(record.get("category"))
    if expected and record["metric_name"] != expected:
        errors.append(ValidationError(line_num, "metric_name", f"类别 {record['category']} 需要 {expected}", record["metric_name"]))

    if errors: return None, errors

    seen_ids.add(record["task_id"])
    return TaskSpec(
        task_id=record["task_id"], category=record["category"],
        prompt=record["prompt"], targets=record["targets"],
        metric_name=record["metric_name"], post_process=record["post_process"],
        few_shot_examples=record.get("few_shot_examples", []),
        metadata=record.get("metadata", {}),
    ), []


def validate_file(tasks_jsonl: str) -> Tuple[List[TaskSpec], List[ValidationError]]:
    all_specs, all_errors, seen = [], [], set()
    for i, line in enumerate(tasks_jsonl.strip().split("\n"), 1):
        if not line.strip(): continue
        try: record = json.loads(line)
        except json.JSONDecodeError as e:
            all_errors.append(ValidationError(i, "_json", f"JSON 解析错误: {e}", line[:100]))
            continue
        spec, errs = validate_task(record, i, seen)
        all_errors.extend(errs)
        if spec: all_specs.append(spec)
    return all_specs, all_errors


def render_with_fewshot(task: TaskSpec) -> str:
    return task.render()


FIXTURES = [
    json.dumps({"task_id":"arith_001","category":"arithmetic","prompt":"Compute: 2 + 2\nAnswer:","targets":["4"],"metric_name":"exact_match","post_process":"strip_whitespace"}),
    json.dumps({"task_id":"arith_002","category":"arithmetic","prompt":"Compute: 7 * 6\nAnswer:","targets":["42"],"metric_name":"exact_match","post_process":"strip_whitespace"}),
    json.dumps({"task_id":"mcq_001","category":"mcq","prompt":"What is 2+2? A:3 B:4 C:5\nAnswer:","targets":["B"],"metric_name":"exact_match","post_process":"extract_letter"}),
    json.dumps({"task_id":"mcq_002","category":"mcq","prompt":"Capital of France? A:Paris B:London C:Berlin\nAnswer:","targets":["A"],"metric_name":"exact_match","post_process":"extract_letter"}),
    json.dumps({"task_id":"code_001","category":"code_exec","prompt":"Write a function f(x)=x*2","targets":["ok"],"metric_name":"code_exec","post_process":"extract_code_block"}),
    json.dumps({"task_id":"code_002","category":"code_exec","prompt":"Write a function f(x)=x+1","targets":["ok"],"metric_name":"code_exec","post_process":"extract_code_block"}),
    json.dumps({"task_id":"cls_001","category":"classification","prompt":"Is this positive: I love it","targets":["positive"],"metric_name":"accuracy","post_process":"lower"}),
    json.dumps({"task_id":"cls_002","category":"classification","prompt":"Is this negative: I hate it","targets":["negative"],"metric_name":"accuracy","post_process":"lower"}),
    json.dumps({"task_id":"sum_001","category":"summary","prompt":"Summarize: The cat sat on the mat.","targets":["cat mat"],"metric_name":"rouge_l","post_process":"strip_whitespace"}),
    json.dumps({"task_id":"sum_002","category":"summary","prompt":"Summarize: Birds fly south in winter.","targets":["birds fly"],"metric_name":"rouge_l","post_process":"strip_whitespace"}),
]

def main():
    tasks_text = "\n".join(FIXTURES)
    specs, errors = validate_file(tasks_text)
    print(f"验证通过: {len(specs)} 条任务")
    print(f"错误: {len(errors)} 条")
    for e in errors:
        print(f"  行{e.line}: {e.field} — {e.rule}")

    print("\n类别分布:")
    from collections import Counter
    cats = Counter(s.category for s in specs)
    for cat, count in cats.items():
        print(f"  {cat}: {count}")

    print(f"\n渲染示例 (arith_001):")
    print(specs[0].render())
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 框架 | 任务格式 | 指标 |
|:----|:--------|:-----|
| BIG-bench | JSONL | 自定义 |
| HELM | YAML/JSON | 模块化 |
| lm-eval-harness | JSONL | 可扩展 |
| 本课 | JSONL | 闭式词汇 |

---

## 5. 工程最佳实践

- 每次添加新类别，必须同时添加新指标、后处理规则和至少一个固定任务
- 像数据库迁移一样对待规范——每个变更经过审查、版本化和测试
- **中文场景建议**：中文任务需要确保提示词和目标编码一致（UTF-8）

---

## 6. 常见错误

- **规范随实现漂移**：验证器必须拒绝不合规记录，而不是静默修复
- **后处理规则组合**：一条记录只能用一种后处理——组合使行为不可预测
- **few-shot 在提示词中硬编码**：运行器应统一渲染 few-shot，不要让任务作者在提示词中嵌入

---

## 7. 面试考点

**Q1：为什么闭式指标词汇表比开放词汇更好？**（难度：⭐⭐）

**参考答案：** 闭式词汇确保每个指标名称有确定的实现——下游代码通过 `if metric == "exact_match"` 分发。开放词汇导致同名不同实现、跨运行不可比较。添加新指标需要正式提案（新课程 + 新条目），防止随意扩展。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 任务规范 | JSONL 记录的 schema 合约 |
| 闭式指标词汇 | 可枚举的指标名称集合 |
| 后处理规则 | 生成输出在评分前的确定性变换 |
| 验证器 | 在记录到达运行器前检查合约 |

---

## 📚 小结

任务规范是评估框架的基石。你定义了 JSONL schema、闭式指标词汇和严格验证器。后续评估课程将基于此规范运行。

---

## ✏️ 练习

1. 【实现】添加 `extract_list` 后处理：提取方括号中的列表
2. 【实验】用错误格式的 JSONL 测试验证器，确认所有规则生效

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 任务规范 | `code/main.py` |
| 固定任务集 | `outputs/fixtures.jsonl` |

---

## 📖 参考资料

1. [论文] BIG-bench. https://arxiv.org/abs/2206.09550
2. [GitHub] lm-evaluation-harness. https://github.com/EleutherAI/lm-evaluation-harness
