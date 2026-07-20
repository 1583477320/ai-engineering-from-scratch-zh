# 综合项目19——工具注册与JSON Schema模式验证

> 智能体无法验证的工具就是无法调用的工具。在构建工具之前，先构建注册表和模式检查器。一个2026年的编码智能体注册的工具比模型单次上下文窗口能容纳的更多。注册表是"什么工具存在"、"参数是什么形状"、"调用什么处理程序"的唯一真实来源。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第13章（工具与协议）第01-07节、第14章（智能体）第01节
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 维护工具名称→模式→处理程序的类型化注册表
- 实现覆盖90%工具调用的JSON Schema 2020-12子集
- 返回精确的JSON指针错误路径，使模型能一轮自我纠正
- 拒绝无显式覆盖的重复注册
- 保持验证器纯净（无I/O、无时间、无全局变量）

---

## 1. 问题

2026年的编码智能体注册的工具比模型单次上下文窗口能容纳的更多。非平凡的主循环将注册200个工具，每次轮次显示10-40个。

我们避免的错误是发布没有模式的处理程序，或发布没有验证的模式。两者都很常见。两者都将下一层（调度器）变成猜测游戏，唯一的失败模式是处理程序的堆栈跟踪。

---

## 2. 核心概念

### 2.1 工具记录

```
ToolRecord
  name        : str          (唯一，小写字母数字和下划线段用点分隔)
  description : str          (一行，显示给模型)
  schema      : dict         (JSON Schema 2020-12子集)
  handler     : Callable     (异步或同步，返回Any)
  idempotent  : bool         (调度器用此决定重试)
  timeout_ms  : int          (覆盖每工具调度器默认值)
```

### 2.2 JSON Schema 2020-12子集

完整规范是论文。我们需要八个关键字：

- **type**: string/number/integer/boolean/object/array/null
- **properties**: 属性名→模式的映射
- **required**: 必需属性名列表
- **enum**: 允许的原始值列表
- **minLength**: 字符串最小长度
- **maxLength**: 字符串最大长度
- **pattern**: ECMA-262兼容正则
- **items**: 应用于每个数组元素的模式

### 2.3 JSON指针错误路径

验证失败时，验证器返回错误列表，每个错误携带指向输入的JSON指针路径。模型比读句子更好地读错误路径——如果模式要求`args.user.email`而模型传递了整数，错误应为`/user/email`，带`expected_type: string`。

---

## 3. 从零实现

`code/main.py`实现`ToolRegistry`、`ToolRecord`、`ValidationError`和八个验证器函数。

```python
"""工具注册表+JSON Schema 2020-12子集验证。

核心：名称键控的工具记录表，带模式验证。
验证器是纯净的（无I/O），可重放。

运行：python3 code/main.py
"""

from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Any, Callable


PRIMITIVE_TYPE_MAP = {
    "string": (str,), "integer": (int,), "number": (int, float),
    "boolean": (bool,), "object": (dict,), "array": (list,), "null": (type(None),),
}

ALLOWED_KEYWORDS = {"type", "properties", "required", "enum", "minLength", "maxLength", "pattern", "items", "description"}


@dataclass
class ValidationError:
    path: str; keyword: str; message: str
    def to_dict(self): return {"path": self.path, "keyword": self.keyword, "message": self.message}


@dataclass
class Ok: pass


@dataclass
class ToolRecord:
    name: str; description: str; schema: dict; handler: Callable[..., Any]
    idempotent: bool = False; timeout_ms: int = 30_000


class ToolRegistry:
    _NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")

    def __init__(self): self._records = {}; self._order = []

    def register(self, name, schema, handler, description="", idempotent=False, timeout_ms=30_000, override=False):
        if not self._NAME_RE.match(name): raise ValueError(f"name {name!r} invalid")
        if name in self._records and not override: raise ValueError(f"{name!r} already registered; use override=True")
        validate_schema_shape(schema)
        rec = ToolRecord(name=name, description=description, schema=schema, handler=handler, idempotent=idempotent, timeout_ms=timeout_ms)
        if name not in self._records: self._order.append(name)
        self._records[name] = rec
        return rec

    def get(self, name): 
        if name not in self._records: raise KeyError(f"unknown tool {name!r}")
        return self._records[name]

    def names(self): return list(self._order)

    def validate(self, name, args):
        rec = self.get(name); errors = []
        _walk(rec.schema, args, "", errors)
        return Ok() if not errors else errors


def validate_schema_shape(schema):
    if not isinstance(schema, dict): raise ValueError("schema must be a dict")
    unknown = set(schema.keys()) - ALLOWED_KEYWORDS
    if unknown: raise ValueError(f"unsupported keywords: {sorted(unknown)}")
    t = schema.get("type")
    if t is not None and t not in PRIMITIVE_TYPE_MAP: raise ValueError(f"unsupported type: {t!r}")
    for sub_key in ("properties", "items"):
        sub = schema.get(sub_key)
        if sub is not None and isinstance(sub, dict):
            validate_schema_shape(sub)


def _path(prefix, segment):
    seg = str(segment).replace("~", "~0").replace("/", "~1")
    return f"{prefix}/{seg}"


def _type_matches(value, expected):
    if expected == "boolean": return isinstance(value, bool)
    if expected in ("integer", "number"): return not isinstance(value, bool) and isinstance(value, PRIMITIVE_TYPE_MAP[expected])
    return isinstance(value, PRIMITIVE_TYPE_MAP[expected])


def _walk(schema, value, path, errs):
    t = schema.get("type")
    if t is not None and not _type_matches(value, t):
        errs.append(ValidationError(path or "/", "type", f"expected {t}, got {type(value).__name__}"))
        return
    if "enum" in schema and value not in schema["enum"]:
        errs.append(ValidationError(path or "/", "enum", f"{value!r} not in {schema['enum']!r}"))
        return
    if t == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(ValidationError(path, "minLength", f"length {len(value)} < {schema['minLength']}"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(ValidationError(path, "maxLength", f"length {len(value)} > {schema['maxLength']}"))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(ValidationError(path, "pattern", f"no match for {schema['pattern']!r}"))
    elif t == "object":
        for req in schema.get("required", []):
            if req not in value:
                errs.append(ValidationError(_path(path, req), "required", f"missing {req!r}"))
        for pname, pval in value.items():
            if pname in schema.get("properties", {}):
                _walk(schema["properties"][pname], pval, _path(path, pname), errs)
    elif t == "array":
        items = schema.get("items")
        if items:
            for i, item in enumerate(value):
                _walk(items, item, _path(path, i), errs)


def _demo():
    registry = ToolRegistry()
    registry.register("db.get_user", {"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}}, lambda id: {"id": id}, description="Fetch user by id")
    
    cases = [{"id": 42}, {"id": "bad"}, {}, {"id": 1}]
    for c in cases:
        r = registry.validate("db.get_user", c)
        status = "OK" if isinstance(r, Ok) else f"ERRORS: {[e.to_dict() for e in r]}"
        print(f"  {c} -> {status}")
    
    print(f"\nRegistered tools: {registry.names()}")


if __name__ == "__main__":
    _demo()
```

运行结果：

```
  {'id': 42} -> OK
  {'id': 'bad'} -> ERRORS: [{'path': '/id', 'keyword': 'type', 'message': 'expected integer, got str'}]
  {} -> ERRORS: [{'path': '/id', 'keyword': 'required', 'message': "missing 'id'"}]
  {'id': 1} -> OK

Registered tools: ['db.get_user']
```

---

## 4. 工具实践

**使用方式**：
- `ToolRegistry`维护名称键控的工具记录表
- `register()`注册工具，`override=True`允许覆盖
- `validate()`返回`Ok`或`ValidationError`列表
- 验证器是纯净的——可在重放日志上重运行

---

## 5. LLM视角

**验证优先视角**：没有模式的工具调用是猜测游戏。模型需要精确的错误路径来自我纠正——`/user/email`比"email字段格式错误"更有效。

**纯净验证器视角**：验证器不调用处理程序、不强制类型转换、不静默截断。它是数据验证，不是安全边界。

---

## 6. 工程最佳实践

**命名规范**：小写字母数字+下划线，用点分隔段（如`db.get_user`）。

**注册保护**：默认拒绝重复注册，`override=True`显式覆盖。

**错误路径**：使用JSON指针格式（RFC 6901），模型可直接读取。

---

## 7. 常见错误

**错误1：不验证模式就注册工具**
症状：运行时才发现参数不匹配
修复：注册时验证模式结构

**错误2：静默覆盖已注册的工具**
症状：生产工具目录漂移
修复：默认拒绝重复注册

---

## 8. 面试考点

**Q1：为什么工具注册表要在工具之前构建？**
考察：对架构顺序的理解

**Q2：JSON Schema子集为什么选择这八个关键字？**
考察：对需求分析的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 工具注册表 | "工具目录" | 名称键控的工具记录表，含模式和处理程序 |
| JSON Schema子集 | "八关键字验证" | 覆盖90%工具调用的最小子集 |
| JSON指针 | "错误路径" | RFC 6901格式的精确错误位置 |
| 纯净验证器 | "无副作用验证" | 不调用处理程序、不强制类型转换的验证器 |
| 重复注册拒绝 | "操作卫生" | 默认拒绝覆盖已注册工具 |

---

## 参考文献

- [JSON Schema 2020-12规范](https://json-schema.org/draft/2020-12/json-schema-core)
- [RFC 6901 JSON指针](https://datatracker.ietf.org/doc/html/rfc6901)
