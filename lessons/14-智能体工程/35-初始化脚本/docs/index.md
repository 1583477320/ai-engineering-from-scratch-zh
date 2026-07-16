# 智能体的初始化脚本

> 每个冷启动的会话都付出一次代价。智能体读取相同的文件，重试相同的探测，重新发现相同的路径。init 脚本付出一次代价，把答案写入状态。

**类型：** 概念课 | **语言：** Python | **前置知识：** 阶段 14 · 32（最小工作台）、34（仓库记忆）| **时间：** ~45 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别智能体每次会话都不应该重做的工作
- [ ] 实现初始化脚本——探测环境并将结果写入仓库记忆
- [ ] 设计脚本的缓存和过期策略
- [ ] 理解为什么 init 脚本是工作台的关键组件

---

## 1. 问题

每次冷启动时，智能体都要：
- 读取项目文件了解结构
- 探测可用的工具和API
- 确定当前工作目录和环境

这些工作每次会话都一样——但智能体每次都重做。**init 脚本付出一次代价，把答案写入状态。**

---

## 2. 概念

### 2.1 init 脚本的工作

```python
# init.sh —— 每次会话开始时执行
INIT_SCRIPT = [
    "读取项目 README",
    "探测可用工具",
    "检查依赖配置",
    "加载历史记忆",
    "建立环境快照",
]
```

### 2.2 缓存 vs 实时探测

| 数据 | 缓存策略 | 说明 |
|------|---------|------|
| 项目结构 | 会话缓存 | 会话内不变 |
| 工具列表 | 每日过期 | 工具可能变动 |
| API 状态 | 实时探测 | 可能不可用 |
| 用户偏好 | 永久缓存 | 不太变动 |

### 2.3 init 脚本结果

```python
INIT_RESULT = {
    "project_structure": {"dirs": ["src/", "tests/"], "files": ["main.py"]},
    "available_tools": ["search", "calculate"],
    "dependencies": {"installed": ["torch", "transformers"], "missing": ["langchain"]},
    "workspace_state": {"branch": "main", "last_commit": "abc123"},
}
```

---

## 3. 从零实现

### Step 1：初始化脚本

```python
import time

class InitScript:
    """初始化脚本——执行一次，结果缓存。"""
    def __init__(self, repo_memory):
        self.repo_memory = repo_memory
        self.probes = []
        self.cache_ttl = 3600  # 1 小时缓存

    def add_probe(self, name, probe_fn, ttl=3600):
        self.probes.append({"name": name, "fn": probe_fn, "ttl": ttl})

    def run(self):
        """运行所有探测，缓存结果。"""
        results = {}
        for probe in self.probes:
            cached = self.repo_memory.retrieve(f"init_{probe['name']}")
            if cached and cached.get("_cached_at", 0) + probe["ttl"] > time.time():
                results[probe["name"]] = cached["value"]
                continue

            value = probe["fn"]()
            self.repo_memory.store(f"init_{probe['name']}", {
                "value": value, "_cached_at": time.time()
            })
            results[probe["name"]] = value

        return results
```

### Step 2：常用探测

```python
def probe_environment():
    """探测运行环境。"""
    return {
        "python_version": "3.11",
        "cuda_available": False,
        "project_type": "python",
    }

def probe_dependencies():
    """探测依赖状态。"""
    return {"installed": ["torch"], "missing": ["transformers"]}
```

---

## 4. 工具

### 4.1 探测函数

| 探测 | 结果 | TTL |
|------|------|------|
| 项目结构 | 目录和文件列表 | 会话 |
| 环境 | Python 版本、CUDA | 会话 |
| 依赖 | 已安装和缺失的包 | 小时级 |

---

## 5. 工程最佳实践

- **避免重复探测**：缓存结果
- **TTL合理**：稳定的信息用长 TTL
- **优雅降级**：探测失败时使用默认值

---

## 6. 常见错误

### 错误 1：没有 init 脚本

**现象：** 每次会话智能体都重复探测环境——浪费时间。

**修复：** init 脚本一次完成探测，缓存结果供整个会话使用。

---

## 7. 面试考点

### Q1：init 脚本解决了什么问题？（难度：⭐⭐）

**参考答案：**
每个冷启动的会话都付出一次代价——读取项目文件、探测工具、检查依赖。这些工作每次都一样。init 脚本一次做完，结果存入仓库记忆。后续会话直接读取缓存，无需重做。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| init 脚本 | "初始化探测" | 会话开始时一次执行，结果缓存供整个会话使用 |
| 冷启动 | "新会话" | 新会话开始时空白状态——没有之前的上下文 |
| 探测 | "检查环境" | 读取项目结构、检查依赖、可用工具 |

---

## 📚 小结

Init 脚本解决重复探测问题——每次会话开始时一次执行，结果缓存到仓库记忆。智能体不再需要每次重新发现路径。关键：探测→缓存→复用。

---

## ✏️ 练习

1. **【设计】** 为一个 Python 项目设计初始化脚本——需要探测哪些内容？
2. **【实现】** 实现 init 脚本的执行引擎——探测 + 缓存 + TTL 过期

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 初始化脚本 | `code/main.py` | 探测 + 缓存 + TTL |

---

## 📖 参考资料

1. [文档] Claude Agent SDK: Session Store
2. [文档] Letta: https://github.com/letta-ai/letta
3. [博客] Anthropic. Building Effective Agents. 2024.
