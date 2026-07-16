# 异步任务（SEP-1686）——即时调用，延后获取

> 真实的智能体工作需要分钟到小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686（2025-11-25 合并）添加了 Tasks 原语：任何请求都可以被增强为任务，结果可以通过状态通知稍后获取或流式获取。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）、09（传输层）| **时间：** ~75 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 识别何时将工具从同步提升为任务增强（>30 秒的服务端工作）
- [ ] 实现异步任务状态机——创建、运行、完成、取消
- [ ] 设计任务进度通知——让 Client 实时了解任务状态
- [ ] 理解长运行任务的错误处理和重试策略

---

## 1. 问题

同步工具调用假设任务在几秒内完成。但 CI 运行需要 5-30 分钟，深度研究需要 10-60 分钟，批量导出需要数小时。同步调用在这些场景下会断开连接、超时或阻塞 UI。

SEP-1686（2025-11-25）添加了 Tasks 原语：将工具调用增强为任务——Client 可以稍后获取结果或通过状态通知流式获取。

---

## 2. 概念

### 2.1 任务状态机

```
创建 → 运行中 → 完成
              ↓
           失败（可选重试）
```

### 2.2 同步 vs 异步工具

| 方面 | 同步工具 | 异步任务 |
|------|---------|---------|
| 执行时间 | <30 秒 | >30 秒到数小时 |
| 结果获取 | 立即 | 稍后或流式 |
| 客户端体验 | 等待 | 非阻塞 |
| 资源使用 | 当前会话 | 独立会话 |

### 2.3 任务状态通知

```
Server → Client: notifications/tasks/progress
{
  "taskId": "task-123",
  "progress": 0.6,
  "message": "已完成 60%"
}
```

---

## 3. 从零实现

### Step 1：异步任务管理器

```python
import threading
import uuid

class AsyncTaskManager:
    """异步任务管理器。"""
    def __init__(self):
        self.tasks = {}

    def create_task(self, name, executor, args):
        """创建异步任务。"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {"status": "pending", "progress": 0, "result": None}

        def run():
            try:
                self.tasks[task_id]["status"] = "running"
                result = executor(**args)
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["result"] = result
            except Exception as e:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = str(e)

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        return task_id

    def get_status(self, task_id):
        return self.tasks.get(task_id, {"status": "not_found"})
```

### Step 2：任务状态通知

```python
def notify_progress(task_id, progress, message=""):
    """发送任务进度通知。"""
    return {
        "jsonrpc": "2.0",
        "method": "notifications/tasks/progress",
        "params": {
            "taskId": task_id,
            "progress": progress,
            "message": message,
        }
    }
```

### Step 3：任务取消

```python
def cancel_task(task_id):
    """取消正在运行的任务。"""
    return {
        "jsonrpc": "2.0",
        "method": "tasks/cancel",
        "params": {"taskId": task_id},
    }
```

---

## 4. 工具

### 4.1 MCP Tasks 规范

| 方法 | 说明 |
|------|------|
| `tasks/create` | 创建新任务 |
| `tasks/status` | 查询任务状态 |
| `tasks/cancel` | 取消任务 |
| `tasks/result` | 获取任务结果 |
| `notifications/tasks/progress` | 进度通知 |

### 4.2 最佳实践

- **超时设置**：为每个任务设置超时——防止资源泄漏
- **重试策略**：失败的任务支持重试——使用指数退避
- **结果缓存**：完成的任务结果可缓存——避免重复计算

---

## 5. 工程最佳实践

### 5.1 何时将工具提升为任务

| 指标 | 同步 | 异步 |
|------|------|------|
| 执行时间 | <30 秒 | >30 秒 |
| 资源消耗 | 低 | 高 |
| 用户体验 | 等待可接受 | 等待不可接受 |
| 并发性 | 低 | 高 |

### 5.2 踩坑经验

- **任务泄漏**：忘记清理完成的任务——内存泄漏
- **进度通知频率**：过高会导致网络拥塞，过低用户感知不到进度
- **取消后清理**：取消任务后需要释放资源

---

## 6. 常见错误

### 错误 1：同步调用长时间任务

**现象：** 客户端超时——无法获取结果。

**修复：** 超过 30 秒的任务应该用异步任务——Client 可以稍后获取结果。

### 错误 2：任务状态不一致

**现象：** Server 显示任务完成但 Client 认为仍在运行。

**修复：** 使用幂等性设计——相同请求 ID 返回相同状态。

---

## 7. 面试考点

### Q1：异步任务和同步工具调用有什么区别？（难度：⭐⭐）

**参考答案：**
同步工具调用在当前会话中立即执行——Client 必须等待结果。异步任务在独立会话中执行——Client 可以继续其他工作，稍后通过 `tasks/result` 获取结果。关键区别：异步任务支持状态通知（进度更新）和取消操作，而同步调用不支持。

### Q2：SEP-1686 的任务状态机是什么？（难度：⭐⭐⭐）

**参考答案：**
任务状态机有三个状态：创建（pending）→ 运行中（running）→ 完成（completed）或失败（failed）。失败的任务可以选择重试。Server 在状态变化时通过 `notifications/tasks/progress` 通知 Client 进度。Client 可以调用 `tasks/cancel` 取消正在运行的任务——Server 应优雅地停止执行并清理资源。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| 异步任务 | "后台任务" | MCP 的 Tasks 原语——长时间运行的工具调用，结果可稍后获取 |
| 任务状态机 | "任务生命周期" | 创建→运行→完成/失败的状态转换 |
| SEP-1686 | "任务规范" | 2025-11-25 规范——将工具调用增强为异步任务 |
| 进度通知 | "实时更新" | Server 通过 `notifications/tasks/progress` 通知 Client 任务进度 |

---

## 📚 小结

异步任务让长时间运行的工具调用变为非阻塞——Client 可以继续其他工作。SEP-1686 定义了任务状态机（创建→运行→完成）。关键是：正确设置超时、优雅处理取消、实时通知进度。对于 >30 秒的服务端工作，应始终使用异步任务。

---

## ✏️ 练习

1. **【实现】** 构建异步任务管理器——支持创建、查询、取消任务
2. **【实验】** 模拟长时间任务——观察状态通知的实时性

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|------|------|------|
| 异步任务管理器 | `code/main.py` | 任务状态机 + 进度通知 |

---

## 📖 参考资料

1. [文档] MCP Tasks 规范: https://spec.modelcontextprotocol.io
2. [文档] SEP-1686: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686
3. [文档] MCP 规范 2025-11-25

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。
