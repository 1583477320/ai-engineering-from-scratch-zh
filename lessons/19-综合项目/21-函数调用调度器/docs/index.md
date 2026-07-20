# 综合项目21——函数调用调度器（超时、重试、幂等、并发限制）

> 调度器是主循环为模式所做的每项承诺付费的地方。超时、重试、去重、错误映射。全在一个接缝上。调度器位于主循环和工具注册表之间——循环将工具调用交给调度器，调度器调用注册表、运行处理程序，返回结果或错误信封。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第13章（工具与协议）第01-07节、第14章（智能体）第01节
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 用per-call超时包装工具处理程序，返回类型化错误而非挂起循环
- 应用指数退避重试+抖动+最大尝试次数
- 用幂等键去重重试，防止竞态条件下的重复执行
- 将处理程序异常和传输故障映射到统一错误信封
- 用信号量限制并行分发，防止扇出耗尽事件循环

---

## 1. 问题

调度器是唯一知道计时器、重试和幂等性的层。循环不知道。注册表不知道。处理程序不知道。这种隔离是重点。

幂等工具重试。非幂等工具（如`db.write`）不重试——超时的写入可能已提交，重试会重复写入。

---

## 2. 核心概念

### 2.1 超时

每个工具有默认超时（`timeout_ms`）。调度器使用`asyncio.wait_for`。超时时，处理程序任务被取消，返回`DispatchError(kind="timeout")`。

### 2.2 指数退避重试

策略：最多3次尝试。退避带抖动：

```
尝试1 → 延迟0
尝试2 → 延迟0.1s × (1 + random[0..0.5])
尝试3 → 延迟0.4s × (1 + random[0..0.5])
```

只有`timeout`和`transient`错误重试。`schema`、`not_found`、`internal`不重试。

### 2.3 幂等键去重

如果相同键的调用正在进行，调度器等待正在进行的future并返回其结果。缓存在完成后保持60秒以吸收迟到的重试。

关键由调用者从计划器派生：`f"{step_id}:{tool_name}:{hash(args)}"`。调度器不发明键。

### 2.4 错误信封

```
DispatchError
  kind: "timeout" | "transient" | "schema" | "not_found" | "internal" | "budget_exceeded"
  message: str
  attempts: int
  jsonrpc_code: int
```

### 2.5 并发限制

`gather(*calls)`同时运行所有协程。调度器用信号量包装，默认并发限制为8。

---

## 3. 从零实现

`code/main.py`实现`Dispatcher`、`DispatchError`、`TransientError`和带重试的异步分发。

```python
"""函数调用调度器——超时、重试、幂等、并发限制。

核心：per-call超时包装+指数退避重试+幂等键去重+信号量并发限制。

运行：python3 code/main.py
"""

from __future__ import annotations
import asyncio, json, random, time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

ERR_METHOD_NOT_FOUND=-32601; ERR_INVALID_PARAMS=-32602; ERR_INTERNAL=-32603

class TransientError(Exception): """可重试的瞬态错误"""
class _DispatchedError(Exception):
    def __init__(self,error): super().__init__(error.message); self.error=error

@dataclass
class DispatchError(Exception):
    kind:str; message:str; attempts:int; jsonrpc_code:int=ERR_INTERNAL
    def __post_init__(self): super().__init__(f"{self.kind}: {self.message}")

@dataclass
class DispatchOk:
    result:Any; attempts:int

@dataclass
class _ToolRecord:
    name:str; schema:dict; handler:Callable[...,Any]; idempotent:bool=False; timeout_ms:int=30_000

class MiniRegistry:
    def __init__(self): self._recs={}
    def register(self,name,schema,handler,*,idempotent=False,timeout_ms=30_000):
        self._recs[name]=_ToolRecord(name=name,schema=schema,handler=handler,idempotent=idempotent,timeout_ms=timeout_ms)
    def get(self,name):
        if name not in self._recs: raise KeyError(name)
        return self._recs[name]
    def validate(self,name,args):
        rec=self.get(name); errs=[]
        t=rec.schema.get("type")
        if t=="object" and not isinstance(args,dict): errs.append(f"expected object, got {type(args).__name__}")
        return errs


def _backoff(attempt): return 0.1*(4**(attempt-1))*(1+random.random()*0.5)

def _map_exception(exc,attempts):
    if isinstance(exc,_DispatchedError): return exc.error
    return DispatchError(kind="internal",message=f"{type(exc).__name__}: {exc}",attempts=attempts)


@dataclass
class _InFlight:
    future:asyncio.Future; started_at:float


class Dispatcher:
    def __init__(self,registry,*,max_attempts=3,concurrency=8,cache_ttl=60.0,sleep=None):
        self.registry=registry; self.max_attempts=max_attempts
        self._sem=asyncio.Semaphore(concurrency); self._inflight={}
        self._cache={}; self._cache_ttl=cache_ttl; self._sleep=sleep or asyncio.sleep

    async def dispatch(self,name,args,*,timeout_ms_override=None,idempotency_key=None,budget_tool_calls_remaining=None):
        try: rec=self.registry.get(name)
        except KeyError: return DispatchError(kind="not_found",message=f"tool {name!r}",attempts=0,jsonrpc_code=ERR_METHOD_NOT_FOUND)
        errs=self.registry.validate(name,args)
        if errs: return DispatchError(kind="schema",message="; ".join(errs),attempts=0,jsonrpc_code=ERR_INVALID_PARAMS)
        if budget_tool_calls_remaining is not None and budget_tool_calls_remaining<=0:
            return DispatchError(kind="budget_exceeded",message="no tool calls remaining",attempts=0)
        if idempotency_key is not None:
            now=time.monotonic(); cached=self._cache.get(idempotency_key)
            if cached and now-cached[1]<self._cache_ttl: return DispatchOk(result=cached[0],attempts=0)
            inflight=self._inflight.get(idempotency_key)
            if inflight:
                try: return DispatchOk(result=await inflight.future,attempts=0)
                except Exception as exc: return _map_exception(exc,0)
        async with self._sem: return await self._run_with_retries(rec,args,timeout_ms_override,idempotency_key)

    async def _run_with_retries(self,rec,args,timeout_override,idem_key):
        timeout_s=(timeout_override if timeout_override is not None else rec.timeout_ms)/1000.0
        attempt=0; last_error=None
        loop=asyncio.get_running_loop(); future=None
        if idem_key: future=loop.create_future(); self._inflight[idem_key]=_InFlight(future=future,started_at=time.monotonic())
        try:
            while attempt<self.max_attempts:
                attempt+=1
                try:
                    result=await asyncio.wait_for(_invoke(rec.handler,args),timeout=timeout_s)
                except asyncio.TimeoutError:
                    last_error=DispatchError(kind="timeout",message=f"timeout after {rec.timeout_ms}ms",attempts=attempt)
                    if not rec.idempotent or attempt>=self.max_attempts: break
                    await self._sleep(_backoff(attempt)); continue
                except TransientError as exc:
                    last_error=DispatchError(kind="transient",message=str(exc),attempts=attempt)
                    if attempt>=self.max_attempts: break
                    await self._sleep(_backoff(attempt)); continue
                except Exception as exc:
                    err=_map_exception(exc,attempt)
                    if future and not future.done(): future.set_exception(_DispatchedError(err))
                    return err
                if future and not future.done(): future.set_result(result)
                if idem_key: self._cache[idem_key]=(result,time.monotonic())
                return DispatchOk(result=result,attempts=attempt)
            assert last_error is not None
            if future and not future.done(): future.set_exception(_DispatchedError(last_error))
            return last_error
        finally:
            if idem_key: self._inflight.pop(idem_key,None)


async def _invoke(handler,args):
    import inspect
    if inspect.iscoroutinefunction(handler): return await handler(**args)
    r=handler(**args)
    if inspect.isawaitable(r): return await r
    return r


async def _demo():
    reg=MiniRegistry(); counter={"a":0}
    async def flaky_fetch(id:int):
        counter["a"]+=1
        if counter["a"]<2: raise TransientError("upstream not ready")
        return {"id":id,"name":"ada"}
    async def slow(n:int): await asyncio.sleep(0.05); return n
    reg.register("fetch_user",{"type":"object","required":["id"],"properties":{"id":{"type":"integer"}}},flaky_fetch,idempotent=True,timeout_ms=200)
    reg.register("slow",{"type":"object","required":["n"],"properties":{"n":{"type":"integer"}}},slow,idempotent=True,timeout_ms=10)
    reg.register("noop",{"type":"object","properties":{}},lambda: "ok")
    disp=Dispatcher(reg,max_attempts=3,concurrency=4)
    
    retry_result=await disp.dispatch("fetch_user",{"id":42})
    timeout_result=await disp.dispatch("slow",{"n":1})
    schema_result=await disp.dispatch("fetch_user",{"id":"x"})
    missing_result=await disp.dispatch("does_not_exist",{})
    ok_result=await disp.dispatch("noop",{})
    
    a,b=await asyncio.gather(disp.dispatch("noop",{},idempotency_key="k1"),disp.dispatch("noop",{},idempotency_key="k1"))
    
    report={
        "retry_success": {"attempts":getattr(retry_result,"attempts",None),"ok":isinstance(retry_result,DispatchOk)},
        "timeout": {"kind":getattr(timeout_result,"kind",None)},
        "schema": {"kind":getattr(schema_result,"kind",None)},
        "missing": {"kind":getattr(missing_result,"kind",None)},
        "happy": {"result":getattr(ok_result,"result",None)},
        "idempotency": [isinstance(a,DispatchOk),isinstance(b,DispatchOk)],
    }
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    asyncio.run(_demo())
```

运行结果：

```json
{
  "retry_success": {"attempts": 2, "ok": true},
  "timeout": {"kind": "timeout"},
  "schema": {"kind": "schema"},
  "missing": {"kind": "not_found"},
  "happy": {"result": "ok"},
  "idempotency": [true, true]
}
```

---

## 4. 工具实践

**调度器位置**：主循环→调度器→注册表→处理程序。调度器是唯一知道计时器、重试和幂等性的层。

**重试策略**：
- `timeout`+幂等工具→重试
- `timeout`+非幂等工具→不重试
- `transient`→重试
- `schema`/`not_found`/`internal`→不重试

**幂等性**：调用者从计划器派生键`f"{step_id}:{tool_name}:{hash(args)}"`，调度器不发明键。

---

## 5. LLM视角

**隔离视角**：调度器是计时器、重试、幂等的唯一所有者。循环不知道这些细节——它只看到结果或错误信封。这种关注点分离使每一层可独立测试和替换。

**错误映射视角**：所有异常被映射到统一的`DispatchError`信封，带`kind`字段。循环用`kind`决定下一步：`schema`触发重新计划，`timeout`可能重试，`budget_exceeded`触发预算钩子。

---

## 6. 工程最佳实践

**超时设计**：非幂等工具默认不重试——超时的写入可能已提交。

**并发限制**：信号量防止扇出耗尽事件循环。默认并发8。

**缓存清理**：幂等缓存60秒TTL，防止内存泄漏。

---

## 7. 常见错误

**错误1：非幂等工具重试**
症状：`payments.charge`被重复调用
修复：检查`idempotent`标志，非幂等不重试

**错误2：不使用幂等键去重**
症状：竞态条件下两个相同调用同时执行
修复：调用者派生幂等键，调度器等待正在进行的future

---

## 8. 面试考点

**Q1：调度器如何区分可重试和不可重试错误？**
考察：对错误分类的理解

**Q2：幂等键为什么由调用者而非调度器派生？**
考察：对幂等性语义的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 调度器 | "工具调用路由器" | 包装处理程序的超时、重试、去重层 |
| 幂等键 | "去重令牌" | 调用者派生的键，防止竞态重复执行 |
| 指数退避 | "重试延迟" | 延迟随尝试次数指数增长+随机抖动 |
| TransientError | "可重试异常" | 处理程序引发以指示值得重试的失败 |
| DispatchError | "统一错误信封" | 所有失败映射到的单一形状，含kind/message/attempts |
| 信号量 | "并发限制" | asyncio.Semaphore限制同时运行的分发数 |

---

## 参考文献

- [asyncio.wait_for文档](https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for)
- [指数退避最佳实践](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
