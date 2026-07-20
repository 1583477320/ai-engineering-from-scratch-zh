"""函数调用调度器——超时、重试、幂等、并发限制。"""
from __future__ import annotations
import asyncio, inspect, json, random, time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

ERR_METHOD_NOT_FOUND=-32601; ERR_INVALID_PARAMS=-32602; ERR_INTERNAL=-32603

class TransientError(Exception): pass
class _DispatchedError(Exception):
    def __init__(self,error): super().__init__(error.message); self.error=error

@dataclass
class DispatchError(Exception):
    kind:str; message:str; attempts:int; jsonrpc_code:int=ERR_INTERNAL
    def __post_init__(self): super().__init__(f"{self.kind}: {self.message}")

@dataclass
class DispatchOk: result:Any; attempts:int

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
        t=self.get(name).schema.get("type")
        if t=="object" and not isinstance(args,dict): return [f"expected object, got {type(args).__name__}"]
        return []

def _backoff(attempt): return 0.1*(4**(attempt-1))*(1+random.random()*0.5)

def _map_exception(exc,attempts):
    if isinstance(exc,_DispatchedError): return exc.error
    return DispatchError(kind="internal",message=f"{type(exc).__name__}: {exc}",attempts=attempts)

@dataclass
class _InFlight: future:asyncio.Future; started_at:float

class Dispatcher:
    def __init__(self,registry,*,max_attempts=3,concurrency=8,cache_ttl=60.0,sleep=None):
        self.registry=registry; self.max_attempts=max_attempts
        self._sem=asyncio.Semaphore(concurrency); self._inflight={}; self._cache={}; self._cache_ttl=cache_ttl; self._sleep=sleep or asyncio.sleep

    async def dispatch(self,name,args,*,timeout_ms_override=None,idempotency_key=None,budget_tool_calls_remaining=None):
        try: rec=self.registry.get(name)
        except KeyError: return DispatchError(kind="not_found",message=f"tool {name!r}",attempts=0,jsonrpc_code=ERR_METHOD_NOT_FOUND)
        errs=self.registry.validate(name,args)
        if errs: return DispatchError(kind="schema",message="; ".join(errs),attempts=0,jsonrpc_code=ERR_INVALID_PARAMS)
        if budget_tool_calls_remaining is not None and budget_tool_calls_remaining<=0:
            return DispatchError(kind="budget_exceeded",message="no calls left",attempts=0)
        if idempotency_key:
            now=time.monotonic(); cached=self._cache.get(idempotency_key)
            if cached and now-cached[1]<self._cache_ttl: return DispatchOk(result=cached[0],attempts=0)
            inflight=self._inflight.get(idempotency_key)
            if inflight:
                try: return DispatchOk(result=await inflight.future,attempts=0)
                except Exception as exc: return _map_exception(exc,0)
        async with self._sem: return await self._run_with_retries(rec,args,timeout_ms_override,idempotency_key)

    async def _run_with_retries(self,rec,args,timeout_override,idem_key):
        timeout_s=(timeout_override if timeout_override is not None else rec.timeout_ms)/1000.0
        attempt=0; last_error=None; loop=asyncio.get_running_loop(); future=None
        if idem_key: future=loop.create_future(); self._inflight[idem_key]=_InFlight(future=future,started_at=time.monotonic())
        try:
            while attempt<self.max_attempts:
                attempt+=1
                try: result=await asyncio.wait_for(_invoke(rec.handler,args),timeout=timeout_s)
                except asyncio.TimeoutError:
                    last_error=DispatchError(kind="timeout",message=f"timeout {rec.timeout_ms}ms",attempts=attempt)
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
            if future and not future.done(): future.set_exception(_DispatchedError(last_error))
            return last_error
        finally:
            if idem_key: self._inflight.pop(idem_key,None)

async def _invoke(handler,args):
    if inspect.iscoroutinefunction(handler): return await handler(**args)
    r=handler(**args)
    if inspect.isawaitable(r): return await r
    return r

async def _demo():
    reg=MiniRegistry(); c={"a":0}
    async def flaky(id:int):
        c["a"]+=1
        if c["a"]<2: raise TransientError("upstream not ready")
        return {"id":id,"name":"ada"}
    async def slow(n:int): await asyncio.sleep(0.05); return n
    reg.register("fetch_user",{"type":"object","required":["id"],"properties":{"id":{"type":"integer"}}},flaky,idempotent=True,timeout_ms=200)
    reg.register("slow",{"type":"object","required":["n"],"properties":{"n":{"type":"integer"}}},slow,idempotent=True,timeout_ms=10)
    reg.register("noop",{"type":"object","properties":{}},lambda:"ok")
    d=Dispatcher(reg,max_attempts=3,concurrency=4)
    r1=await d.dispatch("fetch_user",{"id":42}); r2=await d.dispatch("slow",{"n":1})
    r3=await d.dispatch("fetch_user",{"id":"x"}); r4=await d.dispatch("does_not_exist",{})
    r5=await d.dispatch("noop",{})
    a,b=await asyncio.gather(d.dispatch("noop",{},idempotency_key="k1"),d.dispatch("noop",{},idempotency_key="k1"))
    print(json.dumps({
        "retry": {"attempts":getattr(r1,"attempts"),"ok":isinstance(r1,DispatchOk)},
        "timeout": {"kind":getattr(r2,"kind",None)}, "schema": {"kind":getattr(r3,"kind",None)},
        "missing": {"kind":getattr(r4,"kind",None)}, "happy": {"result":getattr(r5,"result",None)},
        "idempotency": [isinstance(a,DispatchOk),isinstance(b,DispatchOk)],
    },indent=2))

if __name__=="__main__": asyncio.run(_demo())
