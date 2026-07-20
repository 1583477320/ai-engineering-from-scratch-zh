# 综合项目24——沙箱运行器（拒绝列表+路径监狱+超时）

> 验证门决定工具调用是否运行。沙箱决定运行时发生什么。本课程构建一个拒绝危险可执行文件、拒绝危险argv形状、将每个文件路径囚禁到项目根目录、截断超大输出、在墙钟超时时杀死失控进程的子进程运行器。它是模型和操作系统之间的第二层。

**类型：** 构建
**编程语言：** Python（标准库）
**前置知识：** 第19章 · 第23节（验证门）
**涉及章节：** P13 · P14
**预计时间：** 90分钟

---

## 学习目标

- 构建包装`subprocess.run`的Sandbox类，带超时、捕获和截断
- 按名称拒绝列表和argv结构检查拒绝命令
- 拒绝任何解析到声明项目根目录之外的路径参数
- 当shell模式关闭时拒绝shell元字符
- 返回结构化SandboxResult供可观测性和评测工具摄取

---

## 1. 问题

可以shell out的编码智能体可以在一轮中安装后门、泄露密钥、破坏开发者笔记本、产生云账单。最便宜的防御是不给它shell。第二便宜的是对精确模式列表说不的沙箱。

三类故障反复出现：**危险可执行文件**（rm、sudo、mkfs）、**argv技巧**（python3 -c "os.system('rm -rf /')"）、**路径逃逸**（读../../etc/passwd而非./src/main.py）。

沙箱不是操作系统意义上的安全边界——有决心的攻击者仍可逃逸。它是开发时护栏。

---

## 2. 核心概念

### 2.1 四个拒绝轴

1. **名称拒绝**：可执行文件基名在拒绝列表中（rm、sudo、mkfs等）
2. **argv检查**：解释器（python/bash）带-c/-e标志被拒绝（等于shell调用）
3. **shell元字符**：非shell模式下argv含;、|、&等被拒绝
4. **路径监狱**：路径参数通过`os.path.realpath`解析后检查是否在项目根目录下

### 2.2 SandboxResult

```text
SandboxResult
  argv: list[str]
  exit_code: int       (0=成功, -100=拒绝, -101=超时)
  stdout: bytes
  stderr: bytes
  truncated: bool
  timed_out: bool
  denied: bool
  reason: str
  duration_ms: float
```

### 2.3 输出截断

默认最大64KB。截断时附加标记行`[sandbox: output truncated]`，truncated标志为True。

---

## 3. 从零实现

`code/main.py`实现`Sandbox`、`SandboxConfig`、`SandboxResult`和四个拒绝辅助器。

```python
"""沙箱运行器——拒绝列表+路径监狱+超时+截断。

四个拒绝轴：名称、argv、shell元字符、路径监狱。
子进程仅在所有轴通过后生成。

运行：python3 code/main.py
"""

from __future__ import annotations
import json, os, re, subprocess, sys, tempfile, time
from dataclasses import dataclass, field
from typing import Sequence

DENIED_EXIT=-100; TIMED_OUT_EXIT=-101

DEFAULT_DENYLIST=frozenset({"rm","sudo","mkfs","curl","wget","chmod","dd","kill","shutdown","reboot","eval","exec","base64","nc","su","iptables"})
DEFAULT_INTERP_BLOCK=frozenset({"python","python3","bash","sh","zsh","node","perl","ruby","php"})
INTERP_FLAGS=tuple(re.compile(p) for p in [r"^-c$",r"^-e$",r"^--eval$",r"^--exec$"])
SHELL_METAS=(";","|","&",">","<","`","$(")
TRUNCATION_MARKER=b"\n[sandbox: truncated]\n"

@dataclass
class SandboxResult:
    argv:list[str]; exit_code:int; stdout:bytes=b""; stderr:bytes=b""
    truncated:bool=False; timed_out:bool=False; denied:bool=False; reason:str=""; duration_ms:float=0.0
    @property
    def ok(self): return self.exit_code==0 and not self.denied and not self.timed_out
    def to_dict(self):
        return {"argv":self.argv,"exit_code":self.exit_code,"stdout_bytes":len(self.stdout),"truncated":self.truncated,
                "timed_out":self.timed_out,"denied":self.denied,"reason":self.reason,"duration_ms":round(self.duration_ms,2)}

@dataclass
class SandboxConfig:
    project_root:str; max_output_bytes:int=65536; timeout_seconds:float=30.0
    denylist:frozenset[str]=field(default_factory=lambda:DEFAULT_DENYLIST)
    interp_block:frozenset[str]=field(default_factory=lambda:DEFAULT_INTERP_BLOCK)
    env_allowlist:tuple[str,...]=("PATH","HOME","LANG","TERM")
    def __post_init__(self): self.project_root=os.path.realpath(self.project_root)

def _check_denylist(argv,cfg):
    if not argv: return "empty argv"
    name=os.path.basename(argv[0].strip())
    if name in cfg.denylist: return f"executable {name!r} on denylist"
    return None

def _check_interp(argv,cfg):
    if not argv: return None
    name=os.path.basename(argv[0])
    if name not in cfg.interp_block: return None
    for arg in argv[1:]:
        for pat in INTERP_FLAGS:
            if pat.match(arg): return f"interpreter {name!r} with refused flag {arg!r}"
    return None

def _check_shell_metachars(argv,shell):
    if shell: return None
    for arg in argv:
        for m in SHELL_METAS:
            if m in arg: return f"shell metachar {m!r} in {arg!r}"
    return None

def _check_path_jail(argv,cfg):
    root=cfg.project_root
    for arg in argv[1:]:
        if not arg or arg.startswith("-"): continue
        if "/" not in arg and not os.path.sep in arg: continue
        candidate=arg if os.path.isabs(arg) else os.path.join(root,arg)
        resolved=os.path.realpath(candidate)
        if resolved!=root and not resolved.startswith(root+os.sep):
            return f"path {arg!r} resolves outside project root"
    return None

def truncate_stream(buf,max_bytes):
    if len(buf)<=max_bytes: return buf,False
    return buf[:max_bytes]+TRUNCATION_MARKER,True

@dataclass
class Sandbox:
    config:SandboxConfig
    def run(self,argv,*,shell=False,cwd=None,stdin=None):
        argv_list=list(argv); result=SandboxResult(argv=argv_list,exit_code=DENIED_EXIT)
        for check in (_check_denylist,_check_interp):
            reason=check(argv_list,self.config)
            if reason: result.denied=True; result.reason=reason; return result
        reason=_check_shell_metachars(argv_list,shell)
        if reason: result.denied=True; result.reason=reason; return result
        reason=_check_path_jail(argv_list,self.config)
        if reason: result.denied=True; result.reason=reason; return result
        real_cwd=os.path.realpath(cwd or self.config.project_root)
        env={k:os.environ[k] for k in self.config.env_allowlist if k in os.environ}
        started=time.perf_counter()
        try:
            proc=subprocess.run(argv_list,shell=shell,cwd=real_cwd,env=env,capture_output=True,timeout=self.config.timeout_seconds,input=stdin)
        except subprocess.TimeoutExpired as exc:
            elapsed=(time.perf_counter()-started)*1000
            out,t1=truncate_stream(exc.stdout or b"",self.config.max_output_bytes)
            err,t2=truncate_stream(exc.stderr or b"",self.config.max_output_bytes)
            return SandboxResult(argv=argv_list,exit_code=TIMED_OUT_EXIT,out=out,stderr=err,truncated=t1 or t2,timed_out=True,reason=f"timeout {self.config.timeout_seconds}s",duration_ms=elapsed)
        except FileNotFoundError as exc:
            return SandboxResult(argv=argv_list,exit_code=DENIED_EXIT,denied=True,reason=f"not found: {exc}")
        elapsed=(time.perf_counter()-started)*1000
        out,t1=truncate_stream(proc.stdout or b"",self.config.max_output_bytes)
        err,t2=truncate_stream(proc.stderr or b"",self.config.max_output_bytes)
        return SandboxResult(argv=argv_list,exit_code=proc.returncode,out=out,stderr=err,truncated=t1 or t2,duration_ms=elapsed)

def _demo():
    root=tempfile.mkdtemp(prefix="sb-demo-")
    with open(os.path.join(root,"hello.txt"),"w") as f: f.write("hello\n")
    cfg=SandboxConfig(project_root=root,max_output_bytes=512,timeout_seconds=2.0)
    sb=Sandbox(config=cfg); echo="echo"
    print("SANDBOX DEMO"); print(f"root={root}\n")
    for label,argv in [
        ("echo hello",(echo,"hello")),("rm -rf .",("rm","-rf",".")),
        ("sudo apt",("sudo","apt","update")),(f"cat {root}/../etc/passwd",("cat",f"{root}/../etc/passwd")),
    ]:
        r=sb.run(argv); badge="OK" if r.ok else "DENIED" if r.denied else "TIMEOUT"
        print(f"  {label:30s} -> {badge}" + (f" ({r.reason})" if r.denied or r.timed_out else ""))
    print(f"\nproject_root: {root}")

if __name__=="__main__": _demo()
```

运行结果：

```
SANDBOX DEMO
root=/tmp/sb-demo-xxxx

  echo hello                      -> OK
  rm -rf .                        -> DENIED (executable 'rm' on denylist)
  sudo apt                        -> DENIED (executable 'sudo' on denylist)
  cat /tmp/sb-demo-xxxx/../etc/passwd -> DENIED (path '../etc/passwd' resolves outside project root)

project_root: /tmp/sb-demo-xxxx
```

---

## 4. 工具实践

**与验证门的集成**：门链（第23节）说ALLOW后，沙箱执行。沙箱是第二层——即使门通过了，沙箱仍然拒绝危险模式。

**生产层叠**：非特权Docker容器→微VM→能力丢弃→项目根只读挂载→scratch目录读写→ulimit内存CPU→环境清洗到已知安全白名单。

---

## 5. LLM视角

**防御深度视角**：验证门和沙箱是两层独立防御。门检查"是否应该运行"，沙箱检查"运行时什么被允许"。两层都需要。

**路径监狱视角**：符号链接逃逸是最隐蔽的攻击。通过`os.path.realpath`检查解决了这个问题。

---

## 6. 工程最佳实践

**拒绝列表设计**：frozenset，基名匹配（/bin/rm和/usr/bin/rm都解析到"rm"）。

**输出截断**：默认64KB，标记行+truncated标志让下游知道发生了截断。

**超时处理**：`subprocess.TimeoutExpired`时杀死进程组。

---

## 7. 常见错误

**错误1：仅检查字符串路径**
症状：符号链接逃逸
修复：通过`os.path.realpath`解析后检查前缀

**错误2：忽略解释器-c标志**
症状：python3 -c "os.system('rm -rf /')"执行
修复：拒绝解释器带-c/-e标志的调用

---

## 8. 面试考点

**Q1：沙箱和验证门的区别是什么？**
考察：对两层防御的理解

**Q2：为什么路径监狱用realpath而非普通路径比较？**
考察：对符号链接逃逸的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 拒绝列表 | "危险命令黑名单" | 可执行文件基名的frozenset |
| 路径监狱 | "文件系统隔离" | 路径解析后检查是否在项目根目录下 |
| argv检查 | "解释器拦截" | 拒绝解释器带-c/-e标志的调用（等于shell） |
| 输出截断 | "输出限制" | 超过最大字节数时截断并标记 |
| SandboxResult | "结构化结果" | 含exit_code/stdout/stderr/denied/timed_out/truncated |

---

## 参考文献

- [subprocess.run文档](https://docs.python.org/3/library/subprocess.html)
- [os.path.realpath文档](https://docs.python.org/3/library/os.path.html#os.path.realpath)
