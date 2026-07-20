"""沙箱运行器——拒绝列表+路径监狱+超时+截断。"""
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
    return f"executable {name!r} on denylist" if name in cfg.denylist else None

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
    for label,argv in [("echo hello",(echo,"hello")),("rm -rf .",("rm","-rf",".")),
        ("sudo apt",("sudo","apt","update")),(f"cat {root}/../etc/passwd",("cat",f"{root}/../etc/passwd"))]:
        r=sb.run(argv); badge="OK" if r.ok else "DENIED" if r.denied else "TIMEOUT"
        print(f"  {label:40s} -> {badge}" + (f" ({r.reason})" if r.denied or r.timed_out else ""))

if __name__=="__main__": _demo()
