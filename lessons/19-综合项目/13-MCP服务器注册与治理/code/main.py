"""MCP服务器+注册表+OPA策略门控脚手架。"""
from __future__ import annotations

import json, re, time
from dataclasses import asdict, dataclass, field
from typing import Callable

@dataclass
class ToolSchema:
    name: str; required_scope: str; destructive: bool; description: str; input_schema: dict

Handler = Callable[[dict], dict]

@dataclass
class MCPServer:
    name: str; url: str; tools: dict[str, ToolSchema] = field(default_factory=dict); handlers: dict[str, Handler] = field(default_factory=dict)
    def register(self, s: ToolSchema, h: Handler): self.tools[s.name]=s; self.handlers[s.name]=h
    def capabilities(self):
        return {"server":self.name,"transport":"streamable_http","url":self.url,
                "tools":[{"name":t.name,"scope":t.required_scope,"destructive":t.destructive,"description":t.description} for t in self.tools.values()]}

@dataclass
class Token:
    user: str; scopes: set[str]; approved_at: float = 0.0
    def has_scope(self, s): return s in self.scopes
    def fresh_approval(self, now, window_s=900): return "approved:by:human" in self.scopes and (now-self.approved_at)<=window_s

def policy_decide(server, tool, token, args, now):
    if tool not in server.tools: return False, f"no such tool: {tool}"
    s = server.tools[tool]
    if not token.has_scope(s.required_scope): return False, f"missing scope: {s.required_scope}"
    if s.destructive and not token.fresh_approval(now): return False, "destructive: needs fresh human approval"
    return True, "ok"

def redact(p):
    s=json.dumps(p); s=re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+","[email]",s); s=re.sub(r"\b\d{3}-\d{2}-\d{4}\b","[ssn]",s); return json.loads(s)

@dataclass
class AuditEntry:
    ts: float; user: str; tool: str; outcome: str; args_redacted: dict; response_redacted: dict

def dispatch(server, token, tool, args, audit):
    now=time.time(); ok,reason=policy_decide(server,tool,token,args,now)
    if not ok: audit.append(AuditEntry(now,token.user,tool,f"denied:{reason}",redact(args),{})); return {"error":{"code":403,"message":reason}}
    result=server.handlers[tool](args); audit.append(AuditEntry(now,token.user,tool,"ok",redact(args),redact(result))); return {"result":result}

@dataclass
class Registry:
    entries: dict[str,dict]=field(default_factory=dict)
    def register(self, server): self.entries[server.name]=server.capabilities()
    def search(self, q):
        q=q.lower(); return [(sn,t["name"]) for sn,cap in self.entries.items() for t in cap["tools"] if q in t["name"].lower() or q in t["description"].lower()]

def main():
    ro=MCPServer("readonly-mcp","https://mcp.internal/readonly")
    ro.register(ToolSchema("postgres.readonly","postgres:query:readonly",False,"Read-only Postgres query",{"type":"object","properties":{"sql":{"type":"string"}}}),lambda a:{"rows":[[1]]})
    ro.register(ToolSchema("s3.list","s3:list",False,"List S3 objects",{"type":"object","properties":{"bucket":{"type":"string"}}}),lambda a:{"objects":[{"key":"a/b.txt","size":128}]})
    rw=MCPServer("destructive-mcp","https://mcp.internal/destructive")
    rw.register(ToolSchema("jira.create","jira:write",True,"Create Jira issue",{"type":"object","properties":{"title":{"type":"string"}}}),lambda a:{"id":"PROJ-99","created":True})
    registry=Registry(); registry.register(ro); registry.register(rw); audit=[]
    ro_token=Token("u42",{"postgres:query:readonly","s3:list","jira:read"})
    ap_token=Token("u42",{"jira:write","approved:by:human"},approved_at=time.time()-60)
    print("=== 注册表搜索 ==="); print("  jira ->",registry.search("jira"))
    print("\n=== postgres.readonly ==="); print(" ",dispatch(ro,ro_token,"postgres.readonly",{"sql":"SELECT 1"},audit))
    print("\n=== jira.create (无审批) ==="); print(" ",dispatch(rw,Token("u42",{"jira:write"}),"jira.create",{"title":"bug"},audit))
    print("\n=== jira.create (有审批) ==="); print(" ",dispatch(rw,ap_token,"jira.create",{"title":"bug"},audit))
    print("\n=== 审计日志 ===")
    for e in audit: print(" ",json.dumps(asdict(e),default=str))

if __name__=="__main__": main()
