"""工具注册表+JSON Schema 2020-12子集验证。"""
from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Any, Callable

PRIMITIVE_TYPE_MAP = {"string":(str,),"integer":(int,),"number":(int,float),"boolean":(bool,),"object":(dict,),"array":(list,),"null":(type(None),)}
ALLOWED_KEYWORDS = {"type","properties","required","enum","minLength","maxLength","pattern","items","description"}

@dataclass
class ValidationError:
    path:str; keyword:str; message:str
    def to_dict(self): return {"path":self.path,"keyword":self.keyword,"message":self.message}

@dataclass
class Ok: pass

@dataclass
class ToolRecord:
    name:str; description:str; schema:dict; handler:Callable[...,Any]; idempotent:bool=False; timeout_ms:int=30_000

class ToolRegistry:
    _NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
    def __init__(self): self._records={}; self._order=[]
    def register(self,name,schema,handler,description="",idempotent=False,timeout_ms=30_000,override=False):
        if not self._NAME_RE.match(name): raise ValueError(f"name {name!r} invalid")
        if name in self._records and not override: raise ValueError(f"{name!r} already registered")
        validate_schema_shape(schema)
        rec=ToolRecord(name=name,description=description,schema=schema,handler=handler,idempotent=idempotent,timeout_ms=timeout_ms)
        if name not in self._records: self._order.append(name)
        self._records[name]=rec; return rec
    def get(self,name):
        if name not in self._records: raise KeyError(f"unknown tool {name!r}")
        return self._records[name]
    def names(self): return list(self._order)
    def validate(self,name,args):
        rec=self.get(name); errs=[]; _walk(rec.schema,args,"",errs)
        return Ok() if not errs else errs

def validate_schema_shape(schema):
    if not isinstance(schema,dict): raise ValueError("schema must be a dict")
    unknown=set(schema.keys())-ALLOWED_KEYWORDS
    if unknown: raise ValueError(f"unsupported keywords: {sorted(unknown)}")
    t=schema.get("type")
    if t is not None and t not in PRIMITIVE_TYPE_MAP: raise ValueError(f"unsupported type: {t!r}")
    for k in ("properties","items"):
        s=schema.get(k)
        if s is not None and isinstance(s,dict): validate_schema_shape(s)

def _path(p,s): return f"{p}/{str(s).replace('~','~0').replace('/','~1')}"

def _type_matches(v,e):
    if e=="boolean": return isinstance(v,bool)
    if e in ("integer","number"): return not isinstance(v,bool) and isinstance(v,PRIMITIVE_TYPE_MAP[e])
    return isinstance(v,PRIMITIVE_TYPE_MAP[e])

def _walk(schema,value,path,errs):
    t=schema.get("type")
    if t is not None and not _type_matches(value,t):
        errs.append(ValidationError(path or "/","type",f"expected {t}, got {type(value).__name__}")); return
    if "enum" in schema and value not in schema["enum"]:
        errs.append(ValidationError(path or "/","enum",f"{value!r} not in {schema['enum']!r}")); return
    if t=="string":
        if "minLength" in schema and len(value)<schema["minLength"]: errs.append(ValidationError(path,"minLength",f"length {len(value)}<{schema['minLength']}"))
        if "maxLength" in schema and len(value)>schema["maxLength"]: errs.append(ValidationError(path,"maxLength",f"length {len(value)}>{schema['maxLength']}"))
        if "pattern" in schema and not re.search(schema["pattern"],value): errs.append(ValidationError(path,"pattern",f"no match"))
    elif t=="object":
        for r in schema.get("required",[]):
            if r not in value: errs.append(ValidationError(_path(path,r),"required",f"missing {r!r}"))
        for pn,pv in value.items():
            if pn in schema.get("properties",{}): _walk(schema["properties"][pn],pv,_path(path,pn),errs)
    elif t=="array":
        it=schema.get("items")
        if it:
            for i,item in enumerate(value): _walk(it,item,_path(path,i),errs)

def _demo():
    reg=ToolRegistry()
    reg.register("db.get_user",{"type":"object","required":["id"],"properties":{"id":{"type":"integer"}}},lambda id:{"id":id},description="Fetch user")
    cases=[{"id":42},{"id":"bad"},{},{"id":1}]
    for c in cases:
        r=reg.validate("db.get_user",c)
        s="OK" if isinstance(r,Ok) else f"ERRORS:{[e.to_dict() for e in r]}"
        print(f"  {c} -> {s}")
    print(f"\nTools: {reg.names()}")

if __name__=="__main__": _demo()
