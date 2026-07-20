"""JSON-RPC 2.0换行分隔stdio传输。"""
from __future__ import annotations
import io, json
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

ERR_PARSE=-32700; ERR_INVALID_REQUEST=-32600; ERR_METHOD_NOT_FOUND=-32601; ERR_INVALID_PARAMS=-32602; ERR_INTERNAL=-32603

class JsonRpcError(Exception):
    code=ERR_INTERNAL
    def __init__(self,message,data=None): super().__init__(message); self.message=message; self.data=data
class MethodNotFound(JsonRpcError): code=ERR_METHOD_NOT_FOUND
class InvalidParams(JsonRpcError): code=ERR_INVALID_PARAMS

@dataclass
class Request:
    method:str; params:Any; id:int|str|None; is_notification:bool

def _is_valid_envelope(msg):
    if not isinstance(msg,dict) or msg.get("jsonrpc")!="2.0": return False
    if not isinstance(msg.get("method"),str): return False
    if "id" in msg and isinstance(msg["id"],bool): return False
    return True

def parse_request(raw):
    try: msg=json.loads(raw)
    except json.JSONDecodeError as exc: return None,_err(None,ERR_PARSE,str(exc))
    if not _is_valid_envelope(msg): return None,_err(msg.get("id") if isinstance(msg,dict) else None,ERR_INVALID_REQUEST,"invalid envelope")
    return Request(method=msg["method"],params=msg.get("params"),id=msg.get("id"),is_notification="id" not in msg),None

def _err(rid,code,message,data=None):
    e={"code":code,"message":message}
    if data is not None: e["data"]=data
    return {"jsonrpc":"2.0","id":rid,"error":e}

def _ok(rid,result): return {"jsonrpc":"2.0","id":rid,"result":result}

class StdioTransport:
    def __init__(self,stdin:BinaryIO,stdout:BinaryIO): self._in=stdin; self._out=stdout
    def read_line(self): line=self._in.readline(); return line if line else None
    def write_response(self,rid,result): self._write(_ok(rid,result))
    def write_error(self,rid,code,message,data=None): self._write(_err(rid,code,message,data))
    def write_notification(self,method,params=None):
        env={"jsonrpc":"2.0","method":method}
        if params is not None: env["params"]=params
        self._write(env)
    def _write(self,obj):
        self._out.write((json.dumps(obj,separators=(",",":"))+"\n").encode("utf-8"))
        self._out.flush()

Handler=Callable[[str,Any],Any]

def _handle_one(handler,req):
    try: result=handler(req.method,req.params)
    except (MethodNotFound,InvalidParams,JsonRpcError) as exc:
        return None if req.is_notification else _err(req.id,exc.code,exc.message,exc.data)
    except Exception as exc:
        return None if req.is_notification else _err(req.id,ERR_INTERNAL,"internal",{"exception":type(exc).__name__})
    return None if req.is_notification else _ok(req.id,result)

def serve(handler,transport):
    while True:
        line=transport.read_line()
        if line is None: return
        text=line.decode("utf-8").rstrip("\n").strip()
        if not text: continue
        try: parsed=json.loads(text)
        except json.JSONDecodeError as exc: transport.write_error(None,ERR_PARSE,str(exc)); continue
        if isinstance(parsed,list):
            if not parsed: transport.write_error(None,ERR_INVALID_REQUEST,"empty batch"); continue
            out=[]
            for raw in parsed:
                if not _is_valid_envelope(raw): out.append(_err(raw.get("id") if isinstance(raw,dict) else None,ERR_INVALID_REQUEST,"invalid envelope")); continue
                req=Request(method=raw["method"],params=raw.get("params"),id=raw.get("id"),is_notification="id" not in raw)
                resp=_handle_one(handler,req)
                if resp is not None: out.append(resp)
            if out: transport._out.write((json.dumps(out,separators=(",",":"))+"\n").encode("utf-8")); transport._out.flush()
        else:
            req,err=parse_request(text)
            if err is not None: transport._write(err); continue
            resp=_handle_one(handler,req)
            if resp is not None: transport._write(resp)

def _demo():
    def handler(method,params):
        if method=="math.add":
            if not isinstance(params,dict) or "a" not in params or "b" not in params: raise InvalidParams("a and b required")
            return params["a"]+params["b"]
        if method=="echo": return params
        raise MethodNotFound(f"method {method!r}")
    requests=[
        {"jsonrpc":"2.0","id":1,"method":"math.add","params":{"a":2,"b":3}},
        {"jsonrpc":"2.0","id":2,"method":"math.add","params":{"a":5}},
        {"jsonrpc":"2.0","id":3,"method":"missing"},
        {"jsonrpc":"2.0","method":"log","params":{"level":"info"}},
        [{"jsonrpc":"2.0","id":10,"method":"echo","params":{"text":"hi"}},
         {"jsonrpc":"2.0","method":"log","params":{"msg":"skip"}},
         {"jsonrpc":"2.0","id":11,"method":"math.add","params":{"a":1,"b":1}}],
    ]
    stdin=io.BytesIO()
    for r in requests: stdin.write((json.dumps(r)+"\n").encode("utf-8"))
    stdin.write(b"{not json\n"); stdin.seek(0)
    stdout=io.BytesIO()
    serve(handler,StdioTransport(stdin,stdout))
    stdout.seek(0)
    lines=[json.loads(line) for line in stdout.read().decode("utf-8").splitlines() if line]
    print(json.dumps({"responses":lines},indent=2))

if __name__=="__main__": _demo()
