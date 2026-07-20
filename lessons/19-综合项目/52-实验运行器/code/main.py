"""实验运行器——子进程隔离+超时+消融。"""
from __future__ import annotations
import json, os, subprocess, sys, time
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class ExperimentSpec:
    spec_id: str; hypothesis_id: int; script_path: str
    config: Dict[str,Any] = field(default_factory=dict)
    seed: int = 0; wall_timeout_s: float = 60.0
    metric_keys: List[str] = field(default_factory=list)

@dataclass
class ExperimentResult:
    spec_id: str; hypothesis_id: int; exit_code: int; terminal: str
    wall_time_s: float; metrics: Dict[str,Any] = field(default_factory=dict)

class ExperimentRunner:
    def run(self, spec):
        cfg={**spec.config,"__seed":spec.seed}
        path=f"/tmp/{spec.spec_id}_cfg.json"
        with open(path,"w") as f: json.dump(cfg,f)
        start=time.perf_counter()
        proc=subprocess.Popen([sys.executable,spec.script_path,path],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try: stdout,_=proc.communicate(timeout=spec.wall_timeout_s)
        except subprocess.TimeoutExpired: proc.kill(); stdout,_=proc.communicate(); terminal="timeout"
        else: terminal="ok"
        wall=time.perf_counter()-start
        metrics={}
        for line in reversed(stdout.strip().split("\n")):
            try:
                p=json.loads(line)
                if all(k in p for k in spec.metric_keys): metrics=p; break
            except: pass
        if terminal=="ok" and proc.returncode: terminal="crash"
        return ExperimentResult(spec.spec_id,spec.hypothesis_id,proc.returncode or 0,terminal,wall,metrics)


def ablate(base, knob, values):
    return [ExperimentSpec(f"{base.spec_id}_{knob}_{v}",base.hypothesis_id,base.script_path,
                           {**base.config,knob:v},base.seed,base.wall_timeout_s,base.metric_keys)
            for v in values]


def main():
    import tempfile
    s=tempfile.NamedTemporaryFile(mode="w",suffix=".py",delete=False)
    s.write("import json,sys,time; cfg=json.load(open(sys.argv[1]))\ntime.sleep(cfg.get('sleep_s',0.01))\nprint(json.dumps({'loss':0.5}))")
    s.close()
    spec=ExperimentSpec("test",1,s.name,{"sleep_s":0.01},42,10,metric_keys=["loss"])
    r=ExperimentRunner().run(spec)
    print(f"终端: {r.terminal} 指标: {r.metrics} 时间: {r.wall_time_s:.3f}s")
    os.unlink(s.name); return 0

if __name__=="__main__": raise SystemExit(main())
