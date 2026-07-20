"""任务规范格式——JSONL schema + 验证器。"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

VALID_METRICS={"exact_match","f1","bleu_4","rouge_l","accuracy","code_exec"}
VALID_CATS={"arithmetic","mcq","code_exec","classification","summary"}
VALID_POST={"none","strip_whitespace","lower","extract_letter","extract_code_block","extract_first_line"}
CAT_METRIC={"arithmetic":"exact_match","mcq":"exact_match","code_exec":"code_exec","classification":"accuracy","summary":"rouge_l"}

@dataclass
class TaskSpec:
    task_id:str; category:str; prompt:str; targets:List[str]; metric_name:str; post_process:str
    few_shot:List[Dict]=field(default_factory=list); metadata:Dict=field(default_factory=dict)
    def render(self):
        parts=[ex["prompt"]+" "+ex.get("completion","") for ex in self.few_shot]
        parts.append(self.prompt); return "\n\n".join(parts)

@dataclass
class ValidationError:
    line:int; fld:str; rule:str; raw:str

def validate_task(rec,line_num,seen):
    errs=[]
    for r in ["task_id","category","prompt","targets","metric_name","post_process"]:
        if r not in rec: errs.append(ValidationError(line_num,r,"缺少字段",str(rec)[:80])); return None,errs
    if not isinstance(rec["task_id"],str) or " " in rec["task_id"]: errs.append(ValidationError(line_num,"task_id","格式错误",rec["task_id"]))
    if rec["task_id"] in seen: errs.append(ValidationError(line_num,"task_id","重复",rec["task_id"]))
    if rec["category"] not in VALID_CATS: errs.append(ValidationError(line_num,"category","无效",rec["category"]))
    if rec["metric_name"] not in VALID_METRICS: errs.append(ValidationError(line_num,"metric_name","无效",rec["metric_name"]))
    if rec["post_process"] not in VALID_POST: errs.append(ValidationError(line_num,"post_process","无效",rec["post_process"]))
    if not isinstance(rec["targets"],list) or not rec["targets"]: errs.append(ValidationError(line_num,"targets","非空列表",str(rec["targets"])))
    expected=CAT_METRIC.get(rec.get("category"))
    if expected and rec["metric_name"]!=expected: errs.append(ValidationError(line_num,"metric_name",f"类别{rec['category']}需要{expected}",rec["metric_name"]))
    if errs: return None,errs
    seen.add(rec["task_id"])
    return TaskSpec(rec["task_id"],rec["category"],rec["prompt"],rec["targets"],rec["metric_name"],rec["post_process"],rec.get("few_shot_examples",[]),rec.get("metadata",{})),[]

def validate_file(text):
    specs,errs,seen=[],[],set()
    for i,line in enumerate(text.strip().split("\n"),1):
        if not line.strip(): continue
        try: rec=json.loads(line)
        except: errs.append(ValidationError(i,"_json","JSON错误",line[:80])); continue
        spec,e=validate_task(rec,i,seen); errs.extend(e)
        if spec: specs.append(spec)
    return specs,errs

FIXTURES=[
    {"task_id":"arith_001","category":"arithmetic","prompt":"Compute: 2+2\nAnswer:","targets":["4"],"metric_name":"exact_match","post_process":"strip_whitespace"},
    {"task_id":"arith_002","category":"arithmetic","prompt":"Compute: 7*6\nAnswer:","targets":["42"],"metric_name":"exact_match","post_process":"strip_whitespace"},
    {"task_id":"mcq_001","category":"mcq","prompt":"2+2=? A:3 B:4 C:5\nAnswer:","targets":["B"],"metric_name":"exact_match","post_process":"extract_letter"},
    {"task_id":"code_001","category":"code_exec","prompt":"Write f(x)=x*2","targets":["ok"],"metric_name":"code_exec","post_process":"extract_code_block"},
    {"task_id":"cls_001","category":"classification","prompt":"Positive? I love it","targets":["positive"],"metric_name":"accuracy","post_process":"lower"},
    {"task_id":"sum_001","category":"summary","prompt":"Summarize: Cat sat on mat.","targets":["cat mat"],"metric_name":"rouge_l","post_process":"strip_whitespace"},
]

def main():
    text="\n".join(json.dumps(f) for f in FIXTURES)
    specs,errs=validate_file(text)
    print(f"通过: {len(specs)} 条  错误: {len(errs)} 条")
    for e in errs: print(f"  行{e.line}: {e.fld} — {e.rule}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
