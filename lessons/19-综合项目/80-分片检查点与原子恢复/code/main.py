"""分片检查点与原子恢复。"""
import os,json,hashlib,tempfile,pickle
from dataclasses import dataclass
@dataclass
class ShardInfo: rank:int; path:str; sha256:str; offset:int=0; numel:int=0
@dataclass
class Manifest: world_size:int; step:int; shards:list; sv:int=1
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def atomic_write(data,path):
    dir=os.path.dirname(path) or "."
    tmp=tempfile.NamedTemporaryFile(delete=False,dir=dir,prefix=os.path.basename(path)+".")
    with tmp:
        if isinstance(data,bytes): tmp.write(data)
        else: tmp.write(data.encode())
    os.replace(tmp.name,path)
def save(sds,out_dir,step):
    os.makedirs(out_dir,exist_ok=True); shards=[]
    for rank,sd in sds.items():
        fname=f"rank{rank}.bin"; data=pickle.dumps(sd); atomic_write(data,os.path.join(out_dir,fname))
        shards.append(ShardInfo(rank,fname,sha256_file(os.path.join(out_dir,fname))))
    m=Manifest(len(sds),step,shards); atomic_write(json.dumps({"ws":m.world_size,"step":m.step,"shards":[{"r":s.rank,"p":s.path,"s":s.sha256} for s in shards]},indent=2),os.path.join(out_dir,"manifest.json"))
def main():
    sds={i:{"param":b"data"+str(i).encode()*50} for i in range(4)}
    save(sds,"/tmp/sharded_ckpt",10)
    print(f"保存4个分片+清单")
    for f in os.listdir("/tmp/sharded_ckpt"): print(f"  {f}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
