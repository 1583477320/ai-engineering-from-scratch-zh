"""滑动窗口数据集——为下一个token预测训练准备的PyTorch数据集+DataLoader。"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
import torch
from torch.utils.data import DataLoader, Dataset

BYTE_ALPHABET_SIZE=256; DEFAULT_SPECIALS=("<|endoftext|>","<|pad|>")
WORD_SPLIT_RE=re.compile(r"\S+|\s+")

@dataclass
class MiniBPE:
    vocab:dict[int,bytes]=field(default_factory=dict)
    inv_vocab:dict[bytes,int]=field(default_factory=dict)
    merges:dict[tuple[int,int],int]=field(default_factory=dict)
    special_to_id:dict[str,int]=field(default_factory=dict)
    id_to_special:dict[int,str]=field(default_factory=dict)
    @property
    def vocab_size(self): return len(self.vocab)
    def initialize(self,specials=DEFAULT_SPECIALS):
        self.vocab.clear(); self.inv_vocab.clear(); self.merges.clear(); self.special_to_id.clear(); self.id_to_special.clear()
        for i in range(BYTE_ALPHABET_SIZE): self.vocab[i]=bytes([i]); self.inv_vocab[bytes([i])]=i
        for s in specials: i=len(self.vocab); self.vocab[i]=s.encode(); self.inv_vocab[s.encode()]=i; self.special_to_id[s]=i; self.id_to_special[i]=s

def train_bpe(tok,corpus,target):
    tok.initialize(); chunks=WORD_SPLIT_RE.findall(corpus); units={}
    for c in chunks: syms=tuple(c.encode()); units[syms]=units.get(syms,0)+1
    while tok.vocab_size<target:
        pairs=Counter()
        for syms,c in units.items():
            for i in range(len(syms)-1): pairs[(syms[i],syms[i+1])]+=c
        if not pairs: break
        best=sorted(p for p,c in pairs.items() if c==max(pairs.values()))[0]
        if pairs[best]<2: break
        nid=len(tok.vocab); mb=tok.vocab[best[0]]+tok.vocab[best[1]]; tok.vocab[nid]=mb; tok.inv_vocab[mb]=nid; tok.merges[best]=nid
        nu={}
        for syms,c in units.items():
            out=[]; i=0; a,b=best
            while i<len(syms):
                if i<len(syms)-1 and syms[i]==a and syms[i+1]==b: out.append(nid); i+=2
                else: out.append(syms[i]); i+=1
            nu[tuple(out)]=nu.get(tuple(out),0)+c
        units=nu

def encode_text(tok,text):
    ranked={p:r for r,p in enumerate(tok.merges.keys())}; out=[]
    for c in WORD_SPLIT_RE.findall(text):
        syms=list(c.encode())
        while len(syms)>=2:
            br=None; bi=-1; bp=None
            for i in range(len(syms)-1):
                p=(syms[i],syms[i+1]); r=ranked.get(p)
                if r is None: continue
                if br is None or r<br: br=r; bi=i; bp=p
            if bp is None: break
            nid=tok.merges[bp]; syms=syms[:bi]+[nid]+syms[bi+2:]
        out.extend(syms)
    return out

class SlidingWindowDataset(Dataset):
    def __init__(self,ids,context_length,stride=None):
        if context_length<1: raise ValueError
        if not ids: raise ValueError
        if stride is None: stride=context_length
        self.ids=torch.tensor(ids,dtype=torch.long); self.context_length=context_length; self.stride=stride
    @staticmethod
    def count_windows(n,ctx,s):
        u=n-(ctx+1); return 0 if u<0 else 1+u//s
    def __len__(self): return self.count_windows(self.ids.numel(),self.context_length,self.stride)
    def __getitem__(self,idx):
        if idx<0: idx+=len(self)
        start=idx*self.stride; end=start+self.context_length+1; w=self.ids[start:end]
        return w[:-1].clone(),w[1:].clone()

def make_dataloader(dataset,batch_size,shuffle=True,base_seed=0,epoch=0,drop_last=True):
    g=torch.Generator(); g.manual_seed(base_seed+epoch)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,generator=g if shuffle else None)

CORPUS="the quick brown fox jumps over the lazy dog\na journey begins\nthe only way to do great work is to love what you do\n"*8

def main():
    tok=MiniBPE(); ids=encode_text(tok,CORPUS) if False else (train_bpe(tok,CORPUS,320) or encode_text(tok,CORPUS))
    train_bpe(tok,CORPUS,320); ids=encode_text(tok,CORPUS)
    print(f"词表: {tok.vocab_size}  ID数: {len(ids)}")
    ctx=16; stride=8; bs=4
    ds=SlidingWindowDataset(ids,ctx,stride)
    print(f"窗口数: {len(ds)}  (stride={stride})")
    inp,tgt=ds[0]; print(f"输入: {tuple(inp.shape)} 目标: {tuple(tgt.shape)}")
    loader=make_dataloader(ds,bs,base_seed=7,epoch=0)
    inputs,targets=next(iter(loader))
    print(f"批次: {tuple(inputs.shape)}")
    for s in (4,8,16): print(f"  stride={s:>2}: {len(SlidingWindowDataset(ids,ctx,s))}窗口")
    return 0

if __name__=="__main__": raise SystemExit(main())
