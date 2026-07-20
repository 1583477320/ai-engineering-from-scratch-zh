"""字节级BPE分词器从零实现。"""
from __future__ import annotations
import json, re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

BYTE_ALPHABET_SIZE=256; DEFAULT_SPECIALS=("<|endoftext|>","<|pad|>")
WORD_SPLIT_RE=re.compile(r"\S+|\s+")

@dataclass
class BPETokenizer:
    vocab:dict[int,bytes]=field(default_factory=dict)
    inv_vocab:dict[bytes,int]=field(default_factory=dict)
    merges:dict[tuple[int,int],int]=field(default_factory=dict)
    special_to_id:dict[str,int]=field(default_factory=dict)
    id_to_special:dict[int,str]=field(default_factory=dict)
    @property
    def vocab_size(self): return len(self.vocab)
    def _add_token(self,tb):
        if tb in self.inv_vocab: return self.inv_vocab[tb]
        i=len(self.vocab); self.vocab[i]=tb; self.inv_vocab[tb]=i; return i
    def initialize(self,specials=DEFAULT_SPECIALS):
        self.vocab.clear(); self.inv_vocab.clear(); self.merges.clear(); self.special_to_id.clear(); self.id_to_special.clear()
        for i in range(BYTE_ALPHABET_SIZE): self._add_token(bytes([i]))
        for s in specials: i=self._add_token(s.encode("utf-8")); self.special_to_id[s]=i; self.id_to_special[i]=s

def _pretokenize(text): return WORD_SPLIT_RE.findall(text)
def _count_pairs(units):
    p=Counter()
    for syms,c in units.items():
        for i in range(len(syms)-1): p[(syms[i],syms[i+1])]+=c
    return p

def _apply_merge(syms,pair,new_id):
    if len(syms)<2: return syms
    out=[]; i=0; a,b=pair
    while i<len(syms):
        if i<len(syms)-1 and syms[i]==a and syms[i+1]==b: out.append(new_id); i+=2
        else: out.append(syms[i]); i+=1
    return tuple(out)

def train(tokenizer,corpus,target_size,specials=DEFAULT_SPECIALS):
    tokenizer.initialize(specials)
    chunks=_pretokenize(corpus); units={}
    for c in chunks: syms=tuple(c.encode("utf-8")); units[syms]=units.get(syms,0)+1
    while tokenizer.vocab_size<target_size:
        pairs=_count_pairs(units)
        if not pairs: break
        mc=max(pairs.values()); best=sorted(p for p,c in pairs.items() if c==mc)[0]
        if pairs[best]<2: break
        new_id=tokenizer._add_token(tokenizer.vocab[best[0]]+tokenizer.vocab[best[1]])
        tokenizer.merges[best]=new_id
        nu={}; old_units=units
        for syms,c in old_units.items(): m=_apply_merge(syms,best,new_id); nu[m]=nu.get(m,0)+c
        units=nu

def encode(tokenizer,text,allow_special=False):
    ranked={p:r for r,p in enumerate(tokenizer.merges.keys())}; out=[]
    for chunk in _pretokenize(text):
        syms=list(chunk.encode("utf-8"))
        while len(syms)>=2:
            br=None; bi=-1; bp=None
            for i in range(len(syms)-1):
                p=(syms[i],syms[i+1]); r=ranked.get(p)
                if r is None: continue
                if br is None or r<br: br=r; bi=i; bp=p
            if bp is None: break
            new_id=tokenizer.merges[bp]; syms=syms[:bi]+[new_id]+syms[bi+2:]
        out.extend(syms)
    return out

def decode(tokenizer,ids):
    pieces=[]
    for i in ids:
        if i in tokenizer.id_to_special: pieces.append(tokenizer.id_to_special[i].encode())
        else: pieces.append(tokenizer.vocab[i])
    return b"".join(pieces).decode("utf-8",errors="replace")

DEMO="""the quick brown fox jumps over the lazy dog
a journey of a thousand miles begins with a single step
the only way to do great work is to love what you do"""*6

def main():
    tok=BPETokenizer(); train(tok,DEMO,320)
    print(f"词表大小: {tok.vocab_size}  合并数: {len(tok.merges)}")
    held="the fox is quick and the dog is lazy"; ids=encode(tok,held)
    print(f"输入: {held!r}")
    print(f"编码: {ids}  ({len(ids)} tokens vs {len(held.encode('utf-8'))} bytes)")
    print(f"解码: {decode(tok,ids)!r}")
    assert decode(tok,ids)==held
    return 0

if __name__=="__main__": raise SystemExit(main())
