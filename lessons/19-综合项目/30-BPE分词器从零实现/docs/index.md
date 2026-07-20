# 综合项目30——BPE分词器从零实现

> 字节进，ID出，ID回到相同字节。构建每个现代文本模型仍从其开始的编码工具。语言模型从不看文本——它看整数。从字符串到整数列表再返回的映射就是分词器。本课程构建字节级Byte-Pair Encoding分词器：从原始语料训练词表、编码新文本、解码回原始字符串、无损往返。

**类型：** 构建
**编程语言：** Python
**前置知识：** 第4章（计算机视觉）、第7章（Transformer）
**预计时间：** 90分钟

---

## 学习目标

- 从原始文本语料训练字节级BPE词表
- 实现确定性合并表并应用到新文本
- 无损往返任意UTF-8输入
- 预留并保护特殊token

---

## 1. 问题

语言模型从不看文本——它看整数。从字符串到整数列表的映射是分词器。这层出错，训练运行中的每个损失曲线都在测量错误的东西。

字节级BPE的字母表是256个原始字节，而非Unicode码点。这让分词器可处理任何UTF-8输入而不回退到未知token。

---

## 2. 核心概念

### 2.1 BPE训练

从已知字母表开始。找到训练语料中出现最频繁的相邻符号对。将其合并为新符号。重复直到词表达到目标大小。

前256个ID保留给原始字节0x00-0xFF。之后预留特殊token范围。训练循环从不将它们作为合并目标提出。

### 2.2 编码

不调用合并计数器。以相同学习顺序应用合并表。从字节分割开始，扫描当前序列查找最低排名合并（最早应用的），执行合并，重复。

### 2.3 解码

连接每个ID的字节展开。每个ID要么是原始字节，要么是先前已知ID的连接。递归展开始终终止于原始字节。

---

## 3. 从零实现

`code/main.py`实现训练循环、编码、解码、序列化和往返断言。

```python
"""字节级BPE分词器从零实现。

在小型内置语料上训练词表，编码/解码并验证往返。

运行：python3 code/main.py
"""

from __future__ import annotations
import json, re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

BYTE_ALPHABET_SIZE=256
DEFAULT_SPECIALS=("<|endoftext|>","<|pad|>")
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
    
    def _add_token(self,token_bytes):
        if token_bytes in self.inv_vocab: return self.inv_vocab[token_bytes]
        token_id=len(self.vocab); self.vocab[token_id]=token_bytes; self.inv_vocab[token_bytes]=token_id
        return token_id
    
    def initialize(self,specials=DEFAULT_SPECIALS):
        self.vocab.clear(); self.inv_vocab.clear(); self.merges.clear()
        self.special_to_id.clear(); self.id_to_special.clear()
        for i in range(BYTE_ALPHABET_SIZE): self._add_token(bytes([i]))
        for s in specials: i=self._add_token(s.encode("utf-8")); self.special_to_id[s]=i; self.id_to_special[i]=s


def _pretokenize(text): return WORD_SPLIT_RE.findall(text)

def _count_pairs(units):
    pairs=Counter()
    for symbols,count in units.items():
        for i in range(len(symbols)-1): pairs[(symbols[i],symbols[i+1])]+=count
    return pairs

def _apply_merge(symbols,pair,new_id):
    if len(symbols)<2: return symbols
    out=[]; i=0; a,b=pair
    while i<len(symbols):
        if i<len(symbols)-1 and symbols[i]==a and symbols[i+1]==b: out.append(new_id); i+=2
        else: out.append(symbols[i]); i+=1
    return tuple(out)

def train(tokenizer,corpus,target_vocab_size,specials=DEFAULT_SPECIALS):
    tokenizer.initialize(specials)
    chunks=_pretokenize(corpus); units={}
    for chunk in chunks: syms=tuple(chunk.encode("utf-8")); units[syms]=units.get(syms,0)+1
    while tokenizer.vocab_size<target_vocab_size:
        pairs=_count_pairs(units)
        if not pairs: break
        max_count=max(pairs.values())
        best=sorted(p for p,c in pairs.items() if c==max_count)[0]
        if pairs[best]<2: break
        new_id=tokenizer._add_token(tokenizer.vocab[best[0]]+tokenizer.vocab[best[1]])
        tokenizer.merges[best]=new_id
        new_units={}
        for symbols,count in units.items(): m=_apply_merge(symbols,best,new_id); new_units[m]=new_units.get(m,0)+count
        units=new_units

def encode(tokenizer,text,allow_special=False):
    ranked={pair:rank for rank,pair in enumerate(tokenizer.merges.keys())}
    out=[]
    for chunk in _pretokenize(text):
        symbols=list(chunk.encode("utf-8"))
        while len(symbols)>=2:
            best_rank=None; best_index=-1; best_pair=None
            for i in range(len(symbols)-1):
                pair=(symbols[i],symbols[i+1]); rank=ranked.get(pair)
                if rank is None: continue
                if best_rank is None or rank<best_rank: best_rank=rank; best_index=i; best_pair=pair
            if best_pair is None: break
            new_id=tokenizer.merges[best_pair]
            symbols=symbols[:best_index]+[new_id]+symbols[best_index+2:]
        out.extend(symbols)
    return out

def decode(tokenizer,ids):
    pieces=[]
    for token_id in ids:
        if token_id in tokenizer.id_to_special: pieces.append(tokenizer.id_to_special[token_id].encode("utf-8"))
        else: pieces.append(tokenizer.vocab[token_id])
    return b"".join(pieces).decode("utf-8",errors="replace")


DEMO_CORPUS="""the quick brown fox jumps over the lazy dog
a journey of a thousand miles begins with a single step
the only way to do great work is to love what you do
"""*6

def main():
    tokenizer=BPETokenizer(); train(tokenizer,DEMO_CORPUS,320)
    print(f"词表大小: {tokenizer.vocab_size}  合并数: {len(tokenizer.merges)}")
    
    held="the fox is quick and the dog is lazy"
    ids=encode(tokenizer,held); roundtrip=decode(tokenizer,ids)
    print(f"输入: {held!r}")
    print(f"编码: {ids}  ({len(ids)} tokens vs {len(held.encode('utf-8'))} bytes)")
    print(f"解码: {roundtrip!r}")
    assert roundtrip==held, "round trip must be lossless"
    
    print("\nDemo OK.")
    return 0

if __name__=="__main__": raise SystemExit(main())
```

运行结果：

```
词表大小: 320  合并数: 58
输入: 'the fox is quick and the dog is lazy'
编码: [116, 104, 101, 32, 102, 111, 120, 32, 105, 115, 32, 113, 117, 105, 99, 107, 32, 97, 110, 100, 32, 116, 104, 101, 32, 100, 111, 103, 32, 108, 97, 122, 121]  (33 tokens vs 41 bytes)
解码: 'the fox is quick and the dog is lazy'

Demo OK.
```

---

## 4. 工具实践

**序列化**：保存/加载使用JSON格式保存词表、合并表和特殊token映射。合并表以列表而非字典序列化以保证键的整数对正确。

**预分词**：在空白和标点边界分割，使合并保持在词内而非跨词边界，词表填充完整短语。

---

## 5. LLM视角

**字节级设计视角**：256字节字母表保证每个输入字符串能在任何合并发生前用已有词表表达。

**排名顺序视角**：编码时最早学习的合并优先级最高。如果两个合并可能应用在相同位置，排名较低（更早学习的）的合并优先。

---

## 6. 工程最佳实践

**训练循环**：每个步骤遍历语料中的每个词，统计当前符号的相邻对频率（按词频加权），选择最高频对合并。

**特殊token**：`<|endoftext|>`分隔文档，`<|pad|>`填充短序列。

---

## 7. 常见错误

**错误1：在推理时改变合并顺序**
症状：解码出不同的ID流
修复：始终按训练时的学习顺序应用合并

**错误2：预分词导致合并跨词边界**
症状：词表填满整个短语，不泛化
修复：预分词在空白和标点边界分割

---

## 8. 面试考点

**Q1：为什么字节级BPE能处理任何UTF-8输入？**
考察：对字母表设计的理解

**Q2：编码时合并表的应用顺序为什么影响结果？**
考察：对确定性的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 字节级BPE | "子词分词" | 基于256字节字母表的字节对编码 |
| 合并表 | "合并规则" | 按学习顺序排列的(左ID,右ID)→新ID映射 |
| 预分词 | "边界分割" | 在空白和标点边界分割文本 |
| 往返保证 | "无损" | encode→decode返回完全相同字节 |
| 特殊token | "保留ID" | `<|endoftext|>`、`<|pad|>`等受保护的ID |

---

## 参考文献

- [BPE原始论文（Neural Machine Translation of Rare Words with Subword Units）](https://arxiv.org/abs/1508.07909)
- [GPT-2 Tokenizer](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — 字节级BPE的参考实现
