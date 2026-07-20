"""分块策略对比——固定窗口+句子+递归+语义。"""
import re

def fixed_window(text,size=500,overlap=50):
    c=[]; i=0
    while i<len(text): c.append(text[i:i+size]); i+=size-overlap
    return c

def sentence_chunks(text,target=500):
    sents=re.split(r'(?<=[.!?])\s+',text); chunks=[]; cur=[]; cl=0
    for s in sents:
        if cl+len(s)>target and cur: chunks.append(" ".join(cur)); cur=[]; cl=0
        cur.append(s); cl+=len(s)
    if cur: chunks.append(" ".join(cur))
    return chunks

def recursive_split(text,seps=None,target=500):
    if not seps: seps=["\n\n","\n"," ",""]
    if len(text)<=target: return [text]
    if not seps: return [text[i:i+target] for i in range(0,len(text),target)]
    sep=seps[0]; parts=text.split(sep); chunks=[]; cur=[]; cl=0
    for p in parts:
        if cl+len(p)+len(sep)>target and cur: chunks.append(sep.join(cur)); cur=[]; cl=0
        cur.append(p); cl+=len(p)+len(sep)
    if cur: chunks.append(sep.join(cur))
    if len(chunks)==1 and len(chunks[0])>target: return recursive_split(chunks[0],seps[1:],target)
    return chunks

def main():
    doc="注意力稀疏性是提高Transformer效率的关键方向。Top-k路由只保留最显著的注意力头。实验表明稀疏注意力匹配密集注意力。"
    for name,fn in [("固定窗口",lambda t:fixed_window(t,20)),("句子",lambda t:sentence_chunks(t,20)),("递归",lambda t:recursive_split(t,["。"," ",""],20))]:
        print(f"{name}: {len(fn(doc))}块 {fn(doc)}")
    return 0
if __name__=="__main__": import sys; sys.exit(main())
