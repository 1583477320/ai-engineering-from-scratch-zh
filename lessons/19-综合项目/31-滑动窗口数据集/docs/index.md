# 综合项目31——滑动窗口数据集（下一个token预测训练数据管道）

> 预训练运行是从token ID到梯度的函数。本课程构建提供ID的传送带。将原始语料通过分词器编码为ID流，用滑动窗口切片为固定长度窗口，构建PyTorch Dataset返回输入和目标张量，包装为DataLoader带确定性洗牌。

**类型：** 构建
**编程语言：** Python（PyTorch）
**前置知识：** 第4章（计算机视觉）、第7章（Transformer）、第30节（BPE分词器）
**预计时间：** 90分钟

---

## 学习目标

- 将原始语料通过分词器编码为ID流
- 将ID流切片为固定长度窗口（可配置重叠步长）
- 构建返回输入和目标张量的PyTorch Dataset
- 包装DataLoader带每epoch确定性洗牌
- 理解步长、冗余和有效数据集大小的权衡

---

## 1. 问题

预训练运行一次读取一批token ID并更新模型。批次形状固定：(B,T)输入ID和(B,T)目标ID，目标是输入向左移一位。

数据管道的工作是按需从可能是几GB的原始语料中产生这个契约。本课程构建这个管道。

---

## 2. 核心概念

### 2.1 形状契约

因果LM消费(B,T)的ID。目标在位置t的值为输入在位置t+1的值。每个训练示例覆盖T+1个原始ID。

### 2.2 滑动窗口

如果模型只看到非重叠窗口，每个训练示例教它相同的T个边界。调整步长移动这些边界。

- stride=T：非重叠窗口
- stride=T//2：50%重叠，翻倍有效数据集
- stride=1：最大化重叠，增至T倍

### 2.3 确定性洗牌

通过传递显式的`torch.Generator`（每epoch播种base_seed+epoch_index），获得相同洗牌顺序。这使两个仅超参数不同的运行可比。

---

## 3. 从零实现

`code/main.py`实现SlidingWindowDataset、make_dataloader和演示。

```python
"""滑动窗口数据集——为下一个token预测训练准备的PyTorch数据集+DataLoader。

将BPE分词器编码的ID流切片为固定长度窗口，返回input+target对。
运行：python3 code/main.py
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self,ids,context_length,stride=None):
        self.ids=torch.tensor(ids,dtype=torch.long); self.ctx=context_length; self.stride=stride or context_length
    @staticmethod
    def count_windows(n,ctx,s):
        u=n-(ctx+1); return 0 if u<0 else 1+u//s
    def __len__(self): return self.count_windows(self.ids.numel(),self.ctx,self.stride)
    def __getitem__(self,i):
        s=i*self.stride; w=self.ids[s:s+self.ctx+1]; return w[:-1].clone(),w[1:].clone()

def make_dataloader(ds,bs,seed=0,epoch=0,shuffle=True):
    g=torch.Generator(); g.manual_seed(seed+epoch)
    return DataLoader(ds,batch_size=bs,shuffle=shuffle,drop_last=True,generator=g if shuffle else None)

def main():
    ids=list(range(1000)); ctx=16; stride=8
    ds=SlidingWindowDataset(ids,ctx,stride)
    print(f"窗口数: {len(ds)}  (stride={stride})")
    inp,tgt=ds[0]; print(f"输入: {tuple(inp.shape)} 目标: {tuple(tgt.shape)}")
    loader=make_dataloader(ds,4,seed=7,epoch=0)
    ins,tgs=next(iter(loader)); print(f"批次: {tuple(ins.shape)}")
    for s in (4,8,16): print(f"  stride={s:>2}: {len(SlidingWindowDataset(ids,ctx,s))}窗口")
    return 0

if __name__=="__main__": raise SystemExit(main())
```

---

## 4. 工具实践

**Dataset实现**：`__len__`返回示例数，`__getitem__`实时计算窗口起始位置，内存成本为一份ID流副本。

**计数公式**：对于长度为N的ID流、上下文长度T、步长S，示例数=max(0, 1+(N-(T+1))//S)。

---

## 5. LLM视角

**边界多样性视角**：滑动窗口的本质是教模型看到更多样的预测下一个token边界。stride=T时模型每次看到相同的T-1边界。

---

## 6. 工程最佳实践

**seed机制**：base_seed+epoch_index，使超参数搜索可比。

**num_workers**：本课程保持0，生产运行使用workers并行化。

---

## 7. 常见错误

**错误1：窗口与语料边界重叠**
症状：最后一个窗口不足T+1个ID
修复：丢弃不足的最后一个窗口

**错误2：不检查训练/测试污染**
症状：基线测试集泄漏到训练
修复：在token化前拆分并插入`<|endoftext|>`分隔符

---

## 8. 面试考点

**Q1：为什么需要滑动窗口而非非重叠窗口？**
考察：对边界多样性的理解

**Q2：确定性洗牌为什么重要？**
考察：对可复现性的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 滑动窗口 | "窗口切片" | 从ID流切出固定长度重叠窗口 |
| stride | "步长" | 相邻窗口起始位置的间距 |
| 移位目标 | "下一个token" | target=input向左移一位 |
| 确定性洗牌 | "种子洗牌" | base_seed+epoch产生可复现顺序 |
| count_windows | "窗口计数" | 1+(N-(T+1))//S |

---

## 参考文献

- [PyTorch Dataset文档](https://pytorch.org/docs/stable/data.html)
- [NanoGPT数据管道](https://github.com/karpathy/nanoGPT)
