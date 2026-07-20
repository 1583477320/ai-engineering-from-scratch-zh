# 指令微调SFT技能

## 目标
用掩码指令token实现Alpaca风格SFT，用精确匹配评测。

## 构建检查清单

- [ ] InstructionTokenizer（INST/RESP/PAD特殊token）
- [ ] TinyGPT（decoder-only transformer）
- [ ] SFTDataset + sft_collate（-100掩码）
- [ ] shifted_loss（标准因果LM损失）
- [ ] generate（贪心/温度采样+停止启发式）
- [ ] exact_match评测
- [ ] 200对6类任务数据集
- [ ] 每5轮eval打印精确匹配
