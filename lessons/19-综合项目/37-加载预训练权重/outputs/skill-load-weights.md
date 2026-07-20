# 加载预训练权重技能

## 目标
从safetensors文件加载预训练GPT-2风格权重到本地GPT架构。

## 构建检查清单

- [ ] safetensors读取和张量名称检查
- [ ] 名称映射（wte→tok_embed, h.N→blocks.N等）
- [ ] 形状验证（不匹配时拒绝赋值）
- [ ] conv1d转置（c_attn/c_proj/c_fc权重）
- [ ] 权重绑定（lm_head=tok_embed）
- [ ] LoadReport（loaded/missing/unexpected/shape_mismatch）
- [ ] 加载后生成验证
