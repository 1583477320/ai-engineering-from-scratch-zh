# BPE分词器技能

## 目标
训练字节级BPE词表、编码/解码新文本、实现无损往返。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | BPE训练正确 | 产生确定合并表和最终词表 |
| 20 | 编码/解码变换取证 | 任意UTF-8输入encoded再decode后与原字符串相同 |
| 20 | 词表验证 | 256字节基础+特殊token+学习到的合并与目标大小匹配 |
| 20 | 特殊token处理 | <|endoftext|>和<|pad|>在编码/解码中正确保留 |
| 15 | 压缩效果 | 目标词表越大的情况下编码后的token数量越少 |

## 构建检查清单

- [ ] BPETokenizer类（vocab/inv_vocab/merges/special_to_id/id_to_special）
- [ ] initialize（256字节字母表+特殊token）
- [ ] train训练循环（统计相邻对频率、合并）
- [ ] encode编码（按排名应用合并表）
- [ ] decode解码（字节拼接）
- [ ] save/load序列化（JSON格式）
- [ ] 往返断言（编码再解码与原字符串相同）
- [ ] 特殊token处理（allow_special标志）
