# 引用规范 (Citation Guide)

> **规则：** 所有引用必须遵循本指南。引用质量直接影响教程的可信度。

---

## 1. 引用优先级

选择引用来源时，按以下优先级：

```
论文 > 官方文档 > GitHub 仓库 > 技术标准(RFC/W3C) > 权威技术书籍 > 技术报告
```

**禁止引用的来源：**

| ❌ 禁止 | 原因 |
|---|---|
| 知乎专栏 / 知乎回答 | 未经同行评议，质量不可控 |
| CSDN / 博客园 | 大量洗稿/翻译，信息不可追溯 |
| Medium（非官方技术博客） | 无质量门槛 |
| 个人博客（除非作者是该领域的公认专家） | 无法验证准确性 |
| 微信公众号文章 | 不可引用，链接不稳定 |
| B站视频 / YouTube 视频 | 不可引用，除非是官方课程 |
| 未发表的手稿 / Preprint 未经 arXiv 托管 | 无法验证 |
| 中文翻译版论文（引用原版） | 翻译可能不准确 |

---

## 2. 引用格式

### 2.1 论文

```markdown
[序号] [论文] 作者. "标题". 会议/期刊, 年份. URL

# 示例
[1] [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
[2] [论文] Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL, 2019. https://arxiv.org/abs/1810.04805
[3] [论文] Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units". ACL, 2016. https://arxiv.org/abs/1508.07909
```

规则：
- 作者 ≤ 3 人时，列出全部姓氏；> 3 人时用 "et al."（或中文用"等"）
- 标题使用英文原文，用双引号包裹
- 会议/期刊使用标准缩写或全称
- 优先使用 arXiv 链接（永久可访问），其次为会议官网
- 不引用中文翻译版

### 2.2 官方文档

```markdown
[序号] [官方文档] 组织/项目. 页面标题. URL

# 示例
[4] [官方文档] PyTorch. "MultiheadAttention". https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
[5] [官方文档] Hugging Face. "Auto Classes". https://huggingface.co/docs/transformers/model_doc/auto
[6] [官方文档] OpenAI. "tiktoken". https://github.com/openai/tiktoken
```

规则：
- 组织名在前，页面标题在后
- 链接到具体页面，不链接到网站首页
- 优先使用官方域名下的文档（不推荐 readthedocs 镜像，除非是官方唯一入口）

### 2.3 GitHub 仓库

```markdown
[序号] [GitHub] 作者/组织. 仓库名. URL

# 示例
[7] [GitHub] Dao-AILab. "flash-attention". https://github.com/Dao-AILab/flash-attention
[8] [GitHub] vLLM Project. "vllm". https://github.com/vllm-project/vllm
[9] [GitHub] Meta. "llama". https://github.com/meta-llama/llama
```

规则：
- 使用 `作者/仓库名` 格式
- 链接到仓库首页（不是特定文件，特定文件容易失效）
- 如果必须引用特定文件，标注 commit hash：

```markdown
[GitHub] PyTorch. "torch/nn/functional.py". https://github.com/pytorch/pytorch/blob/abc1234/torch/nn/functional.py
```

### 2.4 技术标准

```markdown
[序号] [RFC/RFC中文] 编号: 标题. URL

# 示例
[10] [RFC] RFC 9110: HTTP Semantics. https://www.rfc-editor.org/rfc/rfc9110
[11] [RFC中文] RFC 9110 中文翻译: HTTP 语义. https://www.rfc-editor.org/rfc/rfc9110
```

### 2.5 技术书籍

```markdown
[序号] [书籍] 作者. 《书名》. 出版社, 出版年份.

# 示例
[12] [书籍] Goodfellow, Bengio, Courville. 《Deep Learning》. MIT Press, 2016.
[13] [书籍] 李航. 《统计学习方法（第2版）》. 清华大学出版社, 2019.
```

规则：
- 英文书籍保留原书名，中文书籍使用中文书名
- 标注出版社和年份
- 不提供盗版 PDF 链接

---

## 3. 正文中的引用方式

### 3.1 引用样式

在正文中引用时，使用上标序号：

```markdown
Vaswani 等人在 2017 年首次提出了 Transformer 架构[1]。后续 BERT[2] 证明了预训练 + 微调范式的有效性。

FlashAttention[7] 通过 IO 感知的注意力计算，将内存复杂度从 O(n²) 降低到 O(n)。
```

### 3.2 首次引用

首次引用一个来源时，给出完整上下文：

```markdown
# ✓ 有上下文
在 "Attention Is All You Need" (Vaswani et al., NeurIPS 2017)[1] 中，作者首次提出用纯注意力机制替代循环神经网络。

# ❌ 光秃秃的引用
Transformer 架构[1] 是一种...
```

---

## 4. 引用位置

所有引用放在课程末尾的「📖 参考资料」章节中，按在正文中出现的顺序排列：

```markdown
## 📖 参考资料

1. [论文] Vaswani et al. "Attention Is All You Need". NeurIPS, 2017. https://arxiv.org/abs/1706.03762
2. [论文] Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units". ACL, 2016. https://arxiv.org/abs/1508.07909
3. [官方文档] PyTorch. "MultiheadAttention". https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
4. [GitHub] OpenAI. "tiktoken". https://github.com/openai/tiktoken
```

---

## 5. 最少引用数量

| 课程类型 | 最少引用数 | 说明 |
|---|---|---|
| 概念课 | ≥ 3 | 至少包含 1 篇论文 + 1 份官方文档 |
| 实现课 | ≥ 5 | 至少包含 2 篇论文 + 1 份官方文档 + 1 个 GitHub 仓库 |

---

## 6. 链接可访问性

- 所有 URL 应为 HTTPS
- 不使用短链接（如 bit.ly, t.cn）
- 不使用需要登录才能访问的链接
- 不使用百度网盘 / 阿里云盘等网盘链接
- arXiv 链接使用 `https://arxiv.org/abs/XXXX.XXXXX` 格式（使用 abs 而非 pdf）

---

## 7. 版权声明

本项目的每课文档末尾应附版权声明：

```markdown
---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、案例、LLM 视角分析、工程最佳实践、常见错误、面试考点等均为原创内容。
```

---

## 8. 自检清单

- [ ] 引用数量达到最低要求
- [ ] 优先引用了论文和官方文档
- [ ] 没有引用知乎、CSDN、Medium、微信公众号
- [ ] 所有 URL 使用 HTTPS 且可访问
- [ ] 论文引用了 arXiv 版本（而非会议 paywall）
- [ ] GitHub 仓库链接到正确的仓库首页
- [ ] 正文中的引用位置正确（上标序号）
- [ ] 参考资料列表按出现顺序排列
