# CLAUDE.md

# AI Engineering From Scratch CN

Project instructions for Claude Code.

---

# Project Goal

本项目旨在参考 **AI Engineering From Scratch** 的课程体系，重新构建一套适合中文读者学习的 AI Engineering 教程。

这是一个 **Reconstruction（重构）** 项目，而不是 **Translation（翻译）** 项目。

目标包括：

- 建立系统化的中文 AI Engineering 教程
- 帮助中文读者理解 AI 核心知识
- 强调工程实践，而不是理论堆砌
- 与现代 LLM、Agent、AI Engineering 保持联系

---

# Working Principles

所有输出必须遵循以下原则。

## 1. 不要逐句翻译

不要复制原文。

不要保持原文表达方式。

不要简单改写英文。

应该：

- 理解知识点
- 重新组织内容
- 使用自己的表达
- 加入更多解释

---

## 2. 教材优先

输出应更像一本教材，而不是博客。

要求：

- 循序渐进
- 逻辑清晰
- 易于理解
- 保持专业

不要假设读者已经理解所有背景知识。

---

## 3. 增加原创内容

在原课程基础上增加：

- 中文案例
- 更多解释
- 工程实践
- LLM Perspective
- 常见错误
- 面试考点
- 最佳实践

不要只重复原课程内容。

---

## 4. 面向工程

所有知识尽可能联系真实工程。

例如：

- HuggingFace
- PyTorch
- Transformers
- LangChain
- LangGraph
- vLLM
- TensorRT-LLM

解释：

工业界如何使用这些知识。

---

## 5. 保持一致

整个仓库保持统一：

- Markdown 风格
- 标题层级
- 术语
- 图片风格
- 代码风格

不要自行改变。

---

# Lesson Template

所有 Lesson 必须严格遵循：

docs/README_TEMPLATE.md

不要自行增加或删除章节。

除非用户明确要求。

---

# Style Guide

所有文档必须遵循：

docs/STYLE_GUIDE.md

---

# Terminology

所有术语统一遵循：

docs/TERMINOLOGY.md

不要混用不同翻译。

例如：

✓ Token → 词元

✗ Token → Token

✗ Token → 标记

---

# Code Style

所有代码遵循：

docs/CODE_STYLE.md

要求：

- 可以运行
- 注释使用中文
- 示例尽量简洁
- 避免无意义代码

---

# Figure Style

所有流程图遵循：

docs/FIGURE_GUIDE.md

优先：

- Mermaid
- ASCII

尽量避免复杂图片。

---

# References

引用统一遵循：

docs/CITATION.md

优先引用：

- 官方文档
- 原始论文
- GitHub
- RFC

不要引用低质量博客。

---

# Copyright

原课程：

AI Engineering From Scratch

License：

MIT License

允许：

- 学习
- 修改
- 重构
- 商业使用

本项目应：

- 尊重原作者
- 保留 License
- 保持原创表达

不要逐句翻译。

---

# Claude Behavior

当收到：

"整理"

"写中文版"

"重构"

"继续下一课"

等请求时：

Claude 应：

1. 阅读对应 Lesson
2. 理解知识点
3. 按 README_TEMPLATE 重构
4. 遵循 STYLE_GUIDE
5. 使用统一术语
6. 输出完整 Markdown

不要：

- 输出解释过程
- 输出翻译过程
- 输出 Prompt

直接生成最终文档。

---

# Quality Checklist

每次完成 Lesson 后，自行检查：

- 是否符合 README_TEMPLATE？
- 是否符合 STYLE_GUIDE？
- 是否统一术语？
- 是否增加了工程实践？
- 是否增加了 LLM Perspective？
- 是否增加了中文案例？
- 是否避免逐句翻译？
- 是否可以直接发布？

如果有任何一项不满足，请先修改，再输出最终版本。
