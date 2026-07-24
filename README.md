<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1a1a1a?style=flat-square&labelColor=fafaf5" alt="MIT License"></a>
  <a href="#contents"><img src="https://img.shields.io/badge/课程-503-3553ff?style=flat-square&labelColor=fafaf5" alt="503 lessons"></a>
  <a href="#contents"><img src="https://img.shields.io/badge/阶段-20-3553ff?style=flat-square&labelColor=fafaf5" alt="20 phases"></a>
  <a href="https://github.com/1583477320/ai-engineering-from-scratch-zh/stargazers"><img src="https://img.shields.io/github/stars/1583477320/ai-engineering-from-scratch-zh?style=flat-square&labelColor=fafaf5&color=3553ff" alt="GitHub stars"></a>
  <a href="https://ai-engineering-from-scratch-zh-steel.vercel.app"><img src="https://img.shields.io/badge/在线站点-steel.vercel.app-3553ff?style=flat-square&labelColor=fafaf5" alt="Website"></a>
</p>

```
░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒
```

> 503 节课程，20 个阶段，约 320 小时。Python。
> 每节课产出一个可复用的工具：提示词、技能、Agent、MCP 服务器。
> 免费、开源、MIT 许可。
>
> 不只是学 AI。是从零构建 AI——端到端、亲手实现。

```
░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒
```

## 这不是翻译，是重构

本项目基于 **AI Engineering From Scratch** 的课程体系，重新构建了一套适合中文读者学习的 AI Engineering 教程。

**核心原则：**

- **不要逐句翻译** — 理解知识点，重新组织内容，使用自己的表达
- **教材优先** — 输出更像一本教材，而不是博客，循序渐进、逻辑清晰
- **增加原创内容** — 中文案例、工程实践、LLM Perspective、常见错误、面试考点、最佳实践
- **面向工程** — 所有知识尽可能联系真实工程（PyTorch、HuggingFace、Transformers、LangChain、vLLM 等）

## 课程体系

每节课遵循相同的循环：阅读问题 → 推导数学 → 编写代码 → 运行测试 → 保留产出。

```
课程结构：
┌─ MOTTO（一句话核心洞察）
├─ 学习目标（可验证的能力）
├─ 1. 问题（从真实痛点出发）
├─ 2. 核心概念（直觉优先，配公式）
├─ 3. 从零实现（分步构建）
├─ 4. 工业工具（PyTorch / sklearn / HuggingFace）
├─ 5. 知识连线 / LLM 视角
├─ 6. 工程最佳实践（含中文场景建议）
├─ 7. 常见错误（现象→原因→修复）
├─ 8. 面试考点（概念→编码→设计）
└─ 产出（可复用的提示词 / 技能 / 代码）
```

## 开始使用

**方式 A — 在线阅读。** 访问 [ai-engineering-from-scratch-zh-steel.vercel.app](https://ai-engineering-from-scratch-zh-steel.vercel.app)。无需配置，无需克隆。

**方式 B — 克隆到本地。**
```bash
git clone https://github.com/1583477320/ai-engineering-from-scratch-zh.git
cd ai-engineering-from-scratch-zh
python3 lessons/03-深度学习核心/03-反向传播/code/main.py
```

## 当前进度

已完成 **5 个阶段 · 93 节课程**：

| 阶段 | 课程数 | 状态 |
|------|--------|------|
| 00 · 环境搭建 | 12 | ✅ 已完成 |
| 01 · 数学基础 | 22 | ✅ 已完成 |
| 02 · 机器学习基础 | 18 | ✅ 已完成 |
| 03 · 深度学习核心 | 13 | ✅ 已完成 |
| 04 · 计算机视觉 | 28 | ✅ 已完成 |
| 05 · NLP 基础 | 29 | 🔧 进行中 |
| 06 · 语音与音频 | 17 | ⬚ 待开始 |
| 07 · Transformer 深入 | 16 | ⬚ 待开始 |
| 08 · 生成式 AI | 15 | ⬚ 待开始 |
| 09 · 强化学习 | 14 | ⬚ 待开始 |
| 10 · 从零构建大语言模型 | 15 | ⬚ 待开始 |
| 11 · LLM 工程 | 17 | ⬚ 待开始 |
| 12 · 多模态 AI | 25 | ⬚ 待开始 |
| 13 · 工具与协议 | 23 | ⬚ 待开始 |
| 14 · 智能体工程 | 42 | ⬚ 待开始 |
| 15 · 自主系统 | 22 | ⬚ 待开始 |
| 16 · 多智能体 | 28 | ⬚ 待开始 |
| 17 · 基础设施与生产 | 28 | ⬚ 待开始 |
| 18 · 伦理安全对齐 | 30 | ⬚ 待开始 |
| 19 · 综合项目 | 87 | ⬚ 待开始 |

## 每课产出

每节课不仅仅是学习，还会产出可复用的成果：

| 产出类型 | 说明 |
|---------|------|
| 💡 提示词 | 可直接复制到 AI 助手中的专家提示词 |
| ⚡ 技能 | 可安装到 Claude/Cursor 的交互技能 |
| 🖥️ 代码 | 从零实现的核心算法 |

## 技术栈

- **语言**：Python（主要）
- **基础库**：NumPy, PyTorch, JAX
- **框架**：scikit-learn, HuggingFace Transformers, LangChain
- **部署**：前端纯静态 HTML/CSS/JS，托管于 Vercel

## 参考资料

每节课末尾列出参考信息，优先引用：
- 原始论文
- 官方文档
- GitHub 仓库
- 教科书

## 贡献

欢迎通过 Issue 和 Pull Request 贡献。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## License

MIT License — 允许学习、修改、重构、商业使用。

## 致谢

- 原课程 [AI Engineering From Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare
- 所有为本项目贡献的开发者
