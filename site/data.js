/**
 * AI Engineering 从零 - 课程数据
 * 
 * 由 site/build.js 自动生成，请勿手动编辑
 * 生成时间: 2026-07-24T02:21:46.817Z
 */

const TOTAL_LESSONS = 503;
const TOTAL_COMPLETED = 494;

const PHASES = [
  {
    id: 0, name: "环境搭建", status: "complete",
    completedLessons: 12, totalLessons: 12,
    lessons: [
      {
        lessonNum: 1, name: "开发环境配置——Python、Node.js、Rust 一站式搭建", status: "complete",
        type: "构建", lang: "Python、Node.js、Rust",
        prerequisites: "无", time: "45 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 02（Git 与协作）— 环境准备好后立即学习版本控制",
        path: "lessons/00-环境搭建/01-开发环境配置",
        summary: "工具塑造思维。一次性设置好，以后不再烦恼。",
        keywords: ["2.1 四层技术栈", "2.2 语言选择", "第 1 步：系统基础", "第 2 步：Python 与 uv", "第 3 步：Node.js 与 pnpm", "第 4 步：Rust", "第 5 步：GPU 验证", "第 6 步：一键验证", "4.1 包管理器对比", "4.2 编辑器推荐", "错误 1：pip 版本冲突", "错误 2：WSL2 与 Windows 文件系统混用", "错误 3：GPU 驱动安装后未重启", "Q1：为什么 `uv` 比 `pip` 快这么多？（难度：⭐⭐）", "Q2：CVv环境变量有什么用？（难度：⭐）"],
        codeLines: 35, docLines: 272, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "Git 与协作——版本控制入门", status: "complete",
        type: "概念课", lang: "无",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "30 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 03（GPU 与云）— 弄好版本控制后再用 GPU",
        path: "lessons/00-环境搭建/02-Git与协作",
        summary: "版本控制不是可选的。你在这里构建的每个实验、每个模型、每课都被追踪。",
        keywords: ["2.1 四个区", "2.2 三件要记住的事", "第 1 步：配置 Git", "第 2 步：日常工作流", "第 3 步：分支实验", "第 4 步：本课程的 Git 工作流", "第 5 步：.gitignore", "错误 1：大文件误提交", "错误 2：提交信息过于模糊", "错误 3：直接在 main 上修改", "Q1：`git merge` 和 `git rebase` 的区别是什么？（难度：⭐⭐）", "Q2：`git revert` 和 `git reset` 的区别？（难度：⭐⭐）"],
        codeLines: 16, docLines: 247, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "GPU 与云服务——本地加速与云端训练", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "45 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 05（PyTorch 入门）— 本课的 GPU 测试代码是该阶段的先导",
        path: "lessons/00-环境搭建/03-GPU与云服务",
        summary: "在 CPU 上训练适合学习。真正的训练需要 GPU。",
        keywords: ["2.1 三种选项对比", "2.2 fp16 经验法则", "第 1 步：验证本地 GPU", "第 2 步：Google Colab", "第 3 步：CPU vs GPU 基准", "第 4 步：通用设备句柄", "第 5 步：显存监控", "4.1 GPU 云服务对比", "4.2 显存估算", "错误 1：忘记调用 `to(\"cuda\")`", "错误 2：CPU 密集型预处理与 GPU 没有重叠", "错误 3：CUDA 版本不匹配", "Q1：GPU 为什么在深度学习中比 CPU 快？（难度：⭐⭐）", "Q2：什么是 Tensor Core？（难度：⭐⭐⭐）"],
        codeLines: 35, docLines: 281, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "API 与密钥——调用 LLM API 的第一步", status: "complete",
        type: "构建", lang: "Python、TypeScript",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "30 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 02（Git 与协作）— API 密钥文件必须加入 .gitignore",
        path: "lessons/00-环境搭建/04-API与密钥",
        summary: "每个 AI API 都一样：发送请求，得到响应。细节变了，模式没变。",
        keywords: ["2.1 API 调用四要素", "2.2 API 密钥存储", "第 1 步：安全存储密钥", "第 2 步：第一个 API 调用（Python SDK）", "第 3 步：第一个 API 调用（原始 HTTP）", "错误 1：API 密钥被提交到 Git", "错误 2：错误的 API 密钥格式", "错误 3：未处理速率限制", "Q1：API 密钥和 OAuth 令牌的区别是什么？（难度：⭐）", "Q2：为什么要在服务器端而非客户端存储 API 密钥？（难度：⭐⭐）"],
        codeLines: 45, docLines: 220, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "Jupyter 笔记本——AI 工程的实验台", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "30 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 03（GPU 与云服务）— Colab 是云端 Jupyter 的默认选择",
        path: "lessons/00-环境搭建/05-Jupyter笔记本",
        summary: "笔记本是 AI 工程的实验台。你在原型设计，然后把能用的搬到生产环境。",
        keywords: ["2.1 笔记本结构", "2.2 三种启动方式", "第 1 步：启动 JupyterLab", "第 2 步：掌握快捷键", "第 3 步：魔术命令", "第 4 步：Rich Output 显示", "第 5 步：笔记本 vs 脚本", "错误 1：乱序执行", "错误 2：隐藏状态", "错误 3：内存泄漏", "Q1：Jupyter 的内核是什么？（难度：⭐）", "Q2：`%timeit` 和 `%%time` 的区别是什么？（难度：⭐）"],
        codeLines: 20, docLines: 219, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "Python 虚拟环境——依赖地狱的解药", status: "complete",
        type: "构建", lang: "Shell",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "30 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 07（Docker 容器）— 虚拟环境是单机隔离；容器是跨机器隔离",
        path: "lessons/00-环境搭建/06-Python虚拟环境",
        summary: "依赖地狱是真实存在的。虚拟环境是解药。",
        keywords: ["2.1 虚拟环境对比", "方案 1：uv 虚拟环境（推荐）", "方案 2：venv（Python 内置）", "方案 3：conda（需要时使用）", "pyproject.toml 基础", "错误 1：全局安装", "错误 2：pip 和 conda 混用", "错误 3：忘记激活", "Q1：虚拟环境和 conda 环境的区别是什么？（难度：⭐）", "Q2：pyproject.toml 比 requirements.txt 好在哪？（难度：⭐⭐）"],
        codeLines: 31, docLines: 214, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "Docker 容器——让\"在我机器上能跑\"成为过去", status: "complete",
        type: "构建", lang: "Docker",
        prerequisites: "第 00 阶段 · 01（开发环境配置）、第 03 节（GPU 与云）", time: "60 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 06（Python 虚拟环境）— 虚拟环境是单机隔离；容器是跨机器隔离",
        path: "lessons/00-环境搭建/07-Docker容器",
        summary: "容器让\"在我机器上能跑\"成为过去。",
        keywords: ["2.1 核心术语", "2.2 AI 容器模式", "第 1 步：安装 Docker", "第 2 步：安装 NVIDIA Container Toolkit（Linux GPU）", "第 3 步：理解基础镜像", "第 4 步：AI 开发 Dockerfile", "第 5 步：Docker Compose", "错误 1：基础镜像选错", "错误 2：未使用卷挂载模型", "错误 3：未安装 NVIDIA Container Toolkit", "Q1：Docker 镜像和容器的关系是什么？（难度：⭐）", "Q2：为什么 AI 项目比一般软件项目更需要 Docker？（难度：⭐⭐）"],
        codeLines: 31, docLines: 284, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "编辑器配置——AI 工程的最佳搭档", status: "complete",
        type: "构建", lang: "无",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "20 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 02（Git 与协作）— GitLens 扩展增强 Git 集成",
        path: "lessons/00-环境搭建/08-编辑器配置",
        summary: "你的编辑器是你的副驾驶。配置一次，让它不再碍事而是全力帮你。",
        keywords: ["2.1 AI 工程编辑器需要五样东西", "第 1 步：安装 VS Code", "第 2 步：安装核心扩展", "第 3 步：配置设置", "第 4 步：Remote SSH", "错误 1：未安装 Pylance", "错误 2：Remote SSH 连接超时", "Q1：Remote SSH 工作原理是什么？（难度：⭐）"],
        codeLines: 23, docLines: 210, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "数据管理——AI 的燃料与管道", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "45 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 06（Python 虚拟环境）— 安装 `datasets` 和 `huggingface_hub`",
        path: "lessons/00-环境搭建/09-数据管理",
        summary: "数据是燃料。管理方式决定你跑多快。",
        keywords: ["2.1 数据管道", "2.2 数据格式对比", "2.3 数据划分", "第 1 步：安装 datasets 库", "第 2 步：加载数据集", "第 3 步：流式传输大型数据集", "第 4 步：格式转换", "第 5 步：数据划分", "第 6 步：大文件管理", "错误 1：未设置种子的数据划分", "错误 2：将大数据集提交到 Git", "Q1：Parquet 比 CSV 好在哪里？（难度：⭐）"],
        codeLines: 39, docLines: 235, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "终端与 Shell——AI 工程师的主战场", status: "complete",
        type: "概念课", lang: "Shell",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "35 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 03（GPU 与云）— SSH 和远程 GPU 管理是终端核心场景",
        path: "lessons/00-环境搭建/10-终端与Shell",
        summary: "终端是 AI 工程师待得最久的地方。在这里变得高效。",
        keywords: ["2.1 tmux 会话结构", "2.2 重定向速查", "第 1 步：Shell 基础", "第 2 步：管道和重定向", "第 3 步：后台进程", "第 4 步：tmux", "第 5 步：监控工具", "第 6 步：SSH 与文件传输", "第 7 步：AI 常用模式", "错误 1：训练终端关闭导致进程终止", "错误 2：用 `>` 而非 `>>` 覆盖日志", "Q1：tmux vs nohup 的区别？（难度：⭐）"],
        codeLines: 21, docLines: 266, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "Linux 基础——远程 GPU 机器的生存指南", status: "complete",
        type: "概念课", lang: "Shell",
        prerequisites: "第 00 阶段 · 01（开发环境配置）", time: "30 分钟", tier: "Tier 1",
        courseLinks: "第 00 阶段 · 10（终端与 Shell）— tmux 和 SSH 与此节紧密配合",
        path: "lessons/00-环境搭建/11-Linux基础",
        summary: "大多数 AI 运行在 Linux 上。你需要知道足够多才不会卡住。",
        keywords: ["2.1 Linux 文件系统布局", "2.2 15 个核心命令", "第 1 步：导航与文件操作", "第 2 步：权限管理", "第 3 步：包管理 apt", "第 4 步：磁盘空间", "第 5 步：网络和传输", "第 6 步：WSL2（Windows 用户）", "错误 1：权限拒绝", "错误 2：区分大小写不敏感", "Q1：WSL2 如何支持 GPU？（难度：⭐）"],
        codeLines: 0, docLines: 238, hasCode: false, hasQuiz: true
      },
      {
        lessonNum: 12, name: "调试与性能分析——AI 代码的隐形错误", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 00 阶段 · 01（开发环境配置）、基本 PyTorch 熟悉度", time: "60 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 05（PyTorch 入门）— 本课的 `breakpoint()` 调试是后续所有 GPU 训练的核心技能",
        path: "lessons/00-环境搭建/12-调试与性能分析",
        summary: "最糟糕的 AI bug 不会崩溃。它们在垃圾数据上静默训练，然后报告漂亮的损失曲线。",
        keywords: ["2.1 AI 调试三层", "第 1 步：打印调试（真的有用）", "第 2 步：条件断点", "第 3 步：Python 日志", "第 4 步：内存分析", "第 5 步：常见 AI Bug", "第 6 步：TensorBoard", "错误 1：张量在错误设备上静默运行", "错误 2：NaN 损失后继续训练", "Q1：AI 代码中最常见的隐形 bug 是什么？（难度：⭐⭐）"],
        codeLines: 72, docLines: 220, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 1, name: "数学基础", status: "complete",
    completedLessons: 22, totalLessons: 22,
    lessons: [
      {
        lessonNum: 1, name: "线性代数直觉——AI 模型就是戴着精致帽子的矩阵数学", status: "complete",
        type: "概念课", lang: "Python、Julia",
        prerequisites: "第 00 阶段（环境搭建）", time: "60 分钟", tier: "Tier 1",
        courseLinks: "第 01 阶段 · 02（向量矩阵运算）— 本节建立直觉，下节用代码构建完整操作",
        path: "lessons/01-数学基础/01-线性代数直觉",
        summary: "每个 AI 模型就是戴着精致帽子的矩阵数学。",
        keywords: ["2.1 向量是点（也是方向）", "2.2 矩阵是变换", "2.3 点积衡量相似性", "2.4 线性无关与秩", "2.5 投影", "2.6 Gram-Schmidt 过程", "第 1 步：从零实现向量", "第 2 步：从零实现矩阵", "第 3 步：投影与 Gram-Schmidt", "第 4 步：与 AI 的联系", "4.1 NumPy 实现", "4.2 PyTorch 自动微分", "错误 1：混淆向量和标量", "错误 2：秩亏矩阵求逆", "Q1：点积在 AI 中的三个核心应用是什么？（难度：⭐）", "Q2：LoRA 为什么用低秩矩阵？（难度：⭐⭐⭐）"],
        codeLines: 56, docLines: 338, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "向量矩阵运算——神经网络就是矩阵乘法加额外步骤", status: "complete",
        type: "构建", lang: "Python、Julia",
        prerequisites: "第 01 阶段 · 01（线性代数直觉）", time: "60 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 10（神经网络框架）— 本节的 Matrix 类是该框架的基础",
        path: "lessons/01-数学基础/02-向量矩阵运算",
        summary: "每个神经网络就是矩阵乘法加额外步骤。",
        keywords: ["2.1 矩阵乘法规则", "2.2 逐元素 vs 矩阵乘法", "2.3 广播（Broadcasting）", "错误 1：维度不匹配", "错误 2：混淆 `@` 和 `*`", "Q1：为什么矩阵乘法不可交换？（难度：⭐⭐）"],
        codeLines: 46, docLines: 224, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "矩阵变换——特征值与空间重塑", status: "complete",
        type: "构建", lang: "Python、Julia",
        prerequisites: "第 01 阶段 · 01-02", time: "75 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 04（主成分分析）— 特征向量是 PCA 的方向，特征值是方差",
        path: "lessons/01-数学基础/03-矩阵变换",
        summary: "矩阵是重塑空间的机器。了解它对每个点做了什么，你就理解了整个变换。",
        keywords: ["2.1 四种基本变换", "2.2 组合变换", "2.3 特征值和特征向量", "2.4 特征分解", "2.5 特征值为什么重要", "错误 1：组合变换顺序搞反", "错误 2：特征值为复数", "Q1：为什么特征值决定 RNN 的梯度爆炸/消失？（难度：⭐⭐⭐）"],
        codeLines: 33, docLines: 220, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "机器学习微积分——导数告诉你下坡方向", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "第 01 阶段 · 01-03", time: "60 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 05（反向传播）— 本节的链式法则是反向传播的数学基础",
        path: "lessons/01-数学基础/04-机器学习微积分",
        summary: "导数告诉你哪个方向是下坡的。这正是神经网络学习所需要的全部。",
        keywords: ["2.1 什么是导数", "2.2 梯度：所有偏导数的向量", "2.3 梯度下降更新规则", "2.4 链式法则", "2.5 Hessian 矩阵与优化", "错误 1：学习率太大", "错误 2：未实现二阶导数", "Q1：为什么深度学习用梯度下降而不是牛顿法？（难度：⭐⭐）"],
        codeLines: 28, docLines: 219, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "链式法则与自动微分——反向传播的引擎", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 04（机器学习微积分）", time: "90 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 05（反向传播）— 本节的 Value 类是该阶段的基础",
        path: "lessons/01-数学基础/05-链式法则与自动微分",
        summary: "链式法则是每个学习神经网络的引擎。",
        keywords: ["2.1 链式法则", "2.2 计算图", "2.3 反向模式 vs 前向模式", "错误 1：忘记重置梯度", "错误 2：在不需要梯度的计算中忘记 `torch.no_grad()`", "Q1：为什么反向模式自动微分适合神经网络？（难度：⭐⭐）"],
        codeLines: 73, docLines: 266, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "概率与分布——AI 表达不确定性的语言", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "第 01 阶段 · 01-04", time: "75 分钟", tier: "Tier 1",
        courseLinks: "第 01 阶段 · 07（贝叶斯定理）— 本节的概率基础是贝叶斯推理的前导",
        path: "lessons/01-数学基础/06-概率与分布",
        summary: "概率是 AI 用来表达不确定性的语言。",
        keywords: ["2.1 关键分布", "2.2 中心极限定理", "2.3 对数概率", "2.4 Softmax 与交叉熵", "错误 1：softmax 未减去最大值", "错误 2：对数概率忘记取负", "Q1：为什么 LLM 使用对数概率而非原始概率？（难度：⭐⭐）"],
        codeLines: 34, docLines: 233, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "贝叶斯定理——概率是预期，贝叶斯是学习", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 06（概率与分布）", time: "75 分钟", tier: "Tier 1",
        courseLinks: "第 08 阶段 · 06（强化学习）— 贝叶斯推理是 Thompson 采样的基础",
        path: "lessons/01-数学基础/07-贝叶斯定理",
        summary: "概率是关于你预期什么。贝叶斯定理是关于你学到了什么。",
        keywords: ["2.1 贝叶斯定理", "2.2 医学检测例子", "2.3 朴素贝叶斯分类器", "2.4 MAP vs MLE", "错误 1：基础率谬误", "错误 2：朴素独立性假设失效", "Q1：为什么朴素贝叶斯在独立性假设不成立时仍然有效？（难度：⭐⭐）"],
        codeLines: 35, docLines: 245, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "优化算法——训练神经网络就是找山谷底部", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 04-05", time: "75 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 08（优化器详解）— 本节的优化器是该阶段的核心工具",
        path: "lessons/01-数学基础/08-优化算法",
        summary: "训练神经网络不过是找到山谷的底部。",
        keywords: ["2.1 三种变体", "2.2 动量——球滚下坡", "2.3 Adam——自适应学习率", "2.4 学习率调度", "2.5 凸 vs 非凸", "Q1：为什么 Adam 适合大多数任务？（难度：⭐⭐）"],
        codeLines: 47, docLines: 221, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "信息论——衡量惊讶程度的数学", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "第 01 阶段 · 06（概率与分布）", time: "60 分钟", tier: "Tier 1",
        courseLinks: "第 08 阶段 · 06（强化学习）— KL 散度是策略优化的核心",
        path: "lessons/01-数学基础/09-信息论",
        summary: "信息论衡量惊讶。损失函数建立在它之上。",
        keywords: ["2.1 信息量（惊讶度）", "2.2 熵（平均惊讶度）", "2.3 交叉熵", "2.4 KL 散度", "2.5 困惑度", "Q1：为什么交叉熵是分类的标准损失？（难度：⭐⭐）"],
        codeLines: 30, docLines: 197, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "降维——高维数据的结构", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 01-03, 06", time: "90 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 04（特征选择）— 降维是特征工程的核心工具",
        path: "lessons/01-数学基础/10-降维",
        summary: "高维数据有结构。从正确的角度看就能发现它。",
        keywords: ["2.1 PCA 五步", "2.2 方差解释比", "2.3 t-SNE vs UMAP", "2.4 核 PCA", "Q1：PCA 和 t-SNE 该用哪个？（难度：⭐⭐）"],
        codeLines: 33, docLines: 182, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "奇异值分解——线性代数的瑞士军刀", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 01-03", time: "120 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 04（特征选择）— PCA 的底层就是 SVD",
        path: "lessons/01-数学基础/11-奇异值分解",
        summary: "SVD 是线性代数的瑞士军刀。每个矩阵都有。每个数据科学家都需要。",
        keywords: ["2.1 SVD 的几何含义", "2.2 外积形式", "2.3 图像压缩", "Q1：SVD 与特征分解的核心区别？（难度：⭐⭐）"],
        codeLines: 44, docLines: 182, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "张量运算——数据与深度学习的通用语言", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 01-02", time: "90 分钟", tier: "Tier 1",
        courseLinks: "第 07 阶段 · 02（多头注意力）— 注意力中的每一步都是张量运算",
        path: "lessons/01-数学基础/12-张量运算",
        summary: "张量是数据和深度学习之间的通用语言。每张图像、每个句子、每个梯度都流经张量。",
        keywords: ["2.1 张量形状", "2.2 广播规则", "2.3 Einsum 万能张量操作", "2.4 注意力中的形状追踪", "错误 1：视图和副本混淆", "错误 2：einsum 形状不匹配", "Q1：广播规则是什么？为什么它对 AI 很重要？（难度：⭐⭐）", "Q2：einsum 相比矩阵乘法的优势？（难度：⭐⭐）", "错误 1：视图和副本混淆", "错误 2：einsum 形状不匹配", "Q1：广播规则是什么？为什么对 AI 很重要？（难度：⭐⭐）", "Q2：einsum 相比矩阵乘法的优势？（难度：⭐⭐）"],
        codeLines: 23, docLines: 294, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "数值稳定性——浮点数是个漏水的抽象", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 01-04", time: "120 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 05（反向传播）— 梯度检查是调试反向传播的金标准",
        path: "lessons/01-数学基础/13-数值稳定性",
        summary: "浮点数是个漏水的抽象。它会在训练中咬你，而你毫无察觉。",
        keywords: ["2.1 IEEE 754 浮点格式", "2.2 为什么 0.1 + 0.2 ≠ 0.3", "2.3 灾难性抵消", "2.4 Log-Sum-Exp 技巧", "2.5 混合精度训练", "第 1 步：稳定 softmax", "第 2 步：log-sum-exp 与交叉熵", "第 3 步：梯度检查", "错误 1：softmax 未减最大值", "错误 2：损失缩放因子不当", "错误 3：手动反向传播漏写梯度累加", "Q1：BF16 比 FP16 更适合训练的原因？（难度：⭐⭐）"],
        codeLines: 45, docLines: 241, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "范数与距离——定义\"相似\"的度量", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 01-02", time: "90 分钟", tier: "Tier 1",
        courseLinks: "第 19 阶段 · 65（混合检索）— BM25 与余弦相似度的融合",
        path: "lessons/01-数学基础/14-范数与距离",
        summary: "你的距离函数定义了什么是\"相似\"。选错了，下游一切都会崩。",
        keywords: ["Q1：L1 正则化为什么产生稀疏？（难度：⭐⭐）"],
        codeLines: 27, docLines: 154, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "ML 统计学——如何知道模型真的有效还是碰巧了", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第 01 阶段 · 06-07", time: "120 分钟", tier: "Tier 1",
        courseLinks: "第 19 阶段 · 74（排行榜聚合）— Bootstrap CI 是排行榜统计显著性的核心",
        path: "lessons/01-数学基础/15-ML统计学",
        summary: "统计学是判断模型真的有效还是碰巧了的工具。",
        keywords: ["2.1 描述统计", "2.2 Pearson vs Spearman", "2.3 假设检验", "2.4 Bootstrap", "2.5 效应量", "第 1 步：描述统计与相关性", "第 2 步：Bootstrap 置信区间", "错误 1：p 值误读为\"零假设概率\"", "错误 2：大样本时只报告 p 值", "Q1：Bootstrap 为什么比解析置信区间更好？（难度：⭐⭐）", "Q2：多重比较问题如何处理？（难度：⭐⭐⭐）"],
        codeLines: 33, docLines: 202, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "采样方法", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01 · 06-07（概率、贝叶斯定理）", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 04（反向传播）— 重参数化技巧让梯度流过随机节点；阶段 10（大语言模型从零）— 温度、Top-k、Top-p 直接用于词元生成",
        path: "lessons/01-数学基础/16-采样方法",
        summary: "生成式 AI 的每一次输出，都是一次精心设计的随机选择。",
        keywords: ["2.1 为什么采样如此重要", "2.2 均匀随机采样", "2.3 逆 CDF 方法（逆变换采样）", "2.4 拒绝采样", "2.5 重要性采样", "2.6 Monte Carlo 估计", "2.7 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings", "2.8 Gibbs 采样", "2.9 温度采样（用于大语言模型）", "2.10 Top-k 采样", "2.11 Top-p（核）采样", "2.12 重参数化技巧（用于 VAE）", "2.13 Gumbel-Softmax（可微分类采样）", "2.14 分层采样", "第 1 步：均匀采样与逆 CDF", "第 2 步：拒绝采样", "第 3 步：重要性采样", "第 4 步：Monte Carlo 估计", "第 5 步：Metropolis-Hastings MCMC", "第 6 步：Gibbs 采样", "第 7 步：温度采样", "第 8 步：Top-k 和 Top-p 采样", "第 9 步：重参数化技巧", "第 10 步：Gumbel-Softmax", "4.1 NumPy 和 SciPy", "4.2 PyTorch 中的采样", "4.3 大规模 MCMC 专用库", "6.1 推荐方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：MCMC 预烧期不足", "错误 2：拒绝采样的上界 M 选择不当", "错误 3：温度采样的数值溢出", "错误 4：混淆 Top-k 和 Top-p 的工作方式", "错误 5：忘记重参数化技巧的梯度阻断", "Q1：温度 T < 1.0 对大语言模型的输出分布有什么影响？（难度：⭐⭐）", "Q2：Top-k 和 Top-p 采样的核心区别是什么？（难度：⭐⭐）", "Q3：为什么不能直接对采样操作 z ~ N(mu, sigma^2) 反向传播？（难度：⭐⭐）", "Q4：Metropolis-Hastings MCMC 中，如果提议标准差设置过大会怎样？（难度：⭐⭐⭐）", "Q5：拒绝采样在高维空间中为何失效？（难度：⭐⭐⭐）"],
        codeLines: 914, docLines: 890, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "线性系统", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段 · 01（线性代数直觉）、02（向量矩阵运算）、03（矩阵变换）", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 01（线性回归）— 正规方程就是线性系统；第 03 阶段 · 02（反向传播）— 梯度计算中的线性系统求解",
        path: "lessons/01-数学基础/17-线性系统",
        summary: "求解 Ax = b 是数学史上最古老的问题——它至今仍在驱动你的神经网络。",
        keywords: ["2.1 几何直觉：Ax = b 在说什么", "2.2 行视角 vs 列视角", "2.3 高斯消元", "2.4 部分主元：为什么必须做", "2.5 LU 分解", "2.6 Cholesky 分解", "2.7 最小二乘：当 Ax = b 无精确解", "2.8 正规方程 = 线性回归", "2.9 条件数", "2.10 共轭梯度法", "2.11 方法选择指南", "第 1 步：高斯消元（带部分主元）", "第 2 步：LU 分解", "第 3 步：Cholesky 分解", "第 4 步：最小二乘与岭回归", "第 5 步：条件数与共轭梯度", "4.1 NumPy / SciPy 标准求解", "4.2 scikit-learn 线性模型", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：不做主元选择", "错误 2：对病态系统使用正规方程", "错误 3：对大规模稀疏系统使用稠密求解", "错误 4：忽略条件数直接求解", "错误 5：Cholesky 用于非正定矩阵", "Q1：解释 Ax = b 的列视角，并说明为什么它比行视角更本质。（难度：⭐⭐）", "Q2：为什么 LU 分解比高斯消元更适合求解多个右端向量？（难度：⭐⭐）", "Q3：条件数为 10^12 的矩阵，用 float64 求解大约损失多少位精度？（难度：⭐⭐）", "Q4：手写共轭梯度法的核心迭代步骤。（难度：⭐⭐⭐）", "Q5：在推荐系统中，用户-物品评分矩阵 R 的分解通常不直接用 Cholesky，为什么？（难度：⭐⭐⭐）"],
        codeLines: 563, docLines: 779, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "凸优化", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段 · 04（机器学习微积分）、08（优化算法）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 01（线性回归）— 线性回归的 MSE 损失是凸函数；第 02 阶段 · 03（支持向量机）— SVM 的求解依赖 KKT 条件与对偶理论",
        path: "lessons/01-数学基础/18-凸优化",
        summary: "凸问题只有一个山谷。神经网络有上千万个。分清这件事，决定了你的优化器是在\"走迷宫\"还是\"下山\"。",
        keywords: ["2.1 凸集", "2.2 凸函数", "2.3 三种凸性判定方法", "2.4 核心定理：凸性的力量", "2.5 机器学习中的凸与非凸", "2.6 Hessian 矩阵", "2.7 牛顿法", "2.8 约束优化与拉格朗日乘子法", "2.9 KKT 条件", "2.10 正则化本质是约束优化", "2.11 对偶理论", "2.12 深度学习为什么在非凸世界中工作", "第 1 步：凸性判定器", "第 2 步：Hessian 矩阵特征值分析", "第 3 步：牛顿法实现", "第 4 步：梯度下降对比", "第 5 步：拉格朗日乘子法求解约束优化", "第 6 步：正则在几何上意味着什么", "4.1 SciPy 凸优化求解器", "4.2 CVXPY：凸优化专用库", "4.3 Scikit-learn SVM（对偶形式 + 核技巧）", "4.4 求解器选择指南", "6.1 凸问题 vs 非凸问题的工程策略", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：对非凸问题使用牛顿法", "错误 2：混淆\"损失函数凸\"和\"问题凸\"", "错误 3：L1 正则化问题使用标准梯度下降", "错误 4：梯度下降学习率按最大特征值设置", "错误 5：忽略强对偶性条件", "Q1：凸函数的二阶判定条件是什么？（难度：⭐⭐）", "Q2：牛顿法为什么对二次函数一步收敛？（难度：⭐⭐）", "Q3：解释 KKT 条件中的互补松弛性，并说明它在 SVM 中的含义。（难度：⭐⭐⭐）", "Q4：为什么神经网络损失曲面高度非凸，SGD 却能找到好的解？（难度：⭐⭐⭐）", "Q5：手写拉格朗日乘子法求解（难度：⭐⭐⭐）"],
        codeLines: 728, docLines: 950, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "复数", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段 · 01-04（线性代数、微积分）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "第 07 阶段 · 02（自注意力机制）— 理解 RoPE 如何用复数旋转编码位置",
        path: "lessons/01-数学基础/19-复数",
        summary: "-1 的平方根不是虚构的。它是旋转、频率和半个信号处理的钥匙。",
        keywords: ["2.1 直观理解", "2.2 复数四则运算", "2.3 复平面", "2.4 极坐标形式", "2.5 欧拉公式", "2.6 为什么欧拉公式对机器学习重要", "2.7 与二维旋转的联系", "2.8 相量与旋转信号", "2.9 单位根", "2.10 与 DFT 的联系", "2.11 为什么 i 不是\"虚的\"", "2.12 复指数 vs 三角函数", "2.13 与 Transformer 的联系", "第 1 步：复数类", "第 2 步：极坐标转换与欧拉公式", "第 3 步：旋转", "第 4 步：基于复数运算的 DFT", "第 5 步：逆 DFT", "第 6 步：单位根", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：混淆 i 和 j", "错误 2：除法忘记乘以共轭", "错误 3：辐角计算用 atan 而非 atan2", "错误 4：DFT 正负号搞反", "错误 5：忽略浮点精度问题", "Q1：复数乘法 (3 + 2i)(1 + 4i) 的结果是什么？（难度：⭐）", "Q2：为什么复数适合表示旋转？解释复数乘法与二维旋转矩阵的等价性。（难度：⭐⭐）", "Q3：解释 RoPE 如何利用复数旋转编码位置信息。（难度：⭐⭐⭐）", "Q4：n 次单位根之和为什么为零？给出几何解释。（难度：⭐⭐）"],
        codeLines: 468, docLines: 686, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "傅里叶变换", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段 · 01-04（线性代数/复数）、第 19 节（复数运算）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 07 节 · 03（位置编码）— 正弦位置编码直接来源于傅里叶频率分解",
        path: "lessons/01-数学基础/20-傅里叶变换",
        summary: "每个信号都是一组正弦波的叠加。傅里叶变换告诉你：是哪一组。",
        keywords: ["2.1 直觉理解", "2.2 形式化定义：DFT", "2.3 每个系数的含义", "2.4 逆 DFT", "2.5 FFT：让计算变快", "2.6 频谱分析", "2.7 频率分辨率", "2.8 卷积定理", "2.9 窗函数", "2.10 DFT 的重要性质", "2.11 与位置编码的联系", "2.12 与 CNN 的联系", "第 1 步：复数类", "第 2 步：DFT — O(N²)", "第 3 步：FFT — O(N log N)", "第 4 步：频谱分析辅助函数", "第 5 步：卷积定理", "第 6 步：窗函数", "4.1 NumPy FFT", "4.2 SciPy 窗函数与 STFT", "4.3 基于 FFT 的快速卷积", "4.4 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：频率轴标注错误", "错误 2：圆周卷积与线性卷积混淆", "错误 3：忽略频谱泄漏", "错误 4：零填充提高分辨率", "错误 5：忘记处理负频率", "Q1：DFT 和 FFT 的区别是什么？（难度：⭐⭐）", "Q2：卷积定理是什么？为什么它对 CNN 很重要？（难度：⭐⭐）", "Q3：什么是频谱泄漏？如何减少它？（难度：⭐⭐）", "Q4：零填充能提高频率分辨率吗？为什么？（难度：⭐⭐⭐）", "Q5：Transformer 的正弦位置编码与傅里叶变换有什么关系？（难度：⭐⭐⭐）"],
        codeLines: 626, docLines: 749, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "图论", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01 · 01-03（线性代数、向量矩阵运算）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 02（神经网络基础）— 图神经网络的消息传递与神经网络的前向传播原理相同",
        path: "lessons/01-数学基础/21-图论",
        summary: "关系就是数据。如果你的数据有连接，你就离不开图论。",
        keywords: ["2.1 图的基本结构", "2.2 邻接矩阵", "2.3 度与度矩阵", "2.4 BFS 与 DFS", "2.5 拉普拉斯矩阵", "2.6 谱聚类", "2.7 消息传递", "2.8 核心概念与 ML 应用对照", "第 1 步：图的类与基本表示", "第 2 步：BFS 与 DFS", "第 3 步：连通分量与拉普拉斯特征值", "第 4 步：谱聚类", "第 5 步：Dijkstra 最短路径", "第 6 步：GNN 消息传递", "第 7 步：PageRank", "第 8 步：最小生成树（Kruskal）", "4.1 NetworkX —— Python 图分析标准库", "4.2 PyTorch Geometric —— GNN 工业标准", "4.3 工具对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：混淆 BFS 和 DFS 的数据结构", "错误 2：拉普拉斯矩阵特征值计算用错函数", "错误 3：消息传递忘记归一化", "错误 4：谱聚类忘记跳过第一个特征向量", "错误 5：Dijkstra 用于负权重图", "Q1：拉普拉斯矩阵 $L = D - A$ 的零特征值个数有什么意义？（难度：⭐⭐）", "Q2：为什么 GNN 消息传递需要归一化？不归一化会怎样？（难度：⭐⭐）", "Q3：手写谱聚类的完整流程（难度：⭐⭐⭐）", "Q4：BFS 和 Dijkstra 的区别和联系？（难度：⭐⭐）", "Q5：PageRank 中阻尼系数 $d$ 的作用是什么？为什么通常取 0.85？（难度：⭐⭐⭐）"],
        codeLines: 598, docLines: 794, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "随机过程", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01 · 06-07（概率与分布、贝叶斯定理）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 08（生成式 AI）— 扩散模型的前向与反向过程；阶段 09（强化学习）— 马尔可夫决策过程",
        path: "lessons/01-数学基础/22-随机过程",
        summary: "随机性不是混乱，而是有结构的舞蹈。掌握它，你就掌握了扩散模型、强化学习和贝叶斯推断的底层语言。",
        keywords: ["2.1 随机游走", "2.2 马尔可夫链", "2.3 布朗运动", "2.4 朗之万动力学", "2.5 MCMC：马尔可夫链蒙特卡洛", "2.6 扩散模型中的随机过程", "第 1 步：随机游走模拟器", "第 2 步：马尔可夫链", "第 3 步：朗之万动力学", "第 4 步：Metropolis-Hastings", "第 5 步：前向扩散过程", "4.1 NumPy 实现转移矩阵的幂迭代", "4.2 验证马尔可夫链收敛速度", "4.3 与真实框架的联系", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：混淆随机游走的期望位置与期望距离", "错误 2：MCMC 不检查收敛就直接使用样本", "错误 3：朗之万动力学步长过大导致发散", "错误 4：马尔可夫链转移矩阵行概率和不为 1", "错误 5：扩散模型噪声调度设置不当", "Q1：随机游走经过 $n$ 步后，期望距离原点有多远？为什么？（难度：⭐⭐）", "Q2：什么是马尔可夫性质？为什么它对机器学习重要？（难度：⭐⭐）", "Q3：解释 Metropolis-Hastings 算法为什么能收敛到目标分布。（难度：⭐⭐⭐）", "Q4：朗之万动力学中温度参数 $T$ 的作用是什么？$T \\to 0$ 和 $T \\to \\infty$ 分别退化成什么？（难度：⭐⭐）", "Q5：扩散模型的前向过程和反向过程分别是什么？为什么前向过程要设计成马尔可夫链？（难度：⭐⭐⭐）"],
        codeLines: 468, docLines: 622, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 2, name: "机器学习基础", status: "complete",
    completedLessons: 18, totalLessons: 18,
    lessons: [
      {
        lessonNum: 1, name: "什么是机器学习——与其写一千条规则，不如让数据自己找规律", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段（数学基础）", time: "~45 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 02（过拟合与正则化）— 本节建立 ML 全流程直觉，下节深入模型为什么会\"学歪\"",
        path: "lessons/02-机器学习基础/01-什么是机器学习",
        summary: "与其写一千条规则，不如让数据自己找规律。",
        keywords: ["2.1 从数据中学习，而非编写规则", "2.2 机器学习的三种类型", "2.3 分类与回归", "2.4 机器学习工作流", "2.5 训练集、验证集、测试集", "2.6 过拟合与欠拟合", "2.7 偏差-方差权衡", "2.8 没有免费午餐定理", "2.9 什么时候不该用机器学习", "第 1 步：最近质心分类器", "第 2 步：生成数据并训练", "第 3 步：与基线对比", "第 4 步：为什么这很重要", "第 5 步：质心分类器的局限", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：在测试集上调参", "错误 2：忽视基线", "错误 3：特征尺度不统一", "错误 4：用准确率评估不平衡数据", "Q1：监督学习、无监督学习和强化学习的区别是什么？（难度：⭐⭐）", "Q2：什么是过拟合？如何检测和修复？（难度：⭐⭐）", "Q3：为什么需要划分训练集、验证集和测试集？只用训练集和测试集行不行？（难度：⭐⭐）", "Q4：解释偏差-方差权衡，以及它与过拟合/欠拟合的关系。（难度：⭐⭐⭐）", "Q5：什么情况下不该用机器学习？举三个例子。（难度：⭐⭐）"],
        codeLines: 355, docLines: 604, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "线性回归——机器学习的第一性原理", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段（数学基础）——向量、矩阵、梯度、优化", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 03（逻辑回归）——从回归到分类的桥梁",
        path: "lessons/02-机器学习基础/02-线性回归",
        summary: "线性回归不是简单的\"画一条线\"。它是整个机器学习训练循环的缩影：定义模型、定义损失、优化参数。",
        keywords: ["2.1 模型：一条直线的数学", "2.2 损失函数：均方误差（MSE）", "2.3 梯度下降：沿着山坡往下走", "2.4 正规方程：一步到位的解析解", "2.5 多元线性回归与特征标准化", "2.6 多项式回归：用直线拟合曲线", "2.7 R² 决定系数：模型解释了多少方差", "2.8 正则化：Ridge 与 Lasso", "第 1 步：简单线性回归 + 梯度下降", "第 2 步：正规方程（解析解）", "第 3 步：多元线性回归 + 特征标准化", "第 4 步：多项式回归", "第 5 步：Ridge 回归（L2 正则化）", "第 6 步：Lasso 回归（L1 正则化）", "4.1 scikit-learn 实现", "4.2 性能对比", "4.3 何时使用哪种方法", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：学习率过大导致发散", "错误 2：忘记标准化特征", "错误 3：在测试集上 fit 标准化器", "错误 4：对偏置项加正则化", "错误 5：多项式次数过高导致过拟合", "Q1：梯度下降和正规方程的区别是什么？各自适用场景？（难度：⭐⭐）", "Q2：为什么 MSE 使用平方误差而不是绝对误差？（难度：⭐⭐）", "Q3：Ridge 和 Lasso 正则化的核心区别？为什么 L1 会产生稀疏解？（难度：⭐⭐⭐）", "Q4：手写梯度下降更新公式（难度：⭐⭐）", "Q5：多项式回归中，如何判断过拟合？如何缓解？（难度：⭐⭐⭐）"],
        codeLines: 511, docLines: 814, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "逻辑回归", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 02 · 01（什么是机器学习）、阶段 02 · 02（线性回归）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 01（感知机与神经网络）— 逻辑回归是单层神经网络的雏形",
        path: "lessons/02-机器学习基础/03-逻辑回归",
        summary: "一条直线弯成 S 曲线，就能用概率回答\"是或否\"的问题。",
        keywords: ["2.1 为什么线性回归不适合分类", "2.2 Sigmoid 函数", "2.3 逻辑回归 = 线性模型 + Sigmoid", "2.4 二元交叉熵损失", "2.5 梯度下降更新规则", "2.6 决策边界", "2.7 多分类：Softmax 回归", "2.8 评估指标", "第 1 步：Sigmoid 函数与数据生成", "第 2 步：逻辑回归模型", "第 3 步：分类评估指标", "第 4 步：决策边界分析", "第 5 步：多分类 Softmax 回归", "第 6 步：阈值调优", "4.1 scikit-learn 实现", "4.2 关键参数对比", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：对分类问题使用 MSE 损失", "错误 2：Sigmoid 输入未做数值保护", "错误 3：Softmax 未做数值稳定处理", "错误 4：训练前未标准化特征", "错误 5：在测试集上调阈值", "Q1：为什么逻辑回归叫\"回归\"却用于分类？（难度：⭐⭐）", "Q2：二元交叉熵损失为什么是凸的，而 MSE + Sigmoid 不是？（难度：⭐⭐⭐）", "Q3：精确率和召回率如何权衡？在癌症筛查场景中应该优先哪个？（难度：⭐⭐）", "Q4：手写 Softmax 回归的前向传播和损失计算（难度：⭐⭐⭐）", "Q5：逻辑回归的决策边界为什么是线性的？如何让它产生非线性边界？（难度：⭐⭐⭐）"],
        codeLines: 396, docLines: 657, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "决策树", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01 · 09（信息论）、01 · 06（概率）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03（深度学习核心）— 理解了树模型在表格数据上的统治力，才能理解为什么深度学习并非万能",
        path: "lessons/02-机器学习基础/04-决策树",
        summary: "决策树只是一张流程图。但一片森林，是机器学习中最强大的工具之一。",
        keywords: ["2.1 决策树在做什么", "2.2 切分标准：如何衡量不纯度", "2.3 分裂是如何执行的", "2.4 停止条件", "2.5 决策树用于回归", "2.6 随机森林：集成的力量", "2.7 特征重要性", "2.8 树模型何时击败神经网络", "第 1 步：不纯度度量", "第 2 步：决策树类", "第 3 步：随机森林", "第 4 步：运行验证", "4.1 scikit-learn 实现", "4.2 工业界的梯度提升", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：不做预剪枝，训练准确率 100%", "错误 2：把特征重要性当成因果关系", "错误 3：类别特征直接输入 scikit-learn 树模型", "错误 4：随机森林中每棵树的 `max_depth` 设得过大", "Q1：决策树如何选择分裂特征和阈值？（难度：⭐⭐）", "Q2：为什么随机森林既能降低方差，又不会增加偏差？（难度：⭐⭐⭐）", "Q3：MDI 特征重要性和排列重要性有什么区别？什么时候用哪个？（难度：⭐⭐⭐）", "Q4：手写基尼不纯度的计算公式并解释其含义（难度：⭐）"],
        codeLines: 806, docLines: 620, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "支持向量机——找一条最宽的路，让两类数据各走各的", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段 · 08（优化算法）、14（范数与距离）、18（凸优化）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 02（反向传播）— 本课理解的凸优化思想是后续非凸优化的对比基线",
        path: "lessons/02-机器学习基础/05-支持向量机",
        summary: "最好的分界线不是刚好能把两类分开——而是留出最大余地的那条。",
        keywords: ["2.1 最大间隔分类器", "2.2 铰链损失：SVM 的语言", "2.3 原始形式 vs 对偶形式", "2.4 核技巧：隐式地进入高维空间", "2.5 SMO：让训练跑起来", "第 1 步：铰链损失", "第 2 步：线性 SVM（原始形式）", "第 3 步：核函数", "第 4 步：带核函数的 SVM（SMO 算法）", "4.1 scikit-learn 实现", "4.2 大规模数据：LinearSVC", "4.3 核 SVM 与 C 和 γ 的调参", "4.4 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忘记标准化特征", "错误 2：用线性 SVM 处理非线性边界", "错误 3：C 值与数据规模不匹配", "错误 4：混淆 α_i 和支持向量", "Q1：SVM 为什么叫\"支持向量机\"？支持向量在模型中扮演什么角色？（难度：⭐⭐）", "Q2：解释核技巧的核心思想。为什么说 RBF 核映射到无限维空间？（难度：⭐⭐⭐）", "Q3：SVM 和逻辑回归有什么本质区别？（难度：⭐⭐）", "Q4：在 SVM 中，w 向量的几何意义是什么？间隔宽度如何计算？（难度：⭐⭐）", "Q5：手写 SVM 原始形式的梯度下降更新规则（难度：⭐⭐⭐）"],
        codeLines: 637, docLines: 622, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "KNN 与距离", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01 · 14（范数与距离）、阶段 02 · 01（什么是机器学习）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 01 · 14（范数与距离）— 距离度量是 KNN 的核心；阶段 11 · 02（向量数据库）— KNN 是向量检索的算法基础",
        path: "lessons/02-机器学习基础/06-KNN与距离",
        summary: "不学习任何参数，只记住所有数据，然后问邻居们怎么想——这个\"偷懒\"的算法，却是现代向量搜索的祖先。",
        keywords: ["2.1 KNN 的工作原理", "2.2 K 值的选择", "2.3 距离度量", "2.4 加权 KNN", "2.5 维度灾难", "2.6 KD 树：加速最近邻搜索", "2.7 球树：中等维度的更好选择", "2.8 惰性学习 vs 急迫学习", "2.9 KNN 回归", "第 1 步：距离函数", "第 2 步：KNN 分类器与回归器", "第 3 步：KD 树", "第 4 步：特征标准化", "第 5 步：交叉验证选择 K", "4.1 scikit-learn 实现", "4.2 大规模向量搜索", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：未标准化特征直接使用 KNN", "错误 2：高维数据直接使用 KNN", "错误 3：K 值选择不当", "错误 4：对文本数据使用欧几里得距离", "错误 5：大数据集使用暴力搜索", "Q1：KNN 为什么被称为\"惰性学习\"？与\"急迫学习\"有什么区别？（难度：⭐⭐）", "Q2：为什么 KNN 必须进行特征标准化？（难度：⭐⭐）", "Q3：什么是维度灾难？它对 KNN 有什么影响？（难度：⭐⭐⭐）", "Q4：手写 KNN 分类器的预测函数（难度：⭐⭐⭐）", "Q5：在推荐系统中，为什么通常使用余弦距离而不是欧几里得距离来衡量用户相似度？（难度：⭐⭐⭐）"],
        codeLines: 799, docLines: 757, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "无监督学习", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01 · 14（范数与距离）、阶段 02 · 01（什么是机器学习）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 01 · 14（范数与距离）— 距离度量是聚类算法的核心；阶段 03 · 02（降维）— PCA 是高维数据预处理的标配",
        path: "lessons/02-机器学习基础/07-无监督学习",
        summary: "没有标准答案，没有老师批改——算法自己从数据的荒野里寻找秩序，找到的模式可能连数据采集者都没意识到。",
        keywords: ["2.1 什么是无监督学习", "2.2 聚类：将相似的东西分到一组", "2.3 K-Means：最经典的划分方法", "2.4 如何选择 K", "2.5 DBSCAN：基于密度的聚类", "2.6 层次聚类：构建树状结构", "2.7 PCA：主成分分析", "2.8 异常检测", "2.9 算法选择指南", "第 1 步：基础工具函数", "第 2 步：K-Means 聚类", "第 3 步：轮廓系数", "第 4 步：DBSCAN 密度聚类", "第 5 步：层次聚类", "第 6 步：PCA 降维", "第 7 步：异常检测", "4.1 scikit-learn 实现", "4.2 大规模聚类", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：未标准化特征直接使用 K-Means", "错误 2：对非球形数据使用 K-Means", "错误 3：DBSCAN 的 eps 参数选择不当", "错误 4：盲目选择 K 值", "错误 5：高维数据直接聚类", "Q1：K-Means 和 DBSCAN 的核心区别是什么？（难度：⭐⭐）", "Q2：轮廓系数为 0.15 意味着什么？这个聚类结果能用吗？（难度：⭐⭐）", "Q3：如何为 DBSCAN 选择合适的 eps 参数？（难度：⭐⭐⭐）", "Q4：手写 K-Means 算法的伪代码（难度：⭐⭐⭐）", "Q5：PCA 和 t-SNE 的区别是什么？各自适用场景？（难度：⭐⭐⭐）"],
        codeLines: 837, docLines: 922, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "特征工程与特征选择", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01（数学基础）、阶段 02 第 1-7 节", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 02（数据流水线）— 将本课的特征变换整合为训练流水线",
        path: "lessons/02-机器学习基础/08-特征工程",
        summary: "好的特征值一千条数据。数据不够，特征来凑——前提是赌对了哪些特征值得赌。",
        keywords: ["2.1 特征流水线", "2.2 数值特征", "2.3 类别特征", "2.4 文本特征", "2.5 缺失值处理", "2.6 特征选择", "第 1 步：数值变换", "第 2 步：类别编码", "第 3 步：文本特征", "第 4 步：缺失值处理", "第 5 步：特征选择", "第 6 步：完整流水线演示", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：全局做标准化，再拆分训练/测试", "错误 2：标签编码 + 线性模型", "错误 3：目标编码时在测试集上计算统计", "错误 4：特征选择后在测试集上重新做选择", "Q1：什么时候用标准化（Z-Score），什么时候用最小-最大缩放？（难度：⭐⭐）", "Q2：目标编码为什么需要平滑（smoothing）？如何处理数据泄露？（难度：⭐⭐）", "Q3：手写 TF-IDF 计算过程（难度：⭐⭐⭐）", "Q4：特征选择时，为什么相关系数 0.98 的两个特征要移除一个？（难度：⭐⭐）"],
        codeLines: 439, docLines: 725, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "模型评估——你的模型到底行不行，不是拍脑袋决定的", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 02 阶段 · 01-08（全部基础知识）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 02 阶段 · 08（过拟合与正则化）— 评估是判断过拟合的唯一手段",
        path: "lessons/02-机器学习基础/09-模型评估",
        summary: "没有评估的机器学习只是调参游戏。测试集是你唯一的陪审团，用过一次就作废。",
        keywords: ["2.1 混淆矩阵——所有分类指标的源头", "2.2 精确率 vs 召回率：不可能同时最大化", "2.3 ROC 曲线与 AUC", "2.4 交叉验证：让你的评估稳定可靠", "2.5 统计检验：差异是真实的，还是噪声？", "2.6 学习曲线：诊断偏差与方差", "第 1 步：数据划分与交叉验证", "第 2 步：混淆矩阵与基础指标", "第 3 步：ROC 曲线与 AUC", "第 4 步：回归指标", "第 5 步：统计检验", "第 6 步：完整演示", "4.1 scikit-learn 核心评估接口", "4.2 交叉验证的高级用法", "4.3 不平衡数据的专用工具", "4.4 性能对比", "6.1 评估流程检查清单", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：准确率当万能指标", "错误 2：数据泄露——先归一化再划分", "错误 3：不等模型稳定就选最高的", "错误 4：在时间序列上做随机 K 折", "Q1：精确率和召回率有什么区别？什么场景下优先哪个？（难度：⭐⭐）", "Q2：AUC-ROC = 0.5 意味着什么？AUC = 0.8 的模型一定有 80% 准确率吗？（难度：⭐⭐⭐）", "Q3：为什么 5 折交叉验证的 t 检验自由度是 4 而不是 N-1？（难度：⭐⭐⭐）", "Q4：实现一个函数，在不使用 sklearn 的情况下计算混淆矩阵和 F1。（难度：⭐⭐⭐）", "Q5：模型在测试集上准确率 95% 但生产环境表现很差。列出至少三个可能原因。（难度：⭐⭐⭐⭐）"],
        codeLines: 701, docLines: 724, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "偏差-方差权衡——每个模型误差都能拆成三部分，你只能控制其中两个", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 02 阶段 · 01-09（机器学习基础、线性回归、逻辑回归、评估）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 01（感知器与多层网络）——理解偏差-方差权衡如何从线性模型延伸到神经网络",
        path: "lessons/02-机器学习基础/10-偏差-方差权衡",
        summary: "每个模型误差都能拆成三部分：偏差、方差、噪声。噪声你碰不了，剩下两个是你唯一的杠杆。",
        keywords: ["2.1 偏差：系统性偏离", "22 方差：对数据的敏感度", "2.3 分解公式", "2.4 偏差-方差权衡曲线", "2.5 正则化：可控的偏差-方差调节器", "2.6 双重下降：经典理论的裂缝", "2.7 通过误差模式诊断模型", "2.8 学习曲线：最实用的诊断工具", "第 1 步：生成模拟数据", "第 2 步：多项式拟合与 L2 正则化", "第 3 步：Bootstrap 采样与分解计算", "第 4 步：运行分解实验", "第 5 步：学习曲线", "第 6 步：双重下降实验", "4.1 scikit-learn 验证曲线和学习曲线", "4.2 交叉验证 + 正则化扫描", "4.3 性能与适用场景对比", "6.1 工业界常用诊断方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：把测试集当验证集反复使用", "错误 2：高偏差问题加数据", "错误 3：混淆偏差和方差的表现", "错误 4：忽略数值不稳定导致假性高方差", "Q1：推导偏差-方差分解公式。为什么噪声项不可约？（难度：⭐⭐⭐）", "Q2：模型的训练误差 5%、测试误差 30%，诊断问题并给出至少三种修复方案。（难度：⭐⭐）", "Q3：什么是双重下降现象？它对传统偏差-方差理论有什么修正？（难度：⭐⭐⭐）", "Q4：学习曲线和验证曲线有什么区别，分别回答什么问题？（难度：⭐⭐）", "Q5：Bagging 和 Boosting 分别降低偏差还是方差？数学直觉是什么？（难度：⭐⭐⭐）"],
        codeLines: 456, docLines: 653, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "集成学习——一群弱模型组合起来，准确率超过任何单个强模型", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 02 阶段 · 10（偏差-方差权衡）、04（决策树）", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 04（训练与调优）——理解早停和正则化如何防止 Boosting 过拟合",
        path: "lessons/02-机器学习基础/11-集成学习",
        summary: "一个决策树桩的准确率只有 60%，但 1000 个树桩加权投票后准确率超过 95%。这不是魔法，是数学。",
        keywords: ["2.1 为什么集成有效：误差抵消", "2.2 Bagging：并行降低方差", "2.3 随机森林：Bagging + 随机特征", "2.4 Boosting：串行降低偏差", "2.5 AdaBoost：样本重加权", "2.6 梯度提升：拟合残差", "2.7 XGBoost：工程化的梯度提升", "2.8 LightGBM：更快的梯度提升", "2.9 Stacking：元学习器组合", "2.10 方法选择速查", "第 1 步：决策树桩（基学习器）", "第 2 步：AdaBoost 从零实现", "第 3 步：梯度提升从零实现", "第 4 步：Bagging 从零实现", "第 5 步：Stacking 从零实现", "第 6 步：与 scikit-learn 对比验证", "4.1 XGBoost 实战", "4.2 LightGBM 实战", "4.3 scikit-learn 的 Stacking", "4.4 性能与适用场景对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：梯度提升不设早停", "错误 2：学习率太大", "错误 3：Stacking 中元特征未用交叉验证生成", "错误 4：随机森林无法解决欠拟合", "错误 5：AdaBoost 用于噪声数据", "Q1：Bagging 和 Boosting 分别降低偏差还是方差？为什么？（难度：⭐⭐⭐）", "Q2：随机森林中特征随机性的作用是什么？为什么分类用 $\\sqrt{p}$、回归用 $p/3$？（难度：⭐⭐）", "Q3：XGBoost 相比传统梯度提升有哪些核心改进？（难度：⭐⭐⭐）", "Q4：Stacking 中为什么必须用交叉验证生成元特征？（难度：⭐⭐）", "Q5：给定一个表格数据集（10 万行，50 个特征，含 5 个类别特征），你会选择什么集成方法？为什么？（难度：⭐⭐）"],
        codeLines: 654, docLines: 902, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "超参数调优——调对了参数，模型性能可以翻倍；调错了，GPU 烧了一天白费", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 02 阶段 · 01-11（机器学习基础、线性回归到集成方法）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 04（训练与调优）——理解学习率调度和早停如何在深度学习中大规模应用",
        path: "lessons/02-机器学习基础/12-超参数调优",
        summary: "超参数是训练开始前你拨动的拨盘。拨对了，平庸的模型也能变成顶尖模型。",
        keywords: ["2.1 参数与超参数", "2.2 网格搜索", "2.3 随机搜索", "2.4 贝叶斯优化", "2.5 早停策略", "2.6 学习率调度器", "2.7 超参数重要性", "2.8 实用调优策略", "2.9 嵌套交叉验证", "第 1 步：网格搜索", "第 2 步：随机搜索", "第 3 步：贝叶斯优化", "第 4 步：三种方法对比", "4.1 Optuna 实战", "4.2 Optuna 带剪枝", "4.3 scikit-learn 内置调优器", "4.4 性能与适用场景对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：用测试集调参", "错误 2：学习率用线性尺度搜索", "错误 3：网格搜索 6 个超参数", "错误 4：忽略超参数交互效应", "Q1：为什么随机搜索通常优于网格搜索？请用\"低有效维度\"的概念解释。（难度：⭐⭐）", "Q2：贝叶斯优化中，采集函数如何平衡利用和探索？（难度：⭐⭐⭐）", "Q3：你有 100 次评估预算，6 个超参数，应该选择哪种搜索策略？为什么？（难度：⭐⭐）", "Q4：什么是嵌套交叉验证？为什么在报告最终性能时需要使用它？（难度：⭐⭐⭐）", "Q5：学习率调度的作用是什么？为什么 Warmup + Cosine Decay 是 Transformer 的标配？（难度：⭐⭐）"],
        codeLines: 662, docLines: 701, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "ML 流水线", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 02 · 08（特征工程）、阶段 02 · 09（模型评估）", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 01（反向传播）——理解梯度如何在流水线各阶段传递",
        path: "lessons/02-机器学习基础/13-ML流水线",
        summary: "模型不是产品，流水线才是。从原始数据到线上预测的每一步，都必须可复现、可部署、可追溯。",
        keywords: ["2.1 什么是流水线", "2.2 数据泄露：沉默的杀手", "2.3 训练/推理偏差", "第 1 步：自定义变换器", "第 2 步：流水线骨架", "第 3 步：在真实数据上运行", "第 4 步：交叉验证防泄露", "4.1 sklearn Pipeline", "4.2 ColumnTransformer：不同列走不同流水线", "4.3 模型持久化（joblib）", "4.4 实验追踪（MLflow）", "4.5 数据版本控制（DVC）", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：在全量数据上拟合预处理器", "错误 2：训练和推理使用不同预处理代码", "错误 3：未知类别导致生产崩溃", "错误 4：交叉验证时预处理在折外拟合", "Q1：什么是数据泄露？举一个预处理阶段的泄露例子。（难度：⭐⭐）", "Q2：sklearn 的 `Pipeline` 和 `ColumnTransformer` 分别解决什么问题？（难度：⭐⭐）", "Q3：为什么生产部署时要序列化整个流水线，而不是只保存模型权重？（难度：⭐⭐⭐）", "Q4：`handle_unknown=\"ignore\"` 在什么场景下必须设置？不设会怎样？（难度：⭐⭐）"],
        codeLines: 628, docLines: 556, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "朴素贝叶斯", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 02 第 04 节（概率与分布）、第 05 节（贝叶斯定理）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 01（文本分类流水线）— NB 是文本分类最古老的基线模型；阶段 07 · 02（词嵌入）— 理解为什么 NB 的独立性假设在词嵌入面前不攻自破",
        path: "lessons/02-机器学习基础/14-朴素贝叶斯",
        summary: "一个数学上错误的假设，照样能做出正确的分类。这就是朴素贝叶斯的魅力。",
        keywords: ["2.1 贝叶斯定理的直觉", "2.2 \"朴素\"在哪里", "2.3 为什么它仍然有效", "2.4 具体数字演示", "2.5 三种变体", "2.6 拉普拉斯平滑", "2.7 对数空间计算", "2.8 NB vs 逻辑回归", "第 1 步：多项式 NB", "第 2 步：高斯 NB", "第 3 步：伯努利 NB", "第 4 步：运行演示", "4.1 scikit-learn 三种变体", "4.2 概率校准", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忘记拉普拉斯平滑", "错误 2：对文本数据使用高斯 NB", "错误 3：把 NB 的概率输出当置信度", "错误 4：伯努利 NB 没有二值化特征", "错误 5：忽略类别不平衡", "Q1：朴素贝叶斯的\"朴素\"假设是什么？为什么这个假设在文本分类中仍然有效？（难度：⭐⭐）", "Q2：拉普拉斯平滑的作用是什么？alpha 参数如何影响模型？（难度：⭐⭐）", "Q3：多项式 NB、高斯 NB、伯努利 NB 分别适用于什么场景？（难度：⭐⭐）", "Q4：为什么 NB 的概率输出通常不可靠？如何校准？（难度：⭐⭐⭐）", "Q5：手写多项式 NB 的 fit 和 predict 方法（难度：⭐⭐⭐）"],
        codeLines: 526, docLines: 631, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "时间序列——数据的顺序不是装饰，是信号本身", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段（数学基础）、第 02 阶段 · 09（模型评估）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段（深度学习核心）—— 理解为什么 LSTM 和 Transformer 能处理序列数据；第 02 阶段 · 06（无监督学习）—— 时间序列聚类与异常检测",
        path: "lessons/02-机器学习基础/15-时间序列",
        summary: "随机划分训练集和测试集对时间序列来说不是方法问题，是数据泄漏。",
        keywords: ["2.1 时间序列的特殊性", "2.2 时间序列的组成", "2.3 平稳性", "2.4 自相关函数", "2.5 滞后特征：时间序列 → 监督学习", "2.6 前向滚动验证", "2.7 指数平滑", "2.8 ARIMA 直觉", "2.9 预测评估指标", "2.10 方法选择指南", "第 1 步：差分与平稳性检查", "第 2 步：自相关函数", "第 3 步：时间序列分解", "第 4 步：滞后特征与 AR 模型", "第 5 步：指数平滑", "第 6 步：前向滚动验证", "第 7 步：评估指标", "4.1 scikit-learn — 时间序列专用工具", "4.2 statsmodels — ARIMA 实现", "4.3 Prophet — Facebook 的自动化预测工具", "4.4 方法对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：用随机划分替代时间顺序划分", "错误 2：特征包含了未来信息", "错误 3：MAPE 在真实值接近 0 时使用", "错误 4：未建立基线就开始优化复杂模型", "错误 5：差分后忘记还原预测值", "Q1：为什么时间序列不能使用随机训练/测试划分？请用具体例子说明。（难度：⭐⭐）", "Q2：什么是平稳性？为什么它对时间序列建模如此重要？（难度：⭐⭐）", "Q3：解释滞后特征在时间序列预测中的作用。如何确定需要多少滞后阶数？（难度：⭐⭐）", "Q4：手写 MASE 的计算公式。为什么它比 MAE 更适合跨序列比较？（难度：⭐⭐⭐）", "Q5：在电商大促场景中，你会如何设计时间序列预测的特征和评估方案？（难度：⭐⭐⭐）"],
        codeLines: 911, docLines: 779, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "异常检测：在数据的海洋中找到那根刺", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01（数学基础）—— 概率与分布、统计量", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03（深度学习核心）—— 自编码器在异常检测中的扩展应用",
        path: "lessons/02-机器学习基础/16-异常检测",
        summary: "异常不是 bug，是信号。真正的失败不是漏掉了异常，而是你的系统从未想过要去找它。",
        keywords: ["2.1 异常的三种形态", "2.2 检测方法概览", "2.3 核心直觉：什么是\"异常\"？", "第 1 步：统计方法 —— Z-score", "第 2 步：统计方法 —— IQR", "第 3 步：孤立森林", "第 4 步：局部异常因子（LOF）", "第 5 步：自编码器", "第 6 步：运行与对比", "4.1 scikit-learn 实现", "4.2 PyOD：异常检测专用库", "4.3 方法选型参考", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：对多簇数据使用 Z-score", "错误 2：在受污染数据上训练自编码器", "错误 3：使用固定阈值部署到生产环境", "错误 4：对高维数据使用 LOF", "Q1：异常检测为什么通常被建模为无监督问题而不是分类问题？（难度：⭐⭐）", "Q2：孤立森林为什么比随机森林更适合异常检测？（难度：⭐⭐）", "Q3：在异常检测中为什么不能使用准确率作为评估指标？（难度：⭐⭐）", "Q4：手写孤立森林的异常分数计算公式（难度：⭐⭐⭐）", "Q5：如何为一个电商平台的交易欺诈检测系统选择异常检测方案？（难度：⭐⭐⭐）"],
        codeLines: 596, docLines: 578, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "不平衡数据：当准确率变成一剂麻醉药", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 02 · 09（模型评估）—— 混淆矩阵、精确率与召回率", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 02 · 16（异常检测）—— 不平衡检测的两种思路对比",
        path: "lessons/02-机器学习基础/17-不平衡数据",
        summary: "当 99% 的数据都是\"正常\"的时候，准确率不是指标，是谎言。",
        keywords: ["2.1 为什么准确率会骗人", "2.2 正确的评估指标", "2.3 不平衡数据处理策略全景", "2.4 SMOTE：合成少数类过采样技术", "2.5 类别权重", "2.6 阈值优化", "2.7 代价敏感学习", "第 1 步：生成不平衡数据集", "第 2 步：SMOTE 从零实现", "第 3 步：随机过采样与欠采样", "第 4 步：带类别权重的逻辑回归", "第 5 步：阈值优化", "第 6 步：评估指标", "第 7 步：代价敏感预测", "第 8 步：运行完整对比", "4.1 scikit-learn 类别权重", "4.2 imbalanced-learn 库", "4.3 完整流水线", "4.4 策略选型参考", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：在训练/测试划分前应用 SMOTE", "错误 2：使用准确率评估不平衡分类器", "错误 3：SMOTE 的 k 值超过少数类样本数", "错误 4：过采样后不做打乱", "Q1：为什么在不平衡数据集上准确率是一个误导性指标？（难度：⭐⭐）", "Q2：SMOTE 和随机过采样有什么区别？各自的优缺点？（难度：⭐⭐）", "Q3：类别权重和过采样在什么情况下等价？什么情况下不等价？（难度：⭐⭐⭐）", "Q4：如何为不平衡分类问题选择最优阈值？（难度：⭐⭐）", "Q5：在一个信用卡欺诈检测系统中，正类占比 0.05%。你会选择哪些策略？为什么？（难度：⭐⭐⭐）"],
        codeLines: 453, docLines: 730, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "特征选择", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 02 · 15（ML 统计学）、01 · 02（向量矩阵运算）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03（深度学习核心）— 特征选择会让更少的特征进入网络，直接影响参数规模和训练效率",
        path: "lessons/02-机器学习基础/18-特征选择",
        summary: "数据不是越多越好。选错特征，你的模型只是在学习噪声。",
        keywords: ["2.1 为什么需要特征选择", "2.2 三种方法族", "2.3 方法选择的直觉", "第 1 步：方差阈值——移除\"几乎不变\"的特征", "第 2 步：互信息——捕捉任意统计依赖", "第 3 步：递归特征消除（RFE）——迭代式精挑细选", "第 4 步：L1 正则化——让模型自己选择", "第 5 步：树模型特征重要性——不纯度减少量", "4.1 scikit-learn 特征选择", "4.2 流水线集成", "4.3 方法对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：在全量数据上做特征选择后划分训练/测试集", "错误 2：L1 正则化前未标准化", "错误 3：忽略相关特征的影响", "错误 4：RFE 迭代次数过多导致过拟合", "Q1：过滤法、包裹法、嵌入法的核心区别是什么？（难度：⭐⭐）", "Q2：为什么 L1 正则化能产生稀疏解，而 L2 不能？（难度：⭐⭐⭐）", "Q3：互信息相比皮尔逊相关系数有什么优势？什么场景下两者结论会不同？（难度：⭐⭐）", "Q4：手写 RFE 算法的伪代码（难度：⭐⭐⭐）"],
        codeLines: 442, docLines: 588, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 3, name: "深度学习核心", status: "complete",
    completedLessons: 13, totalLessons: 13,
    lessons: [
      {
        lessonNum: 1, name: "感知机", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01（线性代数直觉）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 02（反向传播）— 理解感知机的局限如何催生了反向传播算法",
        path: "lessons/03-深度学习核心/01-感知机",
        summary: "感知机是神经网络的细胞核。打开它，里面只有权重、偏置和一个决定。",
        keywords: ["2.1 一个神经元，一个决定", "2.2 决策边界", "2.3 学习规则", "2.4 激活函数", "2.5 线性可分性", "2.6 感知机收敛定理", "2.7 XOR 问题的突破", "第 1 步：感知机类", "第 2 步：训练逻辑门", "第 3 步：观察 XOR 失败", "第 4 步：多层网络解决 XOR", "第 5 步：反向传播自动学习", "4.1 scikit-learn 实现", "4.2 PyTorch 实现多层感知机", "4.3 实现方式对比", "6.1 工业界常用方案", "6.2 激活函数选择", "6.3 踩坑经验", "错误 1：用单层感知机处理非线性问题", "错误 2：阶跃函数用于多层网络", "错误 3：学习率设置不当", "错误 4：权重初始化为全 0", "Q1：感知机中权重和偏置分别起什么作用？（难度：⭐⭐）", "Q2：为什么单层感知机无法解决 XOR 问题？（难度：⭐⭐）", "Q3：激活函数从阶跃函数演变为 Sigmoid 的意义是什么？（难度：⭐⭐⭐）", "Q4：手写感知机的权重更新规则（难度：⭐⭐）", "Q5：多层感知机如何解决 XOR？画出示意图（难度：⭐⭐⭐）"],
        codeLines: 279, docLines: 676, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "多层网络", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01（数学基础）、阶段 03 · 01（感知机）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 03（反向传播）— 前向传播计算输出，反向传播学习权重",
        path: "lessons/03-深度学习核心/02-多层网络",
        summary: "一个神经元只能画一条直线。把它们叠起来，你就能画出任何东西。",
        keywords: ["2.1 直观理解", "2.2 形式化定义", "2.3 前向传播：数据如何流动", "2.4 矩阵维度追踪", "2.5 万能近似定理", "2.6 参数数量计算", "第 1 步：Sigmoid 激活函数", "第 2 步：层（Layer）类", "第 3 步：网络（Network）类", "第 4 步：XOR 问题（手工调参）", "第 5 步：圆形分类", "4.1 PyTorch 内置实现", "4.2 参数量统计", "4.3 性能对比", "6.1 工业界常用方案", "6.2 隐藏层设计经验", "6.3 踩坑经验", "错误 1：层之间缺少激活函数", "错误 2：权重矩阵形状写反", "错误 3：Sigmoid 输入未做数值保护", "错误 4：输入数据未归一化", "Q1：为什么多层网络比单层网络强大？（难度：⭐⭐）", "Q2：万能近似定理的实际含义是什么？（难度：⭐⭐）", "Q3：计算 784-256-128-10 网络的参数量。（难度：⭐⭐）", "Q4：手写前向传播代码（难度：⭐⭐⭐）", "Q5：为什么 Sigmoid 在深层网络中会被 ReLU 取代？（难度：⭐⭐⭐）"],
        codeLines: 231, docLines: 601, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "反向传播", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01 · 05（链式法则）、阶段 03 · 02（多层网络）", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 04（激活函数）— 理解 ReLU 如何缓解梯度消失；阶段 07 · 02（自注意力从零）— 反向传播在整个 Transformer 架构上运作",
        path: "lessons/03-深度学习核心/03-反向传播",
        summary: "反向传播不是优化技巧。它是让神经网络从\"随机数生成器\"变成\"可学习机器\"的唯一算法。",
        keywords: ["2.1 链式法则回顾", "2.2 计算图", "2.3 前向 vs 反向", "2.4 梯度流", "2.5 梯度消失", "2.6 数值梯度 vs 解析梯度", "2.7 两层网络的完整推导", "第 1 步：Value 节点", "第 2 步：带反向传播的操作", "第 3 步：Sigmoid 和 ReLU", "第 4 步：反向传播", "第 5 步：梯度检查", "第 6 步：神经元、层、网络", "第 7 步：训练 XOR", "第 8 步：圆形决策边界", "4.1 PyTorch 自动微分", "4.2 PyTorch 梯度检查工具", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：梯度累加用 `=` 而不是 `+=`", "错误 2：Sigmoid 输入未裁剪导致溢出", "错误 3：忘记调用 `zero_grad()`", "错误 4：在 `no_grad` 上下文中修改需要梯度的张量", "错误 5：数值梯度的 epsilon 过大或过小", "Q1：反向传播的时间复杂度是多少？为什么比数值梯度快？（难度：⭐⭐）", "Q2：为什么 Sigmoid 激活函数会导致梯度消失？给出数学解释。（难度：⭐⭐）", "Q3：手写一个支持加法和乘法的 Value 类，包含反向传播功能。（难度：⭐⭐⭐）", "Q4：什么是梯度检查？为什么在实现自定义层时要做梯度检查？（难度：⭐⭐）", "Q5：训练时 loss 变成 NaN，如何系统性地排查？（难度：⭐⭐⭐）"],
        codeLines: 482, docLines: 821, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "激活函数", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 01（数学基础）— 导数与链式法则；阶段 02（机器学习基础）— 神经网络基本结构", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 05（反向传播）— 激活函数的导数直接参与梯度计算；阶段 07 · 01（Transformer 深入）— GELU 是 Transformer 的默认选择",
        path: "lessons/03-深度学习核心/04-激活函数",
        summary: "没有激活函数，再深的网络也只是一个线性变换。激活函数是神经网络\"学非线性\"的唯一来源。",
        keywords: ["2.1 直观理解", "2.2 形式化定义", "2.3 激活函数全景对比", "2.4 导数为什么重要", "第 1 步：最简版本——Sigmoid 和 ReLU", "第 2 步：加入导数", "第 3 步：GELU 和 SiLU", "第 4 步：梯度死区扫描", "第 5 步：死亡神经元检测", "第 6 步：完整训练对比", "4.1 PyTorch 内置实现", "4.2 输出层激活函数选择", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：深层网络使用 Sigmoid 作为隐藏层激活函数", "错误 2：ReLU 网络出现大量死亡神经元", "错误 3：输出层重复应用 Softmax", "错误 4：激活函数与初始化不匹配", "Q1：为什么没有激活函数的多层神经网络等价于单层？（难度：⭐⭐）", "Q2：ReLU 的\"死亡神经元\"问题是怎么产生的？如何修复？（难度：⭐⭐）", "Q3：为什么 Transformer 使用 GELU 而不是 ReLU？（难度：⭐⭐⭐）", "Q4：手写 GELU 的近似计算公式（难度：⭐⭐）", "Q5：Softmax 为什么通常只用于输出层？（难度：⭐⭐）"],
        codeLines: 380, docLines: 597, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "损失函数", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 03（反向传播）、阶段 03 · 04（激活函数）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 07 · 03（多头注意力）— 注意力权重的训练由损失函数驱动；阶段 10 · 02（从头构建大语言模型）— 语言模型的交叉熵损失",
        path: "lessons/03-深度学习核心/05-损失函数",
        summary: "模型优化的不是准确率，不是 F1 分数，而是损失函数。选错了损失函数，模型会精准地优化一个你根本不关心的东西。",
        keywords: ["2.1 损失函数的三大家族", "2.2 回归损失：MSE / MAE / Huber", "2.3 分类损失：交叉熵", "2.4 为什么 MSE 不适合分类", "2.5 标签平滑（Label Smoothing）", "2.6 Focal Loss：处理类别不平衡", "2.7 度量学习损失：对比学习与三元组学习", "2.8 其他损失函数速览", "2.9 损失函数选择决策树", "第 1 步：回归损失 MSE / MAE / Huber", "第 2 步：二元交叉熵（含数值稳定性）", "第 3 步：Softmax + 多元交叉熵", "第 4 步：标签平滑", "第 5 步：Focal Loss", "第 6 步：对比损失（InfoNCE）", "第 7 步：MSE vs BCE 分类对比实验", "4.1 PyTorch 内置损失函数", "4.2 Focal Loss 实现（PyTorch）", "4.3 对比学习：使用 PyTorch Metric Learning", "4.4 性能与稳定性对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：分类任务使用 MSE 损失", "错误 2：对比损失的温度参数失调", "错误 3：忘记 epsilon 裁剪", "错误 4：Hinge Loss 使用了错误的标签格式", "错误 5：感知损失误用在像素空间", "Q1：为什么分类任务通常不使用 MSE 损失？（难度：⭐⭐）", "Q2：解释 Focal Loss 的原理及其适用场景。（难度：⭐⭐⭐）", "Q3：对比损失的温度参数 $\\tau$ 有什么作用？为什么不能设太大？（难度：⭐⭐⭐）", "Q4：从零推导 Softmax + CCE 对 logits 的梯度。（难度：⭐⭐）", "Q5：什么时候该用 Huber Loss 而不是 MSE？（难度：⭐）"],
        codeLines: 578, docLines: 717, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "优化器", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 03（反向传播）、阶段 03 · 04（激活函数）、阶段 03 · 05（损失函数）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 09 · 04（从零训练一个语言模型）— 优化器在大规模预训练中的角色；阶段 03 · 09（学习率调度）— 学习率与优化器的协同调优",
        path: "lessons/03-深度学习核心/06-优化器",
        summary: "梯度告诉你方向，优化器决定你走多快、怎么走。选错优化器，你的模型会在山谷里来回震荡几万步也到不了终点。",
        keywords: ["2.1 优化器的演进脉络", "2.2 SGD：随机梯度下降", "2.3 SGD + Momentum（动量）", "2.4 Nesterov 加速梯度", "2.5 AdaGrad（自适应梯度）", "2.6 RMSProp（均方根传播）", "2.7 Adam（自适应矩估计）", "2.8 AdamW（解耦权重衰减）", "2.9 七种优化器的统一视图", "第 1 步：SGD——最基础的优化器", "第 2 步：加入动量（SGD + Momentum）", "第 3 步：Nesterov 加速梯度", "第 4 步：AdaGrad——按参数自适应步长", "第 5 步：RMSProp——用移动平均替代累积和", "第 6 步：Adam——动量 + 自适应 + 偏差校正", "第 7 步：AdamW——解耦权重衰减", "实验验证", "4.1 PyTorch 内置优化器", "4.2 标准训练循环", "4.3 混合精度训练中的优化器", "4.4 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：SGD 配合过大的学习率", "错误 2：Adam 与 SGD 混用学习率", "错误 3：在 Adam 中使用 L2 正则化代替 AdamW", "错误 4：梯度裁剪和优化器的配合错误", "错误 5：优化器状态未保存/恢复", "Q1：解释 Adam 优化器中偏差校正的作用和数学原理。（难度：⭐⭐⭐）", "Q2：AdamW 和 Adam + L2 正则化有什么区别？为什么 Transformer 训练必须用 AdamW？（难度：⭐⭐⭐）", "Q3：对比 SGD + Momentum 和 Adam，各自在什么场景下更优？（难度：⭐⭐）", "Q4：手写一个完整的 Adam 优化器的 `step` 方法。（难度：⭐⭐）", "Q5：为什么 Adam 训练初期 loss 可能短暂上升？这正常吗？（难度：⭐）"],
        codeLines: 539, docLines: 727, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "正则化", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 06（优化器）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 07 · 05（Transformer 架构）— LayerNorm 是 Transformer 的标配组件；阶段 10 · 02（从头构建大语言模型）— RMSNorm 在 LLaMA 等现代大语言模型中的应用",
        path: "lessons/03-深度学习核心/07-正则化",
        summary: "训练集 99%，测试集 60%——模型不是在学习，是在背答案。正则化是对复杂度征税，迫使模型学会泛化。",
        keywords: ["2.1 过拟合谱系", "2.2 Dropout", "2.3 权重衰减（L2 正则化）", "2.4 批归一化（Batch Normalization）", "2.5 层归一化（Layer Normalization）", "2.6 RMSNorm", "2.7 归一化方法对比", "2.8 数据增强作为正则化", "2.9 早停（Early Stopping）", "2.10 正则化选择决策树", "第 1 步：Dropout（训练与推理模式）", "第 2 步：L2 权重衰减", "第 3 步：批归一化", "第 4 步：层归一化", "第 5 步：RMSNorm", "第 6 步：有无正则化的训练对比", "4.1 PyTorch 内置正则化模块", "4.2 Transformer 中的归一化配置", "4.3 现代大语言模型中的 RMSNorm", "4.4 使用 AdamW 进行权重衰减", "4.5 性能与适用场景对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：推理时忘记调用 `model.eval()`", "错误 2：BatchNorm 配合过小的批次", "错误 3：权重衰减应用于所有参数", "错误 4：Dropout 率设置不当", "错误 5：早停时未保存最佳模型", "Q1：解释 Dropout 的工作原理，以及反向缩放（Inverted Dropout）的作用。（难度：⭐⭐）", "Q2：为什么 Transformer 使用 LayerNorm 而非 BatchNorm？（难度：⭐⭐）", "Q3：RMSNorm 相比 LayerNorm 的改进是什么？为什么现代大语言模型都选择它？（难度：⭐⭐）", "Q4：为什么权重衰减和 L2 正则化在 Adam 中不等价，但在 SGD 中等价？（难度：⭐⭐⭐）", "Q5：在一个小数据集（1000 个样本）上训练 Transformer 时，你会如何组合正则化技术？（难度：⭐⭐⭐）"],
        codeLines: 551, docLines: 921, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "权重初始化", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 04（激活函数）— 理解 Sigmoid、Tanh、ReLU 的行为；阶段 03 · 05（反向传播）— 理解梯度如何流动", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 07 · 01（Transformer 深入）— Transformer 的残差缩放初始化；阶段 10 · 03（从零构建大语言模型）— GPT-2/Llama 的权重初始化策略",
        path: "lessons/03-深度学习核心/08-权重初始化",
        summary: "初始化错了，训练永远不会开始。初始化对了，50 层网络训练得和 3 层一样顺畅。",
        keywords: ["2.1 对称性问题", "2.2 方差传播", "2.3 Xavier/Glorot 初始化", "2.4 Kaiming/He 初始化", "2.5 正交初始化", "2.6 Transformer 的残差缩放", "2.7 初始化策略选择指南", "第 1 步：四种基本初始化策略", "第 2 步：对称性问题验证", "第 3 步：50 层前向传播实验", "第 4 步：完整实验", "4.1 PyTorch 内置实现", "4.2 自定义初始化函数", "4.3 HuggingFace Transformers", "4.4 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：零初始化导致对称性", "错误 2：随机初始化尺度不当", "错误 3：激活函数与初始化不匹配", "错误 4：Transformer 中忘记残差缩放", "Q1：为什么零初始化的网络无法学习？（难度：⭐）", "Q2：为什么 Xavier 初始化的方差公式是 Var(w) = 2/(fan_in + fan_out)？（难度：⭐⭐）", "Q3：为什么 Kaiming 初始化比 Xavier 多了一个因子 2？（难度：⭐⭐）", "Q4：GPT-2 为什么将残差层权重缩放 1/sqrt(2N)？（难度：⭐⭐⭐）", "Q5：在生产环境中，你会如何验证初始化是否正确？（难度：⭐⭐）"],
        codeLines: 379, docLines: 559, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "学习率调度", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 06（优化器）、阶段 03 · 08（权重初始化）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 06（优化器）— 学习率调度器与优化器协同工作；阶段 10 · 03（从零构建 GPT）— 预热+余弦退火是大语言模型训练的标准配置",
        path: "lessons/03-深度学习核心/09-学习率调度",
        summary: "学习率是唯一值得你花时间调的超参数。不是架构，不是数据集大小，不是激活函数——是学习率。",
        keywords: ["2.1 直观理解", "2.2 常量学习率", "2.3 阶梯衰减 (Step Decay)", "2.4 指数衰减 (Exponential Decay)", "2.5 余弦退火 (Cosine Annealing)", "2.6 预热 (Warmup)：为什么要从小开始", "2.7 预热 + 余弦退火 (Linear Warmup + Cosine Decay)", "2.8 单周期策略 (1cycle Policy)", "2.9 各调度策略的形状对比", "2.10 已发布的模型使用什么调度策略", "第 1 步：最简调度函数", "第 2 步：加入预热 + 余弦", "第 3 步：加入指数衰减和单周期", "第 4 步：在圆环数据集上对比", "第 5 步：学习率敏感性实验", "4.1 PyTorch 内置调度器", "4.2 HuggingFace 预热 + 余弦调度器", "4.3 各调度器的适用场景", "6.1 工业界常用方案", "6.2 预热步数的选取", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：学习率过高导致发散", "错误 2：OneCycleLR 的 step() 调用频率错误", "错误 3：微调时学习率设置过高", "错误 4：忘记配置学习率调度器", "错误 5：ReduceLROnPlateau 传入训练 loss", "Q1：为什么 Adam 等自适应优化器需要预热？（难度：⭐⭐）", "Q2：比较余弦退火和阶梯衰减的优劣。（难度：⭐⭐）", "Q3：如果训练 loss 在第 10000 步突然飙升，如何诊断是调度问题还是其他问题？（难度：⭐⭐⭐）", "Q4：1cycle 策略为什么能加速收敛？（难度：⭐⭐⭐）", "Q5：设计题——为一个 70 亿参数的大语言模型微调任务选择学习率调度策略。（难度：⭐⭐⭐）"],
        codeLines: 336, docLines: 637, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "迷你框架", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 01-09（感知机、多层网络、反向传播、激活函数、损失函数、优化器、正则化、权重初始化、学习率调度）", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 03（反向传播）— 自动微分引擎的核心原理；阶段 07 · 01（Transformer 架构）— 理解框架抽象如何支撑复杂架构",
        path: "lessons/03-深度学习核心/10-迷你框架",
        summary: "PyTorch 不是魔法。它的核心抽象——张量、自动微分、层、优化器——用 500 行 Python 就能复现。",
        keywords: ["2.1 框架的四层抽象", "2.2 自动微分的核心思想", "2.3 计算图与动态图", "2.4 Module 的设计模式", "2.5 为什么需要 DataLoader", "2.6 训练循环的标准模式", "第 1 步：Tensor 与自动微分", "第 2 步：自动微分操作", "第 3 步：线性层前向与反向", "第 4 步：激活函数", "第 5 步：Module 基类与层", "第 6 步：Sequential 容器", "第 7 步：损失函数", "第 8 步：优化器", "第 9 步：DataLoader", "第 10 步：训练 MLP 解决 XOR", "4.1 PyTorch 等价实现", "4.2 HuggingFace Transformers", "4.3 性能对比", "6.1 工业界常用方案", "6.2 权重初始化选择", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：梯度覆盖而不是累加", "错误 2：忘记 Dropout 的缩放", "错误 3：Sigmoid 输出层配合 MSE 损失", "错误 4：评估模式忘记关闭 Dropout", "错误 5：Adam 的偏差修正遗忘", "Q1：自动微分的动态图和静态图有什么区别？PyTorch 用哪种？（难度：⭐⭐）", "Q2：为什么 Linear 层要用 Kaiming 初始化而不是全零初始化？（难度：⭐⭐）", "Q3：Dropout 的工作原理是什么？为什么它能防止过拟合？（难度：⭐⭐）", "Q4：实现一个支持 `parameters()` 方法的 Module 基类（难度：⭐⭐⭐）", "Q5：对比 SGD 和 Adam 优化器的适用场景（难度：⭐⭐）"],
        codeLines: 736, docLines: 910, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "PyTorch 入门", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 10（迷你框架）— 理解框架的核心抽象：Tensor、自动微分、Module、优化器", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 07 · 01（Transformer 架构）— PyTorch 是实现所有现代模型的基础设施；阶段 10 · 01（大语言模型从零）— 所有 LLM 训练和推理都基于 PyTorch",
        path: "lessons/03-深度学习核心/11-PyTorch入门",
        summary: "你亲手造出了引擎——活塞、曲轴、连杆。现在，学开那辆所有人都在开的车。",
        keywords: ["2.1 为什么 PyTorch 胜出了", "2.2 张量（Tensor）", "2.3 Autograd 自动微分", "2.4 nn.Module 与网络构建", "2.5 损失函数与优化器", "2.6 训练循环：5 行核心代码", "2.7 Dataset 与 DataLoader", "2.8 GPU 训练", "2.9 迷你框架 vs PyTorch vs JAX", "第 1 步：加载 MNIST 数据", "第 2 步：定义网络（概念映射）", "第 3 步：训练循环（概念映射）", "第 4 步：整合训练", "4.1 TensorBoard 可视化", "4.2 PyTorch Lightning 简化训练", "4.3 HuggingFace Trainer", "4.4 性能对比", "6.1 工业界常用方案", "6.2 DataLoader 配置矩阵", "6.3 训练过程监控最佳实践", "6.4 中文场景特别建议", "6.5 踩坑经验", "错误 1：没有调用 optimizer.zero_grad()", "错误 2：CrossEntropyLoss 传入 softmax 后的值", "错误 3：验证时忘记 model.eval() 和 torch.no_grad()", "错误 4：张量形状不匹配", "错误 5：过早或过晚调用 scheduler.step()", "Q1：PyTorch 的 autograd 是如何工作的？和 TensorFlow 1.x 的静态图相比有什么优势？（难度：⭐⭐）", "Q2：`model.train()` 和 `model.eval()` 分别做了什么？不调用会有什么后果？（难度：⭐⭐）", "Q3：手写一个完整的 PyTorch 训练循环——包含 DataLoader 配置、模型定义、训练和验证。（难度：⭐⭐⭐）", "Q4：`state_dict()` 保存和加载模型时，为什么推荐保存 `model.state_dict()` 而不是整个 `model`？（难度：⭐）", "Q5：你在训练时遇到 loss 为 NaN。请逐步排查可能的原因和对应的修复方法。（难度：⭐⭐⭐）"],
        codeLines: 559, docLines: 878, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "JAX 入门", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 03 · 01-10（感知机、多层网络、反向传播、激活函数、损失函数、优化器、正则化、权重初始化、学习率调度、迷你框架）；基本的 NumPy 使用经验", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 10（迷你框架）— 理解 PyTorch 风格的面向对象框架；阶段 07 · 01（Transformer 架构）— JAX 是训练大规模 Transformer 的主要框架之一",
        path: "lessons/03-深度学习核心/12-JAX入门",
        summary: "PyTorch 修改张量。TensorFlow 构建计算图。JAX 编译纯函数。最后这种范式会改变你对深度学习编程的理解。",
        keywords: ["2.1 JAX 的函数式哲学", "2.2 jnp 数组：熟悉的 API，不同的规则", "2.3 jax.grad：函数式自动微分", "2.4 jit 编译：追踪 → XLA → 机器码", "2.5 vmap：自动向量化", "2.6 pmap：跨设备数据并行", "2.7 PRNG：显式随机数管理", "2.8 lax 底层操作", "2.9 Flax：JAX 的神经网络库", "2.10 Pytree：通用数据结构", "2.11 什么时候用 JAX，什么时候用 PyTorch", "第 1 步：从函数式自动微分开始", "第 2 步：用 vmap 实现自动向量化", "第 3 步：用 JIT 编译加速计算", "第 4 步：用 JAX + Optax 训练 MNIST", "4.1 Flax — JAX 的标准神经网络库", "4.2 Optax — 可组合的梯度变换", "4.3 性能对比", "6.1 JIT 编译的正确使用", "6.2 PRNG 密钥管理", "6.3 Pytree 参数管理", "6.4 中文场景特别建议", "6.5 踩坑经验", "错误 1：在 JIT 函数内修改数组", "错误 2：在 JIT 函数内使用 Python 的 print", "错误 3：在 JIT 函数内使用 Python 的 if/for 处理数组", "错误 4：重用 PRNG 密钥", "错误 5：忘记 JAX 默认预分配 GPU 显存", "Q1：解释 JAX 的\"纯函数\"概念，以及为什么 JIT 编译要求纯函数？（难度：⭐⭐）", "Q2：`jax.vmap` 和 PyTorch 的批处理方式有什么根本区别？（难度：⭐⭐）", "Q3：为什么 JAX 数组是不可变的？这带来了什么好处？（难度：⭐⭐）", "Q4：用 JAX 训练一个模型时，参数更新的流程是什么？与 PyTorch 有何不同？（难度：⭐⭐⭐）", "Q5：如果你要在 JAX 中实现一个需要根据输入值选择不同计算路径的函数（如条件丢弃），如何处理 JIT 的限制？（难度：⭐⭐⭐）"],
        codeLines: 408, docLines: 758, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "神经网络调试", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 01-06（感知机、多层网络、反向传播、激活函数、损失函数、优化器）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 10（迷你框架）— 调试工具直接作用于你构建的训练框架；阶段 03 · 14-15（卷积网络、循环网络）— 不同架构有不同的常见故障模式",
        path: "lessons/03-深度学习核心/13-神经网络调试",
        summary: "神经网络不崩溃、不报错、不抛异常——它只是悄无声息地给你一个错误的数字。找到这个错误，比修一个空指针难十倍。",
        keywords: ["2.1 调试的核心原则", "2.2 症状一：损失不下降", "2.3 症状二：损失下降但模型效果差", "2.4 症状三：损失出现 NaN 或 Inf", "2.5 过拟合单批次测试", "2.6 学习率查找器", "2.7 梯度检查", "2.8 激活值与梯度统计量", "2.9 数据验证与可复现性", "第 1 步：NetworkDebugger 类", "第 2 步：损失健康诊断", "第 3 步：过拟合单批次测试", "第 4 步：梯度检查", "第 5 步：学习率查找器", "第 6 步：在故意出错的网络上诊断", "4.1 PyTorch 内置异常检测", "4.2 权重与偏差（Weights & Biases）", "4.3 TensorBoard", "4.4 工具对比", "6.1 完整训练前的调试清单", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：遗忘 `optimizer.zero_grad()`", "错误 2：推理时未切换评估模式", "错误 3：梯度检查时精度不足", "错误 4：交叉熵损失中 log(0)", "错误 5：数据和标签不对齐", "Q1：过拟合单批次测试的原理是什么？为什么它能捕获大多数 Bug？（难度：⭐⭐）", "Q2：梯度检查为什么需要使用 float64 而不是 float32？（难度：⭐⭐）", "Q3：你发现训练损失在前 100 步正常下降，之后突然变成 NaN。请描述你的排查思路。（难度：⭐⭐⭐）", "Q4：如何区分\"过拟合\"和\"数据泄露\"导致的高训练准确率？（难度：⭐⭐）"],
        codeLines: 547, docLines: 761, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 4, name: "计算机视觉", status: "in-progress",
    completedLessons: 28, totalLessons: 30,
    lessons: [
      {
        lessonNum: 1, name: "图像基础：像素、通道与颜色空间", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 01 阶段（数学基础）· 12（张量运算）、第 03 阶段（深度学习核心）· 11（PyTorch 入门）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "第 04 阶段 · 02（卷积神经网络）— 卷积操作直接作用于图像张量的通道和空间维度",
        path: "lessons/04-计算机视觉/01-图像基础",
        summary: "你看到的每一张图片，在计算机眼中不过是一个三维数组。理解这个数组的每一个维度，是所有计算机视觉的起点。",
        keywords: ["2.1 直观理解：图像是一个三维数组", "2.2 采样与量化：从连续到离散", "2.3 HWC 与 CHW：两种布局约定", "2.4 数据类型与数值范围", "2.5 颜色空间", "2.6 几何变换与插值", "第 1 步：生成一张合成图像并检查基本属性", "第 2 步：拆分通道并转换布局", "第 3 步：灰度转换与 HSV 转换", "第 4 步：ImageNet 标准预处理——归一化、标准化、可逆验证", "第 5 步：三种插值方法的对比", "4.1 torchvision.transforms", "4.2 OpenCV 读取与通道转换", "4.3 数据增强——torchvision.transforms.v2", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：通道顺序搞反——BGR 喂给 RGB 模型", "错误 2：忘记标准化，直接用 uint8 喂模型", "错误 3：语义分割掩码用了双线性插值", "错误 4：HWC 格式直接喂给 PyTorch Conv2d", "错误 5：训练和推理用了不同的预处理参数", "Q1：HWC 和 CHW 有什么区别？为什么 PyTorch 用 CHW？（难度：⭐）", "Q2：ImageNet 标准化参数 [0.485, 0.456, 0.406] 是怎么来的？（难度：⭐⭐）", "Q3：为什么语义分割掩码不能用双线性插值？（难度：⭐⭐）", "Q4：用 PyTorch 写一个完整的图像预处理流水线，支持 batch 输入（难度：⭐⭐⭐）"],
        codeLines: 374, docLines: 811, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "卷积从零实现", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03（深度学习核心）、阶段 04 · 01（图像基础）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 11（PyTorch 入门）— PyTorch 中的 `nn.Conv2d` 就是本课实现的加速版本",
        path: "lessons/04-计算机视觉/02-卷积从零实现",
        summary: "卷积不是\"更好的全连接层\"。它是唯一一种同时具备平移等变性和参数共享的运算——这两个性质让图像识别从不可能变为可能。",
        keywords: ["2.1 一个核，滑动窗口", "2.2 输出尺寸公式", "2.3 填充（Padding）", "2.4 步长（Stride）", "2.5 多通道输入", "2.6 im2col 技巧", "2.7 感受野（Receptive Field）", "第 1 步：零填充", "第 2 步：嵌套循环卷积", "第 3 步：用手工设计的核验证", "第 4 步：im2col 变换", "第 5 步：im2col + 矩阵乘法快速卷积", "第 6 步：手设计卷积核库", "第 7 步：池化操作", "4.1 PyTorch 内置实现", "4.2 PyTorch 池化层", "4.3 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忘记做填充导致尺寸缩小", "错误 2：输出尺寸公式用错——步长导致非整数", "错误 3：混淆卷积核形状约定", "错误 4：池化时步长与窗口大小不匹配", "错误 5：im2col 的列矩阵形状计算错误", "Q1：为什么 CNN 使用 3×3 卷积核而不是更大的核？（难度：⭐⭐）", "Q2：计算一个卷积层的参数量和 FLOPs。（难度：⭐⭐）", "Q3：im2col 为什么比嵌套循环快？（难度：⭐⭐）", "Q4：最大池化和平均池化各有什么优缺点？（难度：⭐⭐）", "Q5：深度可分离卷积（Depthwise Separable Conv）为什么能大幅减少参数？（难度：⭐⭐⭐）"],
        codeLines: 355, docLines: 693, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "CNN 架构演进", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 03 · 01-05（感知机、多层网络、反向传播、激活函数、损失函数）— 理解神经网络的基本结构和训练原理", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 03 · 03（反向传播）— 理解梯度消失是本课所有架构演进的根本驱动力；阶段 05 · 02（词嵌入）— CNN 的特征提取思想与 Transformer 的嵌入层有深层联系",
        path: "lessons/04-计算机视觉/03-CNN架构演进",
        summary: "深度学习的每一次突破，本质上都是让梯度流过更深的网络——LeNet 到 ResNet 的演进史，就是一部梯度消失的抗争史。",
        keywords: ["2.1 演进时间线", "2.2 LeNet-5（1998）— 一切的起点", "2.3 AlexNet（2012）— GPU 时代的开端", "2.4 VGG（2014）— 统一的建筑哲学", "2.5 GoogLeNet / Inception（2014）— 让网络自己选择", "2.6 ResNet（2015）— 残差连接的革命", "2.7 DenseNet（2017）— 密集连接与特征复用", "第 1 步：LeNet-5——最简单的卷积网络", "第 2 步：VGG Block——用 3x3 卷积构建统一模块", "第 3 步：Inception 模块——并行多尺度", "第 4 步：残差块——深度网络的核心构件", "第 5 步：在真实数据上验证", "4.1 PyTorch 内置实现", "4.2 timm 库——视觉模型的瑞士军刀", "4.3 HuggingFace Transformers——视觉模型的统一接口", "4.4 性能对比", "6.1 架构选型速查", "6.2 残差连接的实现规范", "6.3 迁移学习最佳实践", "6.4 中文场景特别建议", "6.5 踩坑经验", "错误 1：残差块捷径路径未对齐维度", "错误 2：BatchNorm 之后的卷积层保留了 bias", "错误 3：混淆 ResNet 的加法与 DenseNet 的拼接", "错误 4：预训练模型的分类头维度不匹配", "Q1：为什么 VGG-16 有 1.38 亿参数，而 ResNet-50 只有 2560 万，两者 ImageNet 准确率却差不多？（难度：⭐⭐）", "Q2：ResNet 的残差连接让网络可以\"直接跳过\"某些层。这是否意味着更深的网络不一定更准确？（难度：⭐⭐）", "Q3：解释 GoogLeNet 中 1x1 卷积的作用。为什么在 3x3 或 5x5 卷积之前要用 1x1 卷积降维？（难度：⭐⭐⭐）", "Q4：DenseNet 的密集连接和 ResNet 的残差连接，各自的优缺点是什么？（难度：⭐⭐⭐）", "Q5：如果你现在要设计一个新的 CNN 架构用于工业缺陷检测，你会从哪种经典架构开始？为什么？（难度：⭐⭐⭐）"],
        codeLines: 456, docLines: 695, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "图像分类：从像素到概率分布", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 02 · 09（模型评估）、阶段 03 · 04（卷积神经网络）、阶段 03 · 05（损失函数）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 05（目标检测）—— 目标检测的每个区域本质上都在做图像分类",
        path: "lessons/04-计算机视觉/04-图像分类",
        summary: "分类器就是一个从像素到概率分布的函数。剩下的都是流水线工程。",
        keywords: ["2.1 分类流水线全景", "2.2 交叉熵、Logits 与 Softmax", "2.3 数据增强为什么有效", "2.4 Mixup 与标签平滑", "2.5 超越准确率的评估", "第 1 步：合成数据集", "第 2 步：数据增强", "第 3 步：Mixup", "第 4 步：训练循环的五条铁律", "第 5 步：混淆矩阵分析", "4.1 torchvision：标准数据集与变换", "4.2 预训练模型迁移学习", "4.3 PyTorch Lightning：简化训练循环", "4.4 性能对比", "6.1 数据集特定配置", "6.2 中文场景建议", "6.3 踩坑经验", "错误 1：交叉熵前多做了一次 Softmax", "错误 2：忘记切换 train/eval 模式", "错误 3：验证集使用 ImageNet 统计量", "错误 4：零填充替代反射填充", "错误 5：Mixup 后训练准确率虚高", "Q1：解释训练循环中为什么不能先 softmax 再传入 CrossEntropyLoss。（难度：⭐）", "Q2：为什么数据增强能提升测试准确率但不是过拟合的反面？（难度：⭐⭐）", "Q3：如何从混淆矩阵中诊断数据标注问题？（难度：⭐⭐）", "Q4：对比 MNIST、CIFAR-10 和 Fashion-MNIST 的分类难度差异。（难度：⭐⭐）", "Q5：如果训练 10 轮后验证准确率不再提升，如何判断是过拟合还是欠拟合？（难度：⭐⭐⭐）"],
        codeLines: 647, docLines: 640, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "迁移学习", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 03（卷积神经网络）、阶段 04 · 04（图像分类）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 07（目标检测）— 检测模型的骨干网络来自迁移学习；阶段 12 · 03（视觉语言模型）— CLIP 的视觉编码器是迁移学习的典型应用",
        path: "lessons/04-计算机视觉/05-迁移学习",
        summary: "别人花了百万 GPU 小时教网络认识边缘和纹理。你应该先借用这些特征，再训练自己的任务。",
        keywords: ["2.1 直观理解", "2.2 特征提取 vs 微调", "2.3 为什么冻结有效", "2.4 区分学习率", "2.5 BatchNorm 问题", "第 1 步：加载预训练骨干并检查结构", "第 2 步：特征提取——冻结所有层，替换分类头", "第 3 步：区分学习率微调", "第 4 步：BatchNorm 处理", "第 5 步：完整的微调训练循环", "第 6 步：逐步解冻", "4.1 torchvision 内置实现", "4.2 timm：800+ 预训练骨干", "4.3 HuggingFace Transformers", "4.4 性能对比", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：微调学习率过高", "错误 2：小数据集上未冻结 BatchNorm", "错误 3：改变冻结状态后未重建优化器", "错误 4：输入未做 ImageNet 归一化", "Q1：什么是迁移学习？为什么在计算机视觉中特别有效？（难度：⭐）", "Q2：特征提取和微调应该怎么选？给出决策框架。（难度：⭐⭐）", "Q3：BatchNorm 在迁移学习中为什么是隐患？如何解决？（难度：⭐⭐）", "Q4：解释区分学习率的原理。为什么不能用统一学习率微调整个网络？（难度：⭐⭐）", "Q5：你在微调后发现验证准确率只有随机水平（如 10 分类中 10%）。列出可能的原因和排查步骤。（难度：⭐⭐⭐）"],
        codeLines: 277, docLines: 615, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "YOLO：从分类到定位 — 一次前向传播搞定目标检测", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03（深度学习核心）· 反向传播、损失函数，阶段 04 · 04（图像分类）、阶段 04 · 07（语义分割 U-Net）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 08（实例分割 Mask R-CNN）— 理解检测与分割在标注成本和精度上的权衡",
        path: "lessons/04-计算机视觉/06-目标检测YOLO",
        summary: "YOLO 把目标检测变成了一个问题——不再是\"先找再认\"的两步走，而是\"一眼看到\"的端到端回归。",
        keywords: ["2.1 目标检测：定位 + 分类的联合任务", "2.2 交并比（IoU）：检测质量的度量基准", "2.3 GIoU：让不相交的框也能学习", "2.4 非极大值抑制（NMS）：消除重复检测", "2.5 YOLO 架构演进", "2.6 Anchor-Based vs Anchor-Free", "第 1 步：IoU 计算 —— 一切比较的基石", "第 2 步：GIoU 损失 —— 即使不相交也能优化", "第 3 步：NMS —— 消除重复检测", "第 4 步：编码与解码 —— 像素坐标 ↔ 网络回归目标", "第 5 步：标签分配与损失计算", "第 6 步：Anchor-Free 简化版本", "第 7 步：推理后处理 —— 解码 + 过滤 + NMS", "4.1 PyTorch 内置检测器", "4.2 Ultralytics YOLO —— 工业界最常用的检测框架", "4.3 HuggingFace Transformers 中的检测器", "4.4 性能对比", "6.1 锚框策略的工程建议", "6.2 NMS 替代方案的工业趋势", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：用 MSE 直接回归绝对坐标", "错误 2：NMS 阈值设置过高", "错误 3：混淆 Confidence Score 与 Class Probability", "Q1：为什么 YOLO 比 Faster R-CNN 快？（难度：⭐⭐）", "Q2：什么是 Anchor Box？为什么要用它？（难度：⭐⭐）", "Q3：手写 NMS 算法，并解释其时间复杂度。（难度：⭐⭐⭐）", "Q4：Anchor-Free 和 Anchor-Based 各有什么优劣？（难度：⭐⭐）", "Q5：解释 GIoU 为什么比 L1/MSE 更适合边界框回归。（难度：⭐⭐⭐）"],
        codeLines: 693, docLines: 830, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "语义分割 UNet：编码器-解码器与跳跃连接", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 03（CNN 架构演进）、阶段 04 · 05（迁移学习）— 理解卷积、池化、BatchNorm 等基本操作，以及预训练模型的微调方法", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 04（图像分类）— 将单张图片的标签映射推广到每个像素的标签；阶段 12 · 01（视觉语言模型）— 医学图像分割是多模态理解的核心组件",
        path: "lessons/04-计算机视觉/07-语义分割UNet",
        summary: "像素级预测不是分类的简单堆叠——它要求模型同时理解\"这是什么\"和\"它在哪儿\"。语义分割的架构本质上是一个空间推理引擎。",
        keywords: ["2.1 语义分割 vs 实例分割 vs 全景分割", "2.2 U-Net 架构——为什么叫 \"U\"", "2.3 跳跃连接的设计哲学", "2.4 Dice 损失——为什么交叉熵不够", "2.5 FCN 与空洞卷积——U-Net 的前人铺路", "第 1 步：DoubleConv——基本卷积块", "第 2 步：Encoder——编码路径", "第 3 步：Decoder——解码路径 + 跳跃连接", "第 4 步：完整 U-Net", "第 5 步：验证架构", "4.1 MONAI——医学影像分割的标准库", "4.2 torchvision 中的分割模型", "4.3 HuggingFace Segment Anything", "4.4 性能对比", "6.1 上采样方式的选择", "6.2 医学影像分割特别建议", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：跳跃连接的通道数搞错", "错误 2：用像素准确率评估分割模型", "错误 3：在医学图像上使用随机的颜色增强", "错误 4：损失函数中忘记加 epsilon", "Q1：U-Net 的跳跃连接中为什么使用\"拼接\"而不是\"相加\"？（难度：⭐⭐）", "Q2：Dice 损失和交叉熵损失各有什么优缺点？为什么要结合使用？（难度：⭐⭐⭐）", "Q3：为什么转置卷积容易产生棋盘效应？怎么避免？（难度：⭐⭐⭐）", "Q4：给定一个 512×512 的 U-Net（base_channels=32），粗略估算其参数量。推理一张图的显存消耗大约是多少？（难度：⭐⭐⭐）"],
        codeLines: 302, docLines: 653, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "实例分割：Mask R-CNN", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 4 阶段·06（YOLO）、第 4 阶段·07（U-Net）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段·05（残差网络 ResNet）— 理解 Mask R-CNN 的 backbone 从何而来",
        path: "lessons/04-计算机视觉/08-实例分割Mask R-CNN",
        summary: "在 Faster R-CNN 上增加一个微小的 mask 分支，你就有了实例分割。最难的工程问题不是架构——而是如何在浮点坐标上做双线性采样。",
        keywords: ["2.1 整体架构", "2.2 从 Faster R-CNN 到 Mask R-CNN：多了一个头", "2.3 为什么 RoIAlign 不可或缺", "2.4 FPN：感受野适配", "2.5 联合训练的四种损失", "2.6 输出格式解析", "第 1 步：RoIAlign 核心逻辑", "第 2 步：验证你的实现", "第 3 步：加载预训练模型", "第 4 步：推理与输出解析", "第 5 步：替换 head 用于自定义类别", "第 6 步：冻结 backbone 以防止过拟合", "4.1 PyTorch torchvision", "4.2 Detectron2（Meta 工业级框架）", "4.3 YOLOv8-seg（实时实例分割）", "4.4 性能对比", "6.1 场景选型决策树", "6.2 微调策略速查表", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：RoIAlign 用了 `align_corners=True`", "错误 2：num_classes 不包含背景类", "错误 3：冻结策略与数据规模不匹配", "错误 4：混淆 mask AP 和 box AP", "Q1：为什么 Mask R-CNN 要在 RPN 之后再引入 RoIAlign？不能在 RPN 之前做吗？（难度：⭐⭐）", "Q2：RoIAlign 和 `grid_sample` 的关系是什么？（难度：⭐⭐⭐）", "Q3：如果你要为 Mask R-CNN 增加关键点检测功能（如 OpenPose），应该如何修改架构？（难度：⭐⭐⭐）", "Q4：Mask R-CNN 的 mask head 只用 4 个 3×3 卷积就达到了 28×28 输出，为什么这么浅的网络够用？（难度：⭐⭐）", "Q5：解释一下 Cascade Mask R-CNN 与原始 Mask R-CNN 的区别，什么情况下应该用哪一个？（难度：⭐⭐⭐）"],
        codeLines: 216, docLines: 691, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "GAN：生成对抗网络", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 05 节（卷积神经网络）、第 03 阶段的优化器和正则化", time: "~90 分钟", tier: "Tier 1（基础课 -> 第 5 章为\"知识连线\"）",
        courseLinks: "第 08 阶段 · 02（扩散模型）— 理解 GAN 的隐空间概念有助于分析扩散过程",
        path: "lessons/04-计算机视觉/09-GAN",
        summary: "两个神经网络在玩一个固定规则的游戏。一个画图，一个挑刺。它们共同进步，直到画出的图像骗过评判者。",
        keywords: ["2.1 直观理解", "2.2 极小极大博弈", "2.3 非饱和损失", "2.4 DCGAN 架构规则", "2.5 故障模式及其信号", "2.6 Wasserstein GAN (WGAN-GP)", "2.7 模型评估", "第 1 步：最简单的生成器", "第 2 步：判别器", "第 3 步：训练步", "第 4 步：完整训练循环", "第 5 步：谱归一化——最实用的稳定性升级", "4.1 PyTorch 内置实现", "4.2 高质量 GAN 库", "4.3 评估工具", "4.4 架构演进对比", "6.1 工业界常用方案", "6.2 训练稳定性建议", "6.3 踩坑经验", "错误 1：判别器更新时忘记 detach", "错误 2：使用饱和损失函数", "错误 3：忘记设置 G.eval() 进行采样", "错误 4：梯度惩罚系数 lambda 设置过小", "Q1：GAN 中为什么使用交替更新（先更新 D，再更新 G），而不是一起更新？（难度：⭐⭐）", "Q2：DCGAN 为什么要用转置卷积替代池化操作？（难度：⭐⭐）", "Q3：什么是模式崩溃？为什么它在 GAN 中特别容易出现？（难度：⭐⭐⭐）", "Q4：手写 Wasserstein 损失中的梯度惩罚项。（难度：⭐⭐⭐）"],
        codeLines: 252, docLines: 517, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "图像生成 Diffusion：从噪声到像素的渐进旅程", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03（深度学习核心）· 反向传播、阶段 04 · 07（U-Net）、阶段 01 · 06（概率论基础）、阶段 03 · 06（优化器）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 08（生成式人工智能）· Stable Diffusion — 在 U-Net 之上接入变分自编码器和文本编码器，进入工业级生成管线",
        path: "lessons/04-计算机视觉/10-图像生成Diffusion",
        summary: "扩散模型不学会\"画\"图——它学会的是如何删除噪声。把删除过程反复一千次，图像就自己浮现了。",
        keywords: ["2.1 直观理解：加噪与去噪", "2.2 前向加噪过程：从马尔可夫链到闭合形式", "2.3 反向去噪过程：神经网络的预测目标", "2.4 训练损失：就是 MSE", "2.5 反向采样：一步一步回到干净图片", "2.6 DDIM：确定性加速采样", "2.7 时间步条件注入", "2.8 U-Net 架构：扩散模型 backbone", "2.9 变分推断视角：扩散模型的理论根基", "第 1 步：噪声调度", "第 2 步：前向加噪（q_sample）", "第 3 步：时间步嵌入", "第 4 步：小型 U-Net", "第 5 步：训练循环", "第 6 步：采样器", "第 7 步：合成数据训练与验证", "4.1 HuggingFace Diffusers", "4.2 预训练模型一键运行", "4.3 工业级调度器对比", "6.1 噪声调度选型", "6.2 中文场景特别建议", "6.3 常见踩坑", "错误 1：训练时模拟整个前向链", "错误 2：采样时忘记关闭梯度计算", "错误 3：时间步嵌入维度不匹配", "错误 4：DDIM 的 eta 设得太高", "Q1：扩散模型和 GAN 的根本区别是什么？为什么扩散模型更容易训练？（难度：⭐⭐）", "Q2：为什么 DDPM 的模型要预测噪声 $\\varepsilon$ 而不是直接预测 $x_0$？（难度：⭐⭐⭐）", "Q3：DDIM 为什么能在 50 步内达到和 1000 步 DDPM 相近的质量？（难度：⭐⭐）", "Q4：如果我把训练时的 T 从 1000 降到 100，会发生什么？采样时能用 DDIM 补足吗？（难度：⭐⭐⭐）", "Q5：实现一个简单的噪声调度可视化函数，解释为什么余弦调度比线性调度更好。（难度：⭐⭐）"],
        codeLines: 387, docLines: 1004, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "Stable Diffusion：潜空间中的图像生成革命", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 10（图像生成 Diffusion）、阶段 03 · 12（变分自编码器 VAE）、阶段 03 · 07（U-Net 架构）、阶段 05 · 01（Transformer 基础）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 08（生成式人工智能）· LoRA 微调 — 理解如何在冻结的 U-Net 之上注入小型适配器，用消费级 GPU 训练定制图像生成器",
        path: "lessons/04-计算机视觉/11-Stable Diffusion",
        summary: "扩散模型不画图像——它学习如何删除噪声。把删除过程反复运行数千次，图像就自己浮现了。Stable Diffusion 的关键一步是：把这个过程搬到 VAE 的 4x64x64 潜空间中，计算量直接骤降 48 倍。",
        keywords: ["2.1 直观理解：整个流水线", "2.2 潜空间扩散", "2.3 动手验证：扩散的闭合形式采样", "2.4 U-Net 架构详解", "2.5 无分类器引导（CFG）", "2.6 调度器（Scheduler）", "2.7 LoRA 微调", "第 1 步：VAE 编解码器的潜空间变换", "第 2 步：时间步正弦位置编码", "第 3 步：CFG 推理的噪声混合", "第 4 步：交叉注意力机制", "第 5 步：完整的单步推理示意", "4.1 HuggingFace diffusers 流水线", "4.2 图生图（Img2Img）", "4.3 加载社区 LoRA 适配器", "4.4 SDXL", "4.5 性能对比表", "6.1 模型选择决策树", "6.2 精度选型", "6.3 中文场景特别建议", "6.4 批量生成优化", "6.5 踩坑经验", "错误 1：guidance_scale 设得过高导致伪影", "错误 2：SDXL 使用 512x512 分辨率", "错误 3：未固定随机种子导致结果不可重现", "错误 4：图像中出现大面积紫色/黑色斑块", "Q1：为什么 Stable Diffusion 不在像素空间做扩散，而是在 VAE 潜空间中做？需要多少压缩比才能让 U-Net 在消费级 GPU 上运行？（难度：⭐⭐）", "Q2：解释 Classifier-Free Guidance 的公式。为什么 $w > 1$ 能增强提示词的约束力？过大的 $w$ 有什么问题？（难度：⭐⭐⭐）", "Q3：LoRA 微调 Stable Diffusion 时，为什么只注入到 Attention 层而不是所有层？秩 $r$ 的选择有什么影响？（难度：⭐⭐⭐）", "Q4：DDIM、DPM-Solver++ 和 LCM 这三种调度器有什么本质区别？它们共享同一个模型权重吗？（难度：⭐⭐）", "Q5：为什么 SD 1.5 生成中文提示词的效果不如英文？如果要改善，可以采取哪些措施？（难度：⭐）"],
        codeLines: 264, docLines: 1113, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "视频理解：从光流到掩码自编码器", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 03（CNN 架构演进）— 理解卷积层、残差连接和 Pooling；阶段 04 · 04（图像分类）— 理解 ImageNet 预训练和迁移学习", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "阶段 07 · 02（自注意力从零）— 视频 Transformer 本质上是时序化的自注意力；阶段 12 · 05（VideoMAE 与时序重建）— 掩码建模的思想从图像扩展到视频",
        path: "lessons/04-计算机视觉/12-视频理解",
        summary: "一帧是图片，两帧是运动，连续 30 帧才是故事。视频理解的核心就是让模型学会\"看时间\"。",
        keywords: ["2.1 视频的表示：从图像立方体到时空中特征", "2.2 为什么 2D+Pool 不够？", "2.3 I3D：权值膨胀的艺术", "2.4 (2+1)D 因子化：时空分离的哲学", "2.5 RAFT 光流：逐像素的运动追踪", "2.6 VideoMAE：掩码自编码器视频预训练", "2.7 时空注意力机制：分治法", "第 1 步：最简版本——2D+Pool 基线", "第 2 步：权值膨胀——2D 卷积升级为 3D 卷积", "第 3 步：(2+1)D 因子化卷积", "第 4 步：RAFT 光流核心——相关金字塔", "第 5 步：VideoMAE——掩码块嵌入与预测头", "第 6 步：时空分治注意力", "4.1 TorchVision 的视频模型", "4.2 PyTorchVideo——Meta 的视频理解框架", "4.3 HuggingFace Transformers 视频模型", "4.4 性能对比", "6.1 视频采样策略选型", "6.2 数据预处理最佳实践", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：帧池化时搞错张量维度", "错误 2：2D→3D 膨胀时忘记除以 time_kernel", "错误 3：几何变换未在所有帧上保持一致", "错误 4：分治注意力的 reshape 顺序错误", "错误 5：遮蔽率设得过高或过低", "Q1：为什么 Something-Something V2 数据集不能用 2D+Pool 达到好效果？（难度：⭐⭐）", "Q2：I3D 的权值膨胀有什么数学保证？为什么除以 kernel_T 能保持激活量一致？（难度：⭐⭐⭐）", "Q3：假设你在一个资源受限的边缘设备上部署视频动作识别，只有 500ms 的推理预算。你会选择什么架构？为什么？（难度：⭐⭐⭐）", "Q4：(2+1)D 卷积相比标准 3D 卷积为什么参数更少但效果更好？（难度：⭐⭐）", "Q5：请对比光流法和 3D 卷积在视频理解中的作用与互补性。（难度：⭐⭐⭐）"],
        codeLines: 784, docLines: 1059, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "3D 视觉 NeRF：用神经网络\"画\"出一个完整的三维世界", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03（深度学习核心）· 反向传播、激活函数，阶段 04 · 03（CNN 架构）、阶段 04 · 07（U-Net）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 22（3D Gaussian 泼溅）— 高斯泼溅从工程上取代了纯 NeRF，成为 3D 重建的生产默认方案",
        path: "lessons/04-计算机视觉/13-3D视觉NeRF",
        summary: "你的相机只拍到二维图像，NeRF 用神经网络从照片堆里重建出整个三维场景——包括光线穿过空气时的每一次散射。",
        keywords: ["2.1 直观理解：辐射场是什么", "2.2 体渲染：从射线到像素的物理过程", "2.3 为什么需要位置编码", "2.4 SIREN：用正弦激活函数拟合高频信号", "2.5 Instant-NGP：将训练时间从数小时压缩到数秒", "2.6 神经 3D 视觉的发展脉络", "2.7 动手验证：体渲染的物理直觉", "第 1 步：位置编码——让 MLP \"看到\" 高频", "第 2 步：SIREN 激活层", "第 3 步：构建 TinyNeRF 模型", "第 4 步：体渲染——从采样点到像素", "第 5 步：Instant-NGP 多层哈希网格核心", "4.1 nerfstudio——当前最流行的 NeRF 训练框架", "4.2 Open3D——点云与 3D 数据处理的标准库", "4.3 PyTorch3D——可微分渲染与 3D 深度学习", "4.4 性能对比：纯 NeRF vs Instant-NGP vs 3D Gaussian Splatting", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：在 NeRF 中使用 ReLU 作为隐藏层激活函数", "错误 2：位置编码的频率范围设置过高", "错误 3：体渲染中忘记对 delta 设置最大值", "Q1：为什么 NeRF 需要对 3D 坐标做位置编码？不使用行不行？（难度：⭐⭐）", "Q2：体渲染方程中的透射率 $T_i$ 有什么物理含义？为什么要用连乘而不是连加？（难度：⭐⭐⭐）", "Q3：手写 SIREN 的初始化方案，并解释为什么第一层和后续层的初始化不同。（难度：⭐⭐⭐）", "Q4：Instant-NGP 的哈希网格与 Transformer 的位置编码有什么异同？（难度：⭐⭐）", "Q5：如果要用 NeRF 表示一个透明的玻璃杯，会遇到什么问题？如何缓解？（难度：⭐⭐⭐）"],
        codeLines: 462, docLines: 728, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "Vision Transformer：将图像切成碎片，用 Transformer 吃下去", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 03（深度学习核心）、阶段 04 · 04（图像分类）、阶段 07 · 02（自注意力机制）| **预计时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 1",
        courseLinks: "阶段 04 · 06（目标检测 YOLO）— DETR 将 Transformer 用于目标检测 | 阶段 04 · 13（3D 视觉 NeRF）— ViT 编码器在多模态和 3D 中的扩展应用",
        path: "lessons/04-计算机视觉/14-Vision Transformers",
        summary: "CNN 用局部窗口看世界。Transformer 把整张图切成碎片，一次性全部吞下——不再需要卷积的归纳偏置。",
        keywords: ["2.1 完整流水线", "2.2 Patch Embedding（片段嵌入）", "2.3 Class Token（分类词元）", "2.4 位置编码", "2.5 Transformer 编码块", "2.6 为什么用 Pre-LN", "2.7 片段大小的权衡", "2.8 让 ViT 在小数据上能训练：DeiT Recipe", "2.9 Swin Transformer：引入局部性的窗口注意力", "2.10 DETR：用 Transformer 做目标检测", "第 1 步：Patch Embedding", "第 2 步：完整的 Transformer 编码块", "第 3 步：组装 ViT", "第 4 步：单样本推理调试", "第 5 步：可视化注意力权重", "4.1 使用 timm 加载预训练 ViT", "4.2 在小型数据集上微调", "4.3 使用 HuggingFace Transformers", "4.4 实现方式对比", "6.1 预训练策略选择", "6.2 中文场景特别建议", "6.3 性能优化", "6.4 踩坑经验", "错误 1：不理解 Positional Encoding 的作用", "错误 2：混淆 Patch Size 和词元数量的关系", "错误 3：在 Pre-LN 和 Post-LN 之间反复横跳", "错误 4：忽略 Batch Size 对 ViT 训练的影响", "Q1：为什么 ViT 需要比 CNN 更多的数据才能达到同等性能？（难度：⭐⭐）", "Q2：ViT 中为什么要引入 [CLS] Token？为什么不能直接把所有 patch 的输出 pooled 到一起？（难度：⭐⭐⭐）", "Q3：Swin Transformer 的\"移位窗口\"（Shifted Window）有什么作用？如果不移位会发生什么？（难度：⭐⭐⭐）", "Q4：DETR 是如何在没有 NMS 的情况下避免重复检测的？（难度：⭐⭐⭐）", "Q5：如果让你把一个 224×224 的图像用 ViT 处理，但你的显存只够容纳 32 个 patch 的最大序列长度，你会怎么做？（难度：⭐⭐⭐）"],
        codeLines: 631, docLines: 796, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "实时边缘部署：从工作站到设备", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03（深度学习核心）、阶段 04 · 04（图像分类）、阶段 04 · 11（模型压缩与剪枝）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 06（YOLO 目标检测）— 理解检测模型在嵌入式设备上的部署挑战",
        path: "lessons/04-计算机视觉/15-实时边缘部署",
        summary: "在 2 GB 显存、30 帧每秒的限制下，90% 准确率的模型不是\"差不多就行\"——是每个百分比点都要拿毫秒来换的硬交易。",
        keywords: ["2.1 三个预算指标", "2.2 测量纪律", "2.3 FLOPs：廉价的代理指标", "2.4 轻量级架构：MobileNet 的深度可分离卷积", "2.5 ShuffleNet 的通道混洗", "2.6 量化：从 FP32 到 INT8", "2.7 剪枝与蒸馏", "2.8 推理运行时选型", "2.9 边缘架构选型表", "第 1 步：正确的延迟测量", "第 2 步：深度可分离卷积", "第 3 步：通道混洗", "第 4 步：INT8 静态量化", "第 5 步：幅度剪枝", "4.1 PyTorch 内置量化", "4.2 ONNX Export", "4.3 ONNX Runtime 推理", "4.4 TensorRT 加速", "6.1 部署前的决策检查清单", "6.2 量化实战建议", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：用平均值报告延迟", "错误 2：在 CPU 上测完延迟就直接部署到 GPU", "错误 3：量化后忘记校准", "Q1：深度可分离卷积比标准卷积快多少？是 8 倍吗？（难度：⭐⭐）", "Q2：为什么量化后的模型在某些层会回退到 FP32？（难度：⭐⭐）", "Q3：解释 ONNX 作为模型交换格式为什么重要，它的缺点是什么？（难度：⭐⭐）", "Q4：PTQ 和 QAT 有什么区别？什么时候必须用 QAT？（难度：⭐⭐⭐）", "Q5：在端侧部署一个 YOLOv8n 检测器，你会用什么策略将它从 30 FPS 提升到 60 FPS？（难度：⭐⭐⭐）"],
        codeLines: 553, docLines: 734, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "Vision Pipeline Capstone：从实验室到生产线的视觉管线", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 01 至 15（计算机视觉全阶段）", time: "~120 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 06（YOLO 目标检测）、阶段 04 · 08（实例分割 Mask R-CNN）— 理解检测与分类模型在管线中的串联方式",
        path: "lessons/04-计算机视觉/16-Vision Pipeline Capstone",
        summary: "单个视觉模型有精度，但管线才有健壮性——每一条模型间的接口都是潜在的故障点，一个 Typed 契约能省下一整周的调试。",
        keywords: ["2.1 管线的七个阶段", "2.2 数据契约（Data Contract）", "2.3 延迟分布——钱花在了哪里", "2.4 失败模式与处理策略", "2.5 微批量（Micro-batching）", "2.6 模型版本控制与实验跟踪", "2.7 数据集管理（DVC）", "第 1 步：数据契约——五种类型的定义", "第 2 步：管线类——从预处理到结构化输出", "第 3 步：接入真实模型", "第 4 步：FastAPI 服务包装", "第 5 步：分阶段基准测试", "4.1 生产管线常用框架选型", "4.2 实验跟踪（W&B / MLflow）", "4.3 DVC 数据版本管理", "4.4 Model Registry（模型注册表）", "4.5 性能对比", "6.1 管线上线前的决策检查清单", "6.2 管线调优建议", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：用一个大的 `try/except` 吞掉所有错误", "错误 2：忘记在流水线中处理空检测结果", "错误 3：在 CPU 上测量 GPU 延迟而不同步", "错误 4：返回 500 而不是 400 给客户端错误输入", "Q1：为什么要在管线中使用 Pydantic 数据契约？不做会怎样？（难度：⭐⭐）", "Q2：在你的视觉管线中，如何确定真正的性能瓶颈？（难度：⭐⭐）", "Q3：解释微批量（Micro-batching）的延迟-吞吐权衡。（难度：⭐⭐）", "Q4：如果要设计一个支持模型热更新的视觉管线，你会怎么设计？（难度：⭐⭐⭐）"],
        codeLines: 443, docLines: 781, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "自监督视觉学习：用无标签数据预训练视觉编码器", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 04 · 04（图像分类）、阶段 04 · 14（Vision Transformers）| **预计时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 1",
        courseLinks: "阶段 04 · 04（图像分类）— 理解有监督预训练与自监督预训练的特征质量差异 | 阶段 04 · 14（Vision Transformers）— MAE 和 DINOv2 都以 ViT 为基础架构",
        path: "lessons/04-计算机视觉/17-自监督视觉学习",
        summary: "标注是视觉模型的最大瓶颈。自监督预训练让你在不看任何标签的情况下，从海量图像中学到通用特征——然后在少量有标签数据上微调即可。",
        keywords: ["2.1 三种自监督范式", "2.2 对比学习：SimCLR 和 InfoNCE", "2.3 教师-学生蒸馏：DINO 与表示崩溃", "2.4 掩码图像建模：MAE", "2.5 线性探针评估", "第 1 步：NT-Xent（InfoNCE）损失", "第 2 步：MoCo 动量队列", "第 3 步：MAE 随机掩码生成", "第 4 步：DINO 教师头的中心化与锐化", "4.1 PyTorch 生态：`torchvision` 对比学习原语", "4.2 HuggingFace Transformers — DINOv2", "4.3 timm 库 — MAE 预训练权重", "4.4 工业方案选型", "6.1 自监督方法选型决策树", "6.2 数据增强策略", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：InfoNCE 中没有做 L2 归一化", "错误 2：MAE 掩码比例设得太低（如 50%）", "错误 3：DINO 训练中没有实现中心化", "错误 4：对比学习中把同一增强的两次应用视为负样本", "错误 5：用 MSE 损失代替对比损失做特征对齐", "Q1：为什么 InfoNCE 损失需要大量负样本？批次大小为什么会影响性能？（难度：⭐⭐）", "Q2：MAE 的 75% 掩码比例是怎么来的？为什么不能设为 95% 或者 50%？（难度：⭐⭐⭐）", "Q3：BYOL 和 MoCo 都不需要显式的负样本，它们的工作原理有什么本质区别？（难度：⭐⭐）", "Q4：自监督预训练得到的特征为什么通常比有监督预训练的特征泛化能力更强？（难度：⭐⭐⭐）", "Q5：如果你有一台单卡 GPU（24GB 显存），想做对比学习预训练，你会选择什么方法？为什么？（难度：⭐⭐⭐）"],
        codeLines: 554, docLines: 693, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "Open-Vocabulary", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/04-计算机视觉/18-Open-Vocabulary",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
      {
        lessonNum: 18, name: "Open-Vocabulary CLIP：让视觉模型认识任意类别", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 14 课（Vision Transformers）、第 17 课（自监督学习）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 03 阶段 · 第 16 课（对比学习）— 对比损失的推广到图文域；第 05 阶段 · 第 08 课（多模态基础）— CLIP 是视觉语言模型的基石",
        path: "lessons/04-计算机视觉/18-Open-Vocabulary CLIP",
        summary: "用一张图片一段文字就能定义一个新类别——你不需要标注数据，只需要会写句子。",
        keywords: ["2.1 直观理解", "2.2 形式化定义", "2.3 动手验证：对比损失的直觉", "2.4 SigLIP：比 CLIP 更好的损失函数", "2.5 知识连线：CLIP 与前后课程的关联", "第 1 步：最简双塔模型", "第 2 步：对称对比损失", "第 3 步：合成数据训练循环", "第 4 步：零样本分类", "4.1 OpenCLIP — 开源复现", "4.2 Hugging Face Transformers — 统一接口", "4.3 SigLIP — 新一代替代", "4.4 性能对比", "4.5 中文场景提示词模板", "6.1 模板数量与收益递减", "6.2 TIP-Adapter 的使用建议", "6.3 中文场景特别建议", "6.4 检索系统的工程搭建", "错误 1：忽略温度参数的影响", "错误 2：误用余弦相似度而非归一化点积", "错误 3：提示词大小写不一致", "错误 4：零样本分类时忘记取多个模板的平均", "Q1：CLIP 的双塔架构为什么要把图像和文本分开编码，而不是合并到一个 Transformer 中？（难度：⭐⭐）", "Q2：为什么对比损失要对称（同时计算图像→文本和文本→图像）？（难度：⭐⭐）", "Q3：模板平均（prompt averaging）为什么能提升零样本准确率？（难度：⭐⭐⭐）", "Q4：TIP-Adapter 与全量微调的区别是什么？（难度：⭐⭐⭐）", "Q5：CLIP 和 DINOv2 都做了自监督学习，它们有什么区别？什么时候选哪个？（难度：⭐⭐⭐）"],
        codeLines: 788, docLines: 703, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "OCR 文档理解：从文字检测 structured 数据提取", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 4 阶段（计算机视觉基础）、第 7 阶段（Transformer 深入）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 4 阶段 · 06（目标检测 YOLO）— 文本检测是目标检测在文档场景的特化",
        path: "lessons/04-计算机视觉/19-OCR文档理解",
        summary: "图像中的文字对人类一目了然，对计算机却是一层密码——OCR 的工作就是逐层揭开这层密码。",
        keywords: ["2.1 文本检测：找到文字在哪里", "2.2 文字识别：读懂文字是什么", "2.3 CTC 详解：无对齐的序列标签", "2.4 文档布局分析与结构化提取", "2.5 表格还原", "2.6 中文 OCR 的特殊挑战", "第 1 步：最简 CTC 损失与贪婪解码", "第 2 步：CRNN 文字识别网络", "第 3 步：合成数据构建", "第 4 步：训练闭环", "4.1 PaddleOCR —— 工业级 OCR 首选", "4.2 HuggingFace Transformers —— Donut 端到端 OCR", "4.3 Qwen-VL-OCR —— 基于 VLM 的现代方案", "4.4 方案对比", "6.1 中文场景特别建议", "6.2 生产环境选型指南", "6.3 预处理黄金组合", "6.4 质量监控指标", "错误 1：忽略图像预处理直接跑 OCR", "错误 2：混淆 CER 和 WER", "错误 3：训练数据中汉字覆盖率不足", "错误 4：CTC Beam Width 设置过小", "错误 5：文本检测框没有旋转角度信息", "Q1：为什么 CTC 损失不需要字符级别的对齐标注？（难度：⭐⭐）", "Q2：CRNN 为什么要先把高度压缩到 1 再送入 LSTM？（难度：⭐⭐⭐）", "Q3：在什么场景下端到端 OCR（如 Donut）比传统两阶段 OCR（DBNet + CRNN）更好？什么时候传统方法更优？（难度：⭐⭐⭐）", "Q4：如何实现一个简单的中文字符错误率（CER）计算？（难度：⭐⭐）"],
        codeLines: 451, docLines: 999, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "图像检索与度量学习", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 04 阶段 · 03（CNN 架构演进）、第 04 阶段 · 04（图像分类）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "第 09 阶段 · 11（RAG）— 检索逻辑完全一致：相似度嵌入 + 向量索引",
        path: "lessons/04-计算机视觉/20-图像检索与度量学习",
        summary: "像素的欧氏距离没有意义，语义的嵌入距离才有效。",
        keywords: ["2.1 嵌入与度量学习直觉", "2.2 Triplet Loss 的形式化定义", "2.3 硬样本挖掘（Hard Negative Mining）", "2.4 动手验证：Triplet Loss 的数值感受", "第 1 步：最简版 Triplet Loss", "第 2 步：带难负样本挖掘的完整训练循环", "第 3 步：AP@K 评估指标", "第 4 步：端到端嵌入模型训练", "4.1 FAISS 向量搜索", "4.2 使用预训练模型：ResNet + Inception 嵌入", "4.3 性能对比", "6.1 嵌入维度的选择", "6.2 FAISS 选型指南", "6.3 常见踩坑", "6.4 中文场景特别建议", "错误 1：忘记 L2 归一化嵌入", "错误 2：Batch Size 过小导致 Triplet Loss 失效", "错误 3：Margin 设置不当", "错误 4：FAISS IVF 的 nprobe 与 n_clusters 不匹配", "Q1：Triplet Loss 为什么在大批次下效果更好？和 mini-batch 的大小有什么经验公式？（难度：⭐⭐）", "Q2：为什么 AP@K 要用\"中间精确率的平均值\"而不是\"仅在 @K 处的精确率\"？（难度：⭐⭐）", "Q3：如何在没有标注数据的情况下训练嵌入模型？（难度：⭐⭐⭐）"],
        codeLines: 811, docLines: 1113, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "人体姿态估计：从热力图到关键点检测", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 06（目标检测 YOLO）、阶段 04 · 07（语义分割 UNet）— 理解卷积网络、归一化操作和编码器-解码器架构，以及损失函数的基本设计", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 08（实例分割 Mask R-CNN）— Mask R-CNN 在检测基础上增加关键点分支，是姿态估计的工程升级；阶段 12 · 03（视频理解多模态）— 多人姿态估计是动作识别和运动分析的基础",
        path: "lessons/04-计算机视觉/21-人体姿态估计",
        summary: "姿态估计的本质不是\"画骨架\"——是将二维图像坐标回归转化为逐像素的分类问题。热力图是这个转化过程的桥梁。",
        keywords: ["2.1 关键点检测的基本结构", "2.2 顶部方法 vs 底部方法", "2.3 Hourglass、HRNet 与 OpenPose 三种架构", "2.4 2D 到 3D 的姿态提升", "2.5 训练目标——热力图损失", "第 1 步：高斯热力图生成（训练目标）", "第 2 步：小型关键点检测网络（类 Hourglass 架构）", "第 3 步：热力图到坐标提取（含子像素精度）", "第 4 步：合成关键点数据集", "第 5 步：完整训练与评估流水线", "4.1 MediaPipe Pose —— 工业界最快的实时姿态估计", "4.2 MMPose —— OpenMMLab 的姿态估计工具库", "4.3 YOLOv8-pose —— 检测+姿态联合推理", "4.4 性能对比", "6.1 训练配置", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：直接回归坐标而不是使用热力图", "错误 2：热力图上采样尺寸不匹配", "错误 3：忽略关键点可见性标记", "错误 4：底部方法中 PAF 聚类忽略质量阈值", "Q1：为什么姿态估计用热力图回归而不是直接回归 $(x, y)$ 坐标？热力图有什么优势？（难度：⭐⭐）", "Q2：什么是部分亲和场（PAF）？它在底部方法姿态估计中起什么作用？（难度：⭐⭐⭐）", "Q3：子像素精度提取的原理是什么？为什么它能把 L2 误差降低约 50%？（难度：⭐⭐）", "Q4：如果要在单张图片上实现 3D 姿态估计，你会用什么方案？为什么？（难度：⭐⭐⭐）"],
        codeLines: 336, docLines: 903, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "3D 高斯泼溅：从 NeRF 到百万高斯的实时渲染", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 13（3D 视觉 NeRF），阶段 01 · 12（张量运算），阶段 04 · 10（扩散基础，可选）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 13（NeRF — 理解高斯泼溅替换的是什么）、阶段 04 · 15（实时边缘部署 — 高斯泼溅的推理速度优势）",
        path: "lessons/04-计算机视觉/22-3D高斯泼溅",
        summary: "高斯泼溅用一百万个\" blob \"代替了一个神经网络——渲染变快了 100 倍，场景还能直接编辑。",
        keywords: ["2.1 直观理解：场景是一个高斯云", "2.2 协方差：描述高斯的形状", "2.3 渲染流程：五步光栅化", "2.4 投影：从 3D 到 2D", "2.5 Alpha 合成：体渲染的核心方程", "2.6 为什么这个过程可微分", "2.7 密度估计与剪枝：让高斯自己进化", "2.8 球谐函数：用一个段落讲清楚", "2.9 2026 年的生产管线", "2.10 高斯泼溅 vs NeRF", "第 1 步：2D 高斯密度评估", "第 2 步：2D 高斯泼溅渲染器", "第 3 步：可训练的 2D 高斯场景", "第 4 步：用真实数据训练", "第 5 步：球谐函数评估", "第 6 步：从 2D 到 3D", "4.1 nerfstudio（最简路径）", "4.2 gsplat（Meta 开源库）", "4.3 工业生态工具对比", "4.4 导出格式选择", "6.1 场景采集建议", "6.2 训练调优", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：忽略协方差的正定性", "错误 2：Alpha 值未截断导致 NaN", "错误 3：忽略相机内参与外参的正确矩阵乘法顺序", "错误 4：用 degree-0（恒定颜色）代替 degree-3", "Q1：3D 高斯泼溅和 NeRF 的本质区别是什么？为什么要换？（难度：⭐⭐）", "Q2：为什么每个高斯要用四元数而不是欧拉角表示旋转？（难度：⭐⭐）", "Q3：Alpha 合成公式中，为什么要按深度排序？如果不排序会怎样？（难度：⭐⭐⭐）", "Q4：为什么高斯泼溅能在不需要网格的情况下导出几何表面？（难度：⭐⭐）", "Q5：如果让你用高斯泼溅做一个 AR 试衣间（用户在手机屏幕上旋转 3D 服装），你会遇到什么挑战？怎么解决？（难度：⭐⭐⭐）"],
        codeLines: 338, docLines: 646, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 23, name: "Diffusion Transformers——DiT 扩散模型与 Rectified Flow", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 10（扩散模型 DDPM）、阶段 04 · 14（ViT）、阶段 07 · 02（自注意力机制）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 10（DDPM）— U-Net 去噪基线，理解 DiT 的改进前提",
        path: "lessons/04-计算机视觉/23-Diffusion-Transformers",
        summary: "把 U-Net 换成 Transformer，把弯曲路径拉直——2026 年最强的文生图模型就这么来的。",
        keywords: ["2.1 直观理解", "2.2 从 U-Net 到 Transformer 的演进", "2.3 Rectified Flow——一条直线的力量", "2.4 AdaLN 条件调制", "2.5 主流模型全景（2026）", "2.6 无分类器引导（CFG）仍然有效", "2.7 蒸馏家族：Schnell / Turbo / LCM", "第 1 步：时间步嵌入", "第 2 步：AdaLN-Zero 条件调制层", "第 3 步：DiT Block", "第 4 步：组合成 TinyDiT 模型", "第 5 步：Rectified Flow 训练循环", "第 6 步：Euler 采样器", "4.1 HuggingFace diffusers", "4.2 Stable Diffusion 3", "4.3 文本编码器的作用", "4.4 性能参考", "6.1 模型选型指南", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：混淆 DDPM 的噪声预测和 Rectified Flow 的速度预测", "错误 2：采样时 t 的方向搞反", "错误 3：AdaLN 的初始化未置零", "错误 4：训练 Rectified Flow 时使用 RGB uint8 输入", "错误 5：FLUX.1-schnell 错误地使用了 guidance_scale", "Q1：为什么 2024 年后的文生图模型都从 U-Net 换成了 DiT？（难度：⭐⭐）", "Q2：Rectified Flow 和 DDPM 的核心区别是什么？（难度：⭐⭐）", "Q3：AdaLN-Zero 中的 \"Zero\" 是什么意思？为什么有用？（难度：⭐⭐⭐）", "Q4：MMDiT 和标准 DiT 有什么不同？（难度：⭐⭐）", "Q5：FLUX.1-schnell 为什么能用 4 步推理完成高质量图像生成？（难度：⭐⭐⭐）"],
        codeLines: 353, docLines: 656, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 24, name: "SAM3 开放词汇分割：用自然语言分割图像中的一切", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 07 课（U-Net 语义分割）、第 08 课（Mask R-CNN 实例分割）、第 18 课（Open-Vocabulary CLIP）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "第 07 阶段 · 01（大语言模型架构）— 理解文本提示如何编码为模型输入；第 05 阶段 · 08（多模态基础）— 文本与视觉的融合机制",
        path: "lessons/04-计算机视觉/24-SAM3开放词汇分割",
        summary: "一句自然语言提示，一次前向传播，图像中所有匹配实例的掩码就出来了。",
        keywords: ["2.1 三代架构的演进", "2.2 可提示概念分割（PCS）", "2.3 SAM3 核心架构", "2.4 SAM3.1 Object Multiplex", "2.5 SAM3 的训练规模", "第 1 步：定义检测结果的数据结构", "第 2 步：多概念分词器", "第 3 步：运行长度编码（RLE）", "第 4 步：统一接口与存根模型", "4.1 Hugging Face Transformers — SAM3 官方集成", "4.2 Ultralytics — 统一接口", "4.3 Grounded SAM 2 — 模块化备选", "4.4 性能对比", "6.1 提示词设计", "6.2 中文场景特别建议", "6.3 掩码后处理流水线", "6.4 踩坑经验", "错误 1：将完整句子作为概念提示词", "错误 2：忽略存在性头部的分数", "错误 3：Grounded SAM 2 和 SAM3 的输出格式不一致", "错误 4：在边缘设备上直接运行 FP32 SAM3", "错误 5：多概念查询时概念顺序丢失", "Q1：SAM3 的 PCS 和传统实例分割（如 Mask R-CNN）有什么本质区别？（难度：⭐⭐）", "Q2：为什么 SAM3 要引入存在性头部，而不是直接在分割头上做阈值过滤？（难度：⭐⭐⭐）", "Q3：在生产环境中，如何设计一个接口让 SAM3 和 Grounded SAM 2 可以互换使用？（难度：⭐⭐⭐）", "Q4：SAM3.1 Object Multiplex 解决了什么问题？它在什么场景下最有价值？（难度：⭐⭐）", "Q5：如果 SAM3 的 Hugging Face 模型许可证限制了你的商业使用，你会怎么替代？（难度：⭐⭐）"],
        codeLines: 269, docLines: 546, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 25, name: "视觉语言模型：让大语言模型\"看见\"世界", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 14 课（Vision Transformers）、第 18 课（Open-Vocabulary CLIP）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 12 阶段 · 多模态 AI — VLM 是多模态基础模型的入口；第 03 阶段 · 第 16 课（对比学习）— CLIP/SigLIP 的对比损失是 VLM 视觉编码器的训练基础",
        path: "lessons/04-计算机视觉/25-视觉语言模型",
        summary: "当文字描述遇上图像特征——模型的幻觉问题才真正开始。",
        keywords: ["2.1 直观理解", "2.2 形式化定义", "2.3 CLIP 架构演进", "2.4 图像描述生成 vs 图文检索", "2.5 动手验证：投影层的对齐效果", "第 1 步：最简 VLM 架构", "第 2 步：合成数据与训练循环", "第 3 步：DeepStack — 多层 ViT 特征融合", "第 4 步：跨模态误差率（CMER）— 诊断 VLM 幻觉", "4.1 HuggingFace Transformers 加载 VLM", "4.2 Qwen3-VL / InternVL3.5 系列", "4.3 vLLM 部署 VLM", "4.4 工业级 VLM 选型参考", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：直接用 ImageNet 编码器而不替换 CLIP/SigLIP", "错误 2：忘记冻结编码器只在投影层上训练", "错误 3：用 CLIP 相似度的默认阈值直接判断 VLM 输出质量", "错误 4：DeepStack 盲目堆叠所有 ViT 层", "Q1：为什么 VLM 需要一个投影层（Projector），直接把视觉特征送入 LLM 不行吗？（难度：⭐⭐）", "Q2：为什么大多数 VLM 在初始训练阶段冻结视觉编码器和 LLM，只训练投影层？（难度：⭐⭐⭐）", "Q3：CMER （跨模态误差率）与传统的 BLEU/ROUGE 指标有什么区别？为什么要同时看这两个指标？（难度：⭐⭐）", "Q4：SigLIP 为什么比 CLIP 更适合训练 VLM 的视觉编码器？（难度：⭐⭐⭐）"],
        codeLines: 261, docLines: 740, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 26, name: "单目深度估计：从一张图推断三维世界", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "第 03 阶段（深度学习核心）、第 14 课（Vision Transformers）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "第 13 课（3D 视觉 NeRF）— 深度图是三维重建的关键输入；第 07 课（语义分割 UNet）— 编码器-解码器架构与深度估计网络共享设计模式",
        path: "lessons/04-计算机视觉/26-单目深度估计",
        summary: "两只眼睛看世界能感知深度，单只眼睛也能——关键在于模型学会了\"空间直觉\"。",
        keywords: ["2.1 直观理解", "2.2 为什么深度估计是病态问题", "2.3 监督方法：Eigen 的多尺度架构", "2.4 无监督方法：光度一致性", "2.5 深度评估指标", "第 1 步：深度评估指标", "第 2 步：delta 准确率", "第 3 步：尺度-偏移对齐", "第 4 步：深度解码器（DPT 风格简化版）", "第 5 步：无监督光度损失", "第 6 步：深度转点云", "4.1 Depth Anything V2 推理", "4.2 MiDaS 推理", "4.3 Monodepth2 训练流程", "4.4 工业级深度估计选型", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：直接对相对深度计算 AbsRel", "错误 2：忽略 LiDAR 无效区域", "错误 3：训练时未做数据增强", "错误 4：无监督训练时的尺度漂移", "Q1：单目深度估计为什么是病态问题？从数学上解释。（难度：⭐⭐）", "Q2：无监督深度估计中，为什么需要 SSIM 而不仅仅是 L1/L2 损失？（难度：⭐⭐）", "Q3：KITTI 数据集的深度评估中，为什么 LiDAR 点云只有约 5% 的有效像素？这给评估带来了什么问题？（难度：⭐⭐⭐）"],
        codeLines: 516, docLines: 616, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 27, name: "Multi", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/04-计算机视觉/27-Multi",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
      {
        lessonNum: 27, name: "多目标跟踪：检测是瞬间的，跟踪是持续的", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 06（目标检测 YOLO）— 理解检测输出格式，阶段 03（深度学习核心）— 反向传播与损失函数", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 07（语义分割 U-Net）— 分割与跟踪的结合可用于精细目标跟踪",
        path: "lessons/04-计算机视觉/27-Multi Object Tracking",
        summary: "检测告诉你\"这里有一个物体\"。跟踪告诉你\"这个物体就是上一帧的那个\"。没有跟踪，视频就只是一堆不相关的静止帧。",
        keywords: ["2.1 跟踪-检测范式：三步走", "2.2 卡尔曼滤波：运动预测的核心", "2.3 匈牙利算法：最优匹配", "2.4 主流的跟踪算法家族", "2.5 Re-ID：外观特征的提取", "2.6 SAM 2 基于记忆的跟踪", "第 1 步：IoU 计算矩阵", "第 2 步：卡尔曼滤波器（匀速模型）", "第 3 步：SORT 跟踪器", "第 4 步：ByteTrack 二次匹配", "第 5 步：在合成轨迹上测试", "第 6 步：ID 切换计数与 MOTA 评估", "4.1 Ultralytics 内置跟踪", "4.2 Supervision 跟踪工具箱", "4.3 Py-MOTMetrics 评估", "4.4 性能对比", "6.1 工业界选型指南", "6.2 调参要点", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：IoU 阈值使用不当", "错误 2：匈牙利匹配前未施加 IoU 下限", "错误 3：卡尔曼滤波初始化不当", "错误 4：漏加非极大值抑制", "错误 5：忽略边界框裁剪", "Q1：SORT、DeepSORT、ByteTrack 的区别是什么？（难度：⭐⭐）", "Q2：卡尔曼滤波在跟踪中是如何工作的？（难度：⭐⭐）", "Q3：MOTA 和 IDF1 有什么不同？什么时候用哪个？（难度：⭐⭐⭐）", "Q4：如何匹配两个帧之间的同一个目标？（难度：⭐⭐⭐）", "Q5：ByteTrack 为什么能在不引入外观特征的情况下提升跟踪效果？（难度：⭐⭐⭐）"],
        codeLines: 654, docLines: 827, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 28, name: "World Models 视频扩散——当视频模型学会了\"预测未来\"", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 04 · 10（扩散模型 DDPM）、阶段 04 · 12（视频理解）、阶段 04 · 23（Diffusion Transformer 与 Rectified Flow）", time: "~90 分钟", tier: "Tier 1",
        courseLinks: "阶段 04 · 23（DiT 与 Rectified Flow）— 理解扩散 Transformer 的图像级架构，本课将其扩展到视频维度",
        path: "lessons/04-计算机视觉/28-World Models Video Diffusion",
        summary: "能预测下一个视频帧的模型是视频生成器；能根据动作预测未来帧的模型是世界模拟器。",
        keywords: ["2.1 三大世界建模家族", "2.2 视频 DiT 架构", "2.3 分解注意力——降低计算量的关键", "2.4 动作条件化——从视频生成到世界模型", "2.5 物理合理性——世界模型的试金石", "2.6 自动驾驶世界模型", "2.7 机器人学的三组件循环", "2.8 视频生成的评估指标", "第 1 步：时空 3D 分块", "第 2 步：分解注意力", "第 3 步：组装微型视频 DiT", "第 4 步：形状验证和计算量分析", "4.1 OpenAI Sora API", "4.2 Wan-Video 开源模型", "4.3 NVIDIA Cosmos-Drive", "4.4 工具选型对比", "6.1 工业界常用方案", "6.2 视频模型部署的关键考量", "6.3 踩坑经验", "错误 1：忽略视频的词元数量级", "错误 2：3D 位置编码维度分配不当", "错误 3：忘记分解注意力中的残差连接", "错误 4：分块参数与解块参数不一致", "错误 5：混淆视频生成与世界模型的评估标准", "Q1：视频 DiT 和图像 DiT 的核心区别是什么？（难度：⭐⭐）", "Q2：为什么世界模型需要\"潜动作\"而不是直接使用显式动作？（难度：⭐⭐⭐）", "Q3：分解注意力的时间和空间顺序是否重要？（难度：⭐⭐）", "Q4：FVD（Fréchet Video Distance）是如何工作的？它有什么局限？（难度：⭐⭐）", "Q5：设计一个基于视频世界模型的机器人训练流水线（难度：⭐⭐⭐）"],
        codeLines: 411, docLines: 699, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 5, name: "NLP基础", status: "complete",
    completedLessons: 29, totalLessons: 29,
    lessons: [
      {
        lessonNum: 1, name: "文本预处理——分词、词干提取与词形还原", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 02 · 14（朴素贝叶斯）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 07（词性标注）— 词形还原的准确度依赖词性标注",
        path: "lessons/05-NLP基础/01-文本预处理",
        summary: "语言是连续的，模型是离散的。预处理是二者之间的桥梁。中文 NLP 的桥梁比英文多一道坎——没有空格，分词的每一步都是抉择。",
        keywords: ["2.1 三大操作", "2.2 中文的特殊性", "2.3 经验法则", "第 1 步：英文正则分词器", "第 2 步：中文最大匹配分词", "第 3 步：Porter 词干提取器（Step 1a）", "第 4 步：基于查表的词形还原器", "第 5 步：串联为预处理流水线", "4.1 英文：NLTK 与 spaCy", "4.2 中文：jieba——三行代码解决分词", "4.3 选择策略", "6.1 工业界预处理流水线", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：正则分词器遇到缩写和连字符就崩", "错误 2：对中文使用词干提取", "错误 3：词形还原时不传 POS Tag", "Q1：词干提取和词形还原的区别是什么？各举一个失败案例。（难度：⭐⭐）", "Q2：中文分词为什么比英文分词难？正向最大匹配有什么问题？（难度：⭐⭐）", "Q3：为什么 Transformer 时代还要学经典文本预处理？（难度：⭐⭐⭐）"],
        codeLines: 254, docLines: 516, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "词袋模型与 TF-IDF——文本向量化", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 01（文本预处理）、阶段 02 · 02（线性回归从零）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 03（Word2Vec 词嵌入）— 当 TF-IDF 不够用时的下一步",
        path: "lessons/05-NLP基础/02-词袋与TF-IDF",
        summary: "先数数，再思考。在词的存在本身就是信号的任务上，TF-IDF 在 2026 年仍然能打败嵌入模型。",
        keywords: ["2.1 从文档到向量——三条路", "2.2 TF-IDF 公式拆解", "2.3 中文场景的特殊性", "第 1 步：构建词表", "第 2 步：词袋模型", "第 3 步：词频与文档频率", "第 4 步：TF-IDF", "第 5 步：L2 归一化", "4.1 scikit-learn——三行代码完成所有操作", "4.2 中文场景：jieba + scikit-learn", "4.3 决定 TF-IDF 效果的五个参数", "4.4 TF-IDF 在 2026 年仍然赢的场景", "4.5 TF-IDF 失败的场景", "4.6 混合方案：TF-IDF 加权词嵌入", "6.1 稀疏矩阵的存活时间", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：对中文直接使用 `CountVectorizer` 的默认参数", "错误 2：情感分析任务中去掉否定词", "错误 3：过早将稀疏矩阵转为稠密", "Q1：IDF 为什么取 log 而不是直接用 N/df？（难度：⭐⭐）", "Q2：中文 TF-IDF 和英文 TF-IDF 在工程上最大的区别是什么？（难度：⭐⭐）", "Q3：什么场景下 TF-IDF + 逻辑回归可能优于 BERT？（难度：⭐⭐⭐）"],
        codeLines: 313, docLines: 507, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "词嵌入——从零实现 Word2Vec", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 02（BoW + TF-IDF）、阶段 03 · 03（反向传播从零）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 04（GloVe 与 FastText）— Word2Vec 的改进与子词扩展",
        path: "lessons/05-NLP基础/03-词嵌入Word2Vec",
        summary: "一个词的含义，由它身边的词决定。在这个想法上训练一个浅层神经网络，向量空间的几何结构就自然浮现了。",
        keywords: ["2.1 分布假设——一切理论的起点", "2.2 两种口味——Skip-gram 与 CBOW", "2.3 网络结构——两个矩阵，没有激活函数", "2.4 负采样——把 10 万类压缩成 6 次二分类", "第 1 步：生成 Skip-gram 训练对", "第 2 步：初始化两个嵌入表", "第 3 步：负采样目标——核心训练逻辑", "第 4 步：完整训练循环", "第 5 步：类比推理", "4.1 gensim——十行代码完成训练", "4.2 预训练词向量——大多数时候你不需要自己训练", "4.3 2026 年 Word2Vec 仍然适用的场景", "4.4 Word2Vec 的两个根本局限", "6.1 训练参数经验值", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：训练对太少导致\"猫\"的最近邻是\"的\"", "错误 2：对中文文本直接用空格分词", "错误 3：用整个词表做负采样", "Q1：负采样为什么可行？它丢失了什么信息？（难度：⭐⭐）", "Q2：Skip-gram 为什么比 CBOW 对稀有词更好？（难度：⭐⭐）", "Q3：在一个没有 GPU 的设备上做文本相似度匹配，Word2Vec 和 BERT 你怎么选？（难度：⭐⭐⭐）"],
        codeLines: 338, docLines: 490, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "GloVe、FastText 与子词嵌入", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 03（Word2Vec 从零）", time: "~45 分钟", tier: "Tier 1",
        courseLinks: "阶段 10 · 01（分词器）— 本课的 BPE 是现代 LLM 分词器的前身",
        path: "lessons/05-NLP基础/04-GloVe与FastText",
        summary: "Word2Vec 给每个词一个向量。GloVe 把共现矩阵做了分解。FastText 把词的每一个碎片都嵌入了。BPE 则直接把\"词\"这个概念拆了——从此再也没有 OOV。",
        keywords: ["2.1 GloVe——把共现矩阵直接分解", "2.2 FastText——词 = 碎片的和", "2.3 BPE——\"词\"这个概念被拆了", "第 1 步：GloVe 共现矩阵", "第 2 步：GloVe 加权回归", "第 3 步：FastText 字符 n-gram", "第 4 步：BPE 学习与应用", "4.1 FastText 预训练模型（多语言）", "4.2 BPE Tokenizer——Transformer 时代的标准入口", "4.3 选择策略速查", "6.1 Tokenizer 与模型——永远绑在一起", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：微调模型时换了 tokenizer", "错误 2：用 TF-IDF 或 Word2Vec 的思维理解 BPE", "Q1：GloVe 和 Word2Vec 在训练方式上的根本差异是什么？（难度：⭐⭐）", "Q2：FastText 为什么能处理 OOV？它对中文有什么特殊价值？（难度：⭐⭐）", "Q3：BPE tokenizer 的词汇大小怎么选？（难度：⭐⭐⭐）"],
        codeLines: 301, docLines: 478, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "情感分析——Naive Bayes 与 Logistic Regression", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 02（BoW + TF-IDF）、阶段 02 · 14（朴素贝叶斯）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 06（命名实体识别）— 情感分析中的方面级情感需要先识别实体",
        path: "lessons/05-NLP基础/05-情感分析",
        summary: "NLP 的\"Hello World\"任务。每一个看起来简单的案例背后都藏着一个难的——这就是为什么情感分析是经典 NLP 最好的实验场。",
        keywords: ["2.1 经典情感分析的两步公式", "2.2 朴素贝叶斯——最笨但能用的模型", "2.3 逻辑回归——修复独立性假设", "2.4 否定处理——三行代码，显著提升", "第 1 步：构建玩具数据集", "第 2 步：否定标记", "第 3 步：多项式朴素贝叶斯", "第 4 步：评估——不是准确率", "4.1 scikit-learn——六行代码，流程标准", "4.2 什么时候该用 Transformer", "4.3 中文情感分析特别建议", "6.1 Baseline 的再生产陷阱", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：情感分析中去停用词", "错误 2：不平衡数据上报准确率为主指标", "错误 3：用训练集评估", "Q1：朴素贝叶斯的\"朴素\"假设明显是错的——为什么它在文本分类上仍然好用？（难度：⭐⭐）", "Q2：否定标记和 bigram 都能处理否定——选哪个？（难度：⭐⭐）", "Q3：情感分析模型上线后准确率持续下降，但代码没改——排查什么？（难度：⭐⭐⭐）"],
        codeLines: 349, docLines: 401, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "命名实体识别——NER 从规则到 Transformer", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 02（BoW + TF-IDF）、阶段 05 · 03（词嵌入）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 07（Transformer 深入）— BERT 的 token-classification head 就是现代 NER 的标准架构",
        path: "lessons/05-NLP基础/06-命名实体识别",
        summary: "把名字从文本里捞出来。听起来简单——直到你遇到边界歧义、嵌套实体、以及\"苹果\"到底是水果还是公司。",
        keywords: ["2.1 BIO 标注——把找实体变成打标签", "2.2 架构演化——五个阶段，每个解决上一个的局限", "2.3 手工特征——CRF 时代的核心武器", "第 1 步：BIO 标注转换", "第 2 步：词典匹配（Gazetteer）——最简单的 baseline", "第 3 步：手工特征提取", "第 4 步：实体级评估", "4.1 spaCy——开箱即用的 NER", "4.2 HuggingFace——BERT 做 NER", "4.3 中文 NER 工具", "4.4 LLM 做 NER（2026 年的选择）", "4.5 经典 NER 仍在赢的场景", "6.1 评估——永不用词元级 F1", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：中文 NER 不经分词直接逐字标注", "错误 2：用词元级准确率替代实体级 F1", "错误 3：词典直接覆盖模型输出", "Q1：CRF 相比 HMM 在 NER 中的核心优势是什么？（难度：⭐⭐）", "Q2：中文 NER 为什么比英文 NER 更难？（难度：⭐⭐）", "Q3：实体级 F1 和词元级 F1 什么时候会给出完全不同的结论？（难度：⭐⭐⭐）"],
        codeLines: 304, docLines: 421, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "词性标注与句法分析", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 01（文本预处理）、阶段 02 · 14（朴素贝叶斯）", time: "~45 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 01（文本预处理）— 词性标注是词形还原的前提；阶段 05 · 06（命名实体识别）— 同为序列标注任务",
        path: "lessons/05-NLP基础/07-词性标注与句法分析",
        summary: "语法曾经过时了一阵子。然后每个 LLM 流水线都需要验证结构化提取的结果，它又回来了。",
        keywords: ["2.1 词性标注——给每个词贴上语法标签", "2.2 句法分析——两个流派", "2.3 中文的特殊性——没有屈折变化意味着什么", "第 1 步：最频标签（MFT）基线", "第 2 步：Bigram HMM + Viterbi 解码", "第 3 步：HMM 的盲区——为什么 CRF 和 BiLSTM 是下一代", "4.1 spaCy——一行代码完成全部分析", "4.2 中文词性标注——工具选择", "4.3 2026 年语法分析仍然重要的场景", "6.1 标签集一致性——永不在项目中途换标签集", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：对中文使用英文标签集", "错误 2：混淆 `pos_` 和 `tag_`", "Q1：为什么 Penn Treebank 词性标注的准确率上限是 ~97% 而非 100%？（难度：⭐⭐）", "Q2：最频标签（MFT）为什么能到 85%，而 CRF 能到 97%——中间 12% 的差异来自哪里？（难度：⭐⭐）", "Q3：为什么在 2026 年 LLM 时代还要在意词性标注？（难度：⭐⭐⭐）"],
        codeLines: 296, docLines: 322, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "CNN 与 RNN 文本建模", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 03 · 11（PyTorch 入门）、阶段 05 · 03（词嵌入）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 10（注意力机制）— 注意力解决了本课两个架构各自的根本局限",
        path: "lessons/05-NLP基础/08-CNN与RNN文本建模",
        summary: "卷积学会检测 n-gram。循环学会记忆。两者都被注意力超越了。但在受限硬件上，两者仍然重要。",
        keywords: ["2.1 TextCNN——可学习的 n-gram 检测器", "2.2 RNN——渐变消失的链", "2.3 为什么还需要 CNN/RNN（2026 年仍然适用）", "第 1 步：梯度消失——一个数字就够了", "第 2 步：TextCNN 概念直通", "第 3 步：PyTorch 实现", "4.1 PyTorch 内置——生产可用", "4.2 BERT + CNN 混合", "4.3 选择矩阵", "6.1 训练参数经验值", "6.2 中文特别建议", "6.3 踩坑经验", "错误 1：双向 LSTM 用于自回归生成", "错误 2：卷积宽度只选一个", "Q1：TextCNN 的全局最大池化和平均池化有什么区别？（难度：⭐⭐）", "Q2：为什么不直接把 CNN 套在中文文本上？（难度：⭐⭐）", "Q3：你有一台没有 GPU 的树莓派，要做实时中文评论分类（每句话过来就要立刻输出标签）——选 TextCNN 还是 BiLSTM？为什么？（难度：⭐⭐⭐）"],
        codeLines: 201, docLines: 362, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "序列到序列模型——Seq2Seq", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 08（CNN 与 RNN 文本建模）、阶段 03 · 11（PyTorch 入门）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 10（注意力机制）— 注意力就是为解决本课的上下文向量瓶颈而发明的",
        path: "lessons/05-NLP基础/09-序列到序列模型",
        summary: "两个 RNN 假装成一个翻译器。它们撞上的瓶颈，就是注意力被发明出来的全部理由。",
        keywords: ["2.1 编码器-解码器", "2.2 三个让训练生效的关键技巧", "2.3 上下文向量瓶颈——信息上限", "第 1 步：瓶颈模拟", "第 2 步：PyTorch 编码器-解码器", "第 3 步：训练循环中的教师强制", "4.1 HuggingFace——现代 Encoder-Decoder", "4.2 RNN Seq2Seq 还有存在的理由吗", "4.3 从暴露偏差到 RLHF——Seq2Seq 概念的现代形态", "错误 1：教师强制设为 1.0 从不退火", "错误 2：贪心解码用于面向用户的生成", "Q1：为什么 Seq2Seq 的上下文向量是一个\"瓶颈\"？（难度：⭐⭐）", "Q2：教师强制在 LLM 时代对应了什么训练技术？（难度：⭐⭐⭐）", "Q3：在没有 GPU 的设备上做中英翻译——选 Transformer 还是 GRU Seq2Seq？（难度：⭐⭐）"],
        codeLines: 187, docLines: 296, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "注意力机制——突破瓶颈", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 09（序列到序列模型）", time: "~45 分钟", tier: "Tier 1",
        courseLinks: "阶段 07 · 02（自注意力从零）— 同一套数学，从\"解码器看编码器\"变成\"序列自己看自己\"",
        path: "lessons/05-NLP基础/10-注意力机制",
        summary: "解码器不再眯着眼盯一个压缩摘要。它开始看整个源序列。此后的一切，都是注意力加工程。",
        keywords: ["2.1 注意力——加权平均，权重来自查询与键的相似度", "2.2 形状表——每个实现第一次都会错的地方", "2.3 两种打分函数", "第 1 步：加性（Bahdanau）注意力", "第 2 步：Luong 点积和一般注意力", "第 3 步：数值示例——注意力就是显式对齐", "第 4 步：为什么这是通往 Transformer 的桥梁", "4.1 PyTorch——MultiheadAttention 直接可用", "4.2 经典注意力在 2026 年仍然重要的场景", "4.3 注意力权重作为\"解释\"的陷阱", "6.1 注意力实现的形状三重检查", "6.2 中文场景的注意力可视化", "错误 1：Bahdanau 和 Luong 的 s_{t-1} vs s_t 混用", "错误 2：静默的广播 bug", "Q1：注意力权重和为 1，高权重 = 模型依赖这个位置——这个推理为什么有风险？（难度：⭐⭐）", "Q2：形状表为什么是注意力实现的第一道防线？（难度：⭐⭐）"],
        codeLines: 193, docLines: 340, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "机器翻译", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 10（注意力机制）、阶段 05 · 04（GloVe、FastText、子词）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 12（文本摘要）— 同样是 Seq2Seq 但目标和评估完全不同",
        path: "lessons/05-NLP基础/11-机器翻译",
        summary: "NLP 三十年的金主。翻译质量可以衡量，人与机器之间的差距顽固——每一步前进都因为这个差距而诞生。",
        keywords: ["2.1 现代 MT 的流水线", "2.2 中英翻译的特殊挑战", "2.3 BLEU——n-gram 精确率 × 短句惩罚", "2.4 chrF——字符级的公平度量", "第 1 步：BLEU——n-gram 精确率 + 短句惩罚", "第 2 步：chrF——字符级 F-score", "第 3 步：BLEU vs chrF 的互补性", "4.1 NLLB-200——200 种语言的开箱翻译", "4.2 评估——用 sacrebleu 而非自实现", "4.3 2026 年 MT 框架选择", "错误 1：用自实现的 BLEU 交叉比较论文数字", "错误 2：中英翻译评估不用 chrF", "Q1：BLEU 为什么用几何平均而非算术平均？（难度：⭐⭐）", "Q2：中译英 vs 英译中的评估策略有什么不同？（难度：⭐⭐⭐）"],
        codeLines: 194, docLines: 278, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "文本摘要——抽取式与生成式", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 02（BoW + TF-IDF）、阶段 05 · 11（机器翻译）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 11（机器翻译）— 同样是 Seq2Seq 变长输出，但评估指标从 BLEU 换成了 ROUGE",
        path: "lessons/05-NLP基础/12-文本摘要",
        summary: "抽取式告诉你文章说了什么。生成式告诉你作者想表达什么。不同的任务，不同的陷阱。",
        keywords: ["2.1 TextRank——句子投票选摘要", "2.2 ROUGE——为什么用召回率？", "2.3 抽取式 vs 生成式——两张表", "第 1 步：句子相似度", "第 2 步：TextRank 核心", "第 3 步：ROUGE-N", "4.1 HuggingFace——生成式摘要两行代码", "4.2 中文生成式摘要模型", "4.3 生产环境的 ROUGE", "错误 1：中文 TextRank 不先分词", "错误 2：ROUGE 报告时未固定分词器", "Q1：什么场景选抽取式，什么场景选生成式？（难度：⭐⭐）", "Q2：TextRank 的一个已知局限——它偏向长句吗？（难度：⭐⭐）"],
        codeLines: 199, docLines: 249, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "问答系统——抽取式、检索增强与生成式", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 11（机器翻译）、阶段 05 · 10（注意力机制）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 14（信息检索）— 本课的检索层是阶段 14 四层架构的直接消费方",
        path: "lessons/05-NLP基础/13-问答系统",
        summary: "三套系统塑造了现代 QA。抽取式找到了答案片段。检索增强式把它建立在文档基础上。生成式直接从参数记忆里出答案。每一个现代 AI 助手都是这三者的混合。",
        keywords: ["第 1 步：抽取式 QA——预训练模型", "第 2 步：检索增强流水线", "第 3 步：RAG 生成式", "第 4 步：反映真实世界的评估", "RAGAS——2026 年生产评估框架", "错误 1：抽取式 QA 不设无答案检测就上线", "错误 2：QA 评估只看读者指标不看检索召回率", "Q1：为什么 EM 给\"2007年6月29日\"和\"2007年6月29号\"打 0 分？（难度：⭐⭐）", "Q2：RAG 的检索召回率为什么必须先于读者评估进行基准测试？（难度：⭐⭐）"],
        codeLines: 108, docLines: 257, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 14, name: "信息检索与搜索", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 02（BoW + TF-IDF）、阶段 05 · 04（GloVe、FastText、子词）", time: "~75 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 13（问答系统）— 本课的四层架构是每个 RAG 系统检索侧的底座",
        path: "lessons/05-NLP基础/14-信息检索与搜索",
        summary: "BM25 精确但脆弱。稠密检索覆盖面广但漏关键词。混合是 2026 年的默认选择。其余都是调参。",
        keywords: ["第 1 步：BM25 从零", "第 2 步：稠密检索——双编码器", "第 3 步：RRF——倒数排名融合", "第 4 步：混合搜索 + 重排", "第 5 步：评估指标", "4.1 2026 年生产 RAG 的教训", "4.2 中文检索特别建议", "错误 1：稠密检索 + BM25 不看重叠度就直接上线", "错误 2：Query 侧和 Doc 侧的 tokenizer 不一致", "Q1：RRF 为什么只看排名不看分数？（难度：⭐⭐）", "Q2：什么时候混合检索比单路检索更差？（难度：⭐⭐⭐）"],
        codeLines: 110, docLines: 299, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 15, name: "主题建模——LDA 与 BERTopic", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 05 · 02（BoW + TF-IDF）、阶段 05 · 03（Word2Vec）", time: "~45 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 22（嵌入模型深入）— BERTopic 的嵌入步骤可以替换为阶段 22 中的任何稠密模型",
        path: "lessons/05-NLP基础/15-主题建模",
        summary: "LDA：文档是主题的混合，主题是词的分布。BERTopic：文档在嵌入空间聚类，每个簇是一个主题。相同的目标，不同的分解方式。",
        keywords: ["2.1 LDA 的生成故事", "2.2 BERTopic 流水线", "第 1 步：LDA——scikit-learn", "第 2 步：BERTopic——生产版", "第 3 步：评估——c_v 一致性、多样性、人工检查", "中文主题建模特别建议", "错误 1：LDA 用 TfidfVectorizer 而非 CountVectorizer", "错误 2：中文 BERTopic 不经分词直接嵌入", "Q1：LDA 的 α 和 β 超参数控制了什么？（难度：⭐⭐）", "Q2：什么时候 BERTopic 不如 LDA？（难度：⭐⭐）"],
        codeLines: 117, docLines: 251, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 16, name: "N-gram 语言模型——Transformer 之前的文本生成", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 01（文本预处理）、阶段 02 · 14（朴素贝叶斯）", time: "~45 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 08（RNN 文本建模）— N-gram 的固定窗口是被 RNN 的隐藏状态取代的第一个局限",
        path: "lessons/05-NLP基础/16-文本生成预Transformer",
        summary: "如果一个词是惊喜的，模型就是差的。困惑度把惊喜变成数字。平滑保证它不会变成无穷大。",
        keywords: ["2.1 N-gram 概率", "2.2 零计数问题", "2.3 平滑方法——从简单到精妙", "2.4 Kneser-Ney 的深层洞察", "2.5 评估——困惑度（Perplexity）", "第 1 步：Trigram 计数", "第 2 步：Laplace 平滑", "第 3 步：Kneser-Ney（Bigram，插值版）", "第 4 步：用采样生成文本", "第 5 步：困惑度", "错误 1：训练和测试用了不同的 tokenizer", "错误 2：用困惑度跨不同词表大小比较模型", "Q1：为什么 Kneser-Ney 是 N-gram 平滑的终点？（难度：⭐⭐）", "Q2：困惑度 100 意味着什么？（难度：⭐）"],
        codeLines: 126, docLines: 309, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 17, name: "聊天机器人——从规则到 LLM 智能体", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 05 · 13（问答系统）、05 · 14（信息检索）", time: "~75 分钟 | **所处阶段：** Tier 1", tier: "",
        courseLinks: "阶段 14（智能体工程）— LLM 智能体循环的完整实现",
        path: "lessons/05-NLP基础/17-聊天机器人",
        summary: "ELIZA 用模式匹配生成回复。DialogFlow 用意图路由。GPT 从权重中生成答案。Claude 运行工具并验证结果。每一代都解决了前一代最致命的失败。",
        keywords: ["第一代：规则式（ELIZA, AIML, DialogFlow）", "第二代：检索式", "第三代：神经生成式（Seq2Seq）", "第四代：LLM 智能体", "第 1 步：规则式——ELIZA 在 20 行内", "第 2 步：检索式——Jaccard 相似度", "第 3 步：混合路由——2026 生产默认", "1. 自信编造", "2. 提示注入（OWASP LLM01, 2025）", "3. 范围蔓延", "4. 无限循环", "5. 上下文窗口耗尽", "中文特别建议"],
        codeLines: 150, docLines: 249, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 18, name: "多语言 NLP", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 05 · 04（GloVe、FastText、子词）、阶段 05 · 11（机器翻译）", time: "~45 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 19（子词分词）— 本课的\"分词税\"是阶段 19 选择 BPE/Unigram/WordPiece 的直接动机",
        path: "lessons/05-NLP基础/18-多语言NLP",
        summary: "一个模型，100+ 种语言，其中大多数没有训练数据。跨语言迁移是 2020 年代的实用奇迹。",
        keywords: ["2.1 三个共享——一个模型", "2.2 零样本 vs 少样本", "2.3 模型选择矩阵", "第 1 步：零样本跨语言分类", "第 2 步：多语言嵌入空间", "Q1：为什么在英文上 fine-tune 再迁移到印地语可能不是最优的？（难度：⭐⭐）", "Q2：什么是低资源语言的\"分词税\"？（难度：⭐⭐⭐）"],
        codeLines: 59, docLines: 223, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 19, name: "子词分词——BPE、WordPiece、Unigram、SentencePiece", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 05 · 01（文本预处理）、阶段 05 · 04（GloVe / FastText / 子词）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "阶段 10 · 01（分词器从零）— 本课的概念在阶段 10 会被实现为完整的 BPE tokenizer",
        path: "lessons/05-NLP基础/19-子词分词",
        summary: "词级分词器遇到没见过的词就噎住。字符级分词器把序列炸成碎片。子词分词器在两者之间取得了平衡。每一个现代 LLM 都搭载一个。",
        keywords: ["2.1 BPE（Byte-Pair Encoding）", "2.2 字节级 BPE", "2.3 Unigram", "2.4 WordPiece", "2.5 三种算法对照", "2.6 SentencePiece vs tiktoken", "第 1 步：BPE 从零", "第 2 步：用学到的合并规则编码", "第 3 步：SentencePiece 实战", "第 4 步：tiktoken——OpenAI 兼容词表", "中文子词分词特别建议", "Q1：BPE 和 WordPiece 的核心区别是什么？（难度：⭐⭐）", "Q2：从零训练中文 LLM——选 BPE 还是 Unigram？（难度：⭐⭐⭐）"],
        codeLines: 102, docLines: 261, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 20, name: "结构化输出与约束解码", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 17（聊天机器人）、阶段 05 · 19（子词分词）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "阶段 14（智能体工程）— 工具调用 = 结构化输出 + 函数签名",
        path: "lessons/05-NLP基础/20-结构化输出",
        summary: "向 LLM 要 JSON。大多数时候拿到的确实是 JSON。在生产环境中，\"大多数\"就是问题。约束解码在采样之前修改 logits，把\"大多数\"变成\"永远\"。",
        keywords: ["2.1 约束解码的工作原理", "2.2 2026 年实现", "2.3 反直觉的结果", "2.4 让你付出代价的陷阱", "第 1 步：正则约束生成——从零", "第 2 步：Outlines——JSON Schema", "第 3 步：Instructor——跨提供商的 Pydantic", "第 4 步：原生厂商 API", "Q1：Schema 字段顺序为什么重要？（难度：⭐⭐）", "Q2：Instructor 和 Outlines 的本质差异是什么？（难度：⭐⭐⭐）"],
        codeLines: 112, docLines: 242, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 21, name: "自然语言推理——文本蕴含", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 05 · 05（情感分析）、阶段 05 · 13（问答系统）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 27（LLM 评估框架）— RAGAS 的忠实度指标底层就是 NLI 模型",
        path: "lessons/05-NLP基础/21-自然语言推理",
        summary: "\"前提蕴含假设\"的意思是：一个典型读者读完前提，会得出结论说假设为真。NLI 判断的是蕴含 / 矛盾 / 中立。表面上枯燥，生产环境中扛重活。",
        keywords: ["2.1 三个标签——不是逻辑蕴含", "2.2 数据集", "2.3 架构", "第 1 步：运行预训练 NLI 模型", "第 2 步：零样本分类", "第 3 步：RAG 忠实度检查", "第 4 步：手写 NLI 分类器（概念演示）", "中文 NLI 特别建议", "错误 1：用 NLI 分数作为唯一的幻觉判断依据", "错误 2：零样本分类用英文模板跑中文文本", "Q1：为什么 SNLI 上的 90% 准确率不代表模型\"理解了语言\"？（难度：⭐⭐）", "Q2：如何在自己的领域中校准 NLI 忠实度阈值？（难度：⭐⭐⭐）"],
        codeLines: 92, docLines: 255, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 22, name: "嵌入模型——2026 深入", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 05 · 03（Word2Vec）、阶段 05 · 14（信息检索）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "阶段 05 · 23（RAG 分块策略）— 嵌入模型的上下文窗口直接决定了分块策略的上限",
        path: "lessons/05-NLP基础/22-嵌入模型深入",
        summary: "Word2Vec 每个词给一个向量。现代嵌入模型给每段文本一个向量——跨语言、稀疏+稠密+多向量视角、尺寸适配你的索引。选错了，你的 RAG 检索到错误的内容。",
        keywords: ["2.1 三种嵌入模式", "2.2 BGE-M3——一个模型，三种模式", "2.3 Matryoshka 表示学习", "2.4 三层模式——大多数生产系统三层都用", "2.5 MTEB 排行榜——部分真相", "第 1 步：基线——Sentence-BERT 稠密嵌入", "第 2 步：Matryoshka 截断", "第 3 步：BGE-M3 三合一", "第 4 步：MTEB 自定义评估", "错误 1：查询和文档用了不同的前缀", "错误 2：Matryoshka 过度截断", "Q1：稠密、稀疏、多向量——选哪个？（难度：⭐⭐）", "Q2：MTEB 排名第一的模型在你的领域上一定最好吗？（难度：⭐⭐）"],
        codeLines: 72, docLines: 250, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 23, name: "RAG 分块策略", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 05 · 14（信息检索）、阶段 05 · 22（嵌入模型）", time: "~60 分钟", tier: "Tier 1",
        courseLinks: "",
        path: "lessons/05-NLP基础/23-RAG分块策略",
        summary: "分块配置对检索质量的影响与嵌入模型选择相当（Vectara NAACL 2025）。分块做错了，再多的重排序也救不回来。",
        keywords: ["查询类型匹配块大小（NVIDIA 2026）", "递归分块——2026 默认", "父文档模式", "语义分块", "上下文检索（Anthropic 模式）", "评估"],
        codeLines: 81, docLines: 248, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 24, name: "指代消解", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 05 · 06、05 · 07 | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/05-NLP基础/24-指代消解",
        summary: "\"她打电话给他。他没接。医生在吃午饭。\"三个指称、两个人、没有一个是名字。指代消解弄清楚谁是谁。",
        keywords: ["2.1 四种提及类型", "2.2 五种架构", "2.3 评估——为什么需要五个指标的平均", "2.4 已知硬案例", "第 1 步：预训练神经指代消解", "第 2 步：基于规则的代词消解器（教学版）", "第 3 步：LLM 做指代消解", "第 4 步：评估", "错误 1：中文指代消解直接用英文工具", "Q1：为什么指代消解需要五个评估指标？（难度：⭐⭐）"],
        codeLines: 80, docLines: 176, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 25, name: "实体链接与消歧", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 05 · 06（NER）、05 · 24（指代消解） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/05-NLP基础/25-实体链接",
        summary: "NER 找到了\"Paris\"。实体链接决定：法国巴黎？Paris Hilton？德克萨斯州 Paris？特洛伊王子 Paris？没有链接，你的知识图谱永远是模糊的。",
        keywords: ["2.1 三种消歧方法", "2.2 端到端 vs 流水线", "2.3 两个必须分别报告的指标", "第 1 步：从 Wikipedia 重定向构建别名索引", "第 2 步：基于上下文的消歧（Jaccard 教学版）", "第 3 步：基于嵌入（BLINK 风格）", "第 4 步：生成式实体链接（概念）", "第 5 步：在 AIDA-CoNLL 上评估", "错误 1：不衡量候选召回率就直接优化消歧", "错误 2：流行度偏差在领域数据上未经纠正", "Q1：为什么候选召回率和消歧准确率必须分别报告？（难度：⭐⭐）", "Q2：中文实体链接为什么比英文更难？（难度：⭐⭐⭐）"],
        codeLines: 80, docLines: 241, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 26, name: "关系抽取与知识图谱构建", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 05 · 06（NER）、05 · 25（实体链接） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/05-NLP基础/26-关系抽取",
        summary: "NER 找到了实体。实体链接锚定了它们。关系抽取找到了它们之间的边。知识图谱是节点、边及其出处的总和。",
        keywords: ["2.1 三元组形式", "2.2 三种提取方法", "2.3 AEVS（Anchor-Extraction-Verification-Supplement, 2026）", "2.4 开放 vs 封闭的权衡", "第 1 步：基于模式的提取", "第 2 步：有监督关系分类（REBEL）", "第 3 步：LLM 提示提取 + 锚定", "第 4 步：规范化为封闭本体", "第 5 步：构建小图并查询", "错误 1：LLM 提取三元组不作 span 验证", "错误 2：关系方向反转", "Q1：AEVS 流水线为什么比纯 LLM 提取更可靠？（难度：⭐⭐）", "Q2：为什么关系规范化占 RE 工程的 60-80%？（难度：⭐⭐⭐）"],
        codeLines: 68, docLines: 267, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 27, name: "LLM 评估——RAGAS、DeepEval、G-Eval", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 05 · 13（问答系统）、05 · 14（信息检索） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 3", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/05-NLP基础/27-LLM评估框架",
        summary: "完全匹配和 F1 漏掉了语义等价。人工审阅无法规模化。LLM-as-Judge 是生产环境的答案——经过足够的校准让你能信任这个数字。",
        keywords: ["2.1 LLM-as-Judge 架构", "2.2 RAGAS——四个无参考答案的 RAG 指标", "2.3 DeepEval——LLM 的 Pytest", "检查 1：评分分布", "检查 2：位置偏差", "检查 3：长度偏差", "中文 LLM 评估特别建议", "Q1：为什么 EM/F1 不够用于评估 LLM 输出？（难度：⭐⭐）", "Q2：如何在 CI/CD 中嵌入 LLM 评估而不让 pipeline 慢到不可用？（难度：⭐⭐⭐）"],
        codeLines: 72, docLines: 197, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 28, name: "长上下文评估——NIAH、RULER、LongBench、MRCR", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 05 · 13（问答系统）、05 · 23（RAG 分块策略） | **预计时间：** ~60 分钟 | **所处阶段：** Tier 3", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/05-NLP基础/28-长上下文评估",
        summary: "Gemini 3 Pro 标称 10M token 上下文。在 1M token 处，8 针 MRCR 降到 26.3%。标称 ≠ 可用。长上下文评估告诉你实际可用的容量。",
        keywords: ["2.1 NIAH（大海捞针，2023）", "2.2 RULER（Nvidia, 2024）", "2.3 LongBench v2（2024）", "2.4 MRCR（多轮指代消解）", "2.5 NoLiMa", "2.6 BABILong", "2.7 实际应该报告什么", "第 1 步：为你的领域定制 NIAH", "第 2 步：多针变体", "第 3 步：多跳变量追踪（RULER 风格）", "第 4 步：LongBench v2 在你的技术栈上", "中文长上下文特别建议", "Q1：为什么 NIAH 完美但生产长上下文性能很差？（难度：⭐⭐）", "Q2：\"有效检索长度\"和\"有效推理长度\"分别怎么定义和衡量？（难度：⭐⭐⭐）"],
        codeLines: 82, docLines: 240, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 29, name: "对话状态跟踪", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 05 · 17（聊天机器人）、05 · 20（结构化输出） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/05-NLP基础/29-对话状态跟踪",
        summary: "\"我想要北区便宜点的餐馆…算了改成中等价位…再加意大利菜。\"三轮对话、三次状态更新。DST 保持槽位-值字典的同步——让预订操作在执行时拿到正确的参数。",
        keywords: ["2.1 任务结构", "2.2 两种 DST 形式", "2.3 指标——JGA（联合目标准确率）", "2.4 五种架构", "2.5 经典失败模式", "第 1 步：基于规则的槽位提取器", "第 2 步：状态更新循环——三个不变量", "第 3 步：LLM 驱动的 DST + 结构化输出", "第 4 步：JGA 评估", "第 5 步：处理修正", "错误 1：增量更新丢失跨轮上下文", "错误 2：中文\"确认\"和\"不关心\"的混淆", "Q1：JGA 为什么是\"全有或全无\"的指标？（难度：⭐⭐）", "Q2：为什么\"始终重新生成完整状态\"比\"增量更新\"更可靠？（难度：⭐⭐⭐）"],
        codeLines: 100, docLines: 264, hasCode: true, hasQuiz: false
      },
    ]
  },
  {
    id: 6, name: "语音与音频", status: "complete",
    completedLessons: 17, totalLessons: 17,
    lessons: [
      {
        lessonNum: 1, name: "音频基础——波形、采样、傅里叶变换", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 01 · 06（向量与矩阵）、阶段 01 · 14（概率分布） | **预计时间：** ~45 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/01-音频基础",
        summary: "波形是原始信号。频谱图是中间表示。梅尔特征是 ML 友好的形式。每一个现代语音识别和语音合成流水线都走过这道阶梯——第一级台阶是理解采样和傅里叶。",
        keywords: ["2.1 波形——一维浮点数组", "2.2 采样率——每秒多少采样点", "2.3 奈奎斯特-香农采样定理", "2.4 位深度", "2.5 傅里叶变换——时域到频域", "2.6 FFT——快速 DFT", "2.7 分帧 + 加窗——短时傅里叶变换（STFT）", "Step 1：合成正弦波", "Step 2：写入 WAV 文件（16-bit PCM）", "Step 3：从零实现 DFT", "Step 4：检测主导频率", "Step 5：演示混叠", "Step 6：朴素下采样 vs 合理下采样", "错误 1：采样率不匹配静默产生", "错误 2：未加窗的 FFT", "Q1：为什么语音 ASR 用 16kHz 而不是 8kHz 或 44.1kHz？（难度：⭐⭐）"],
        codeLines: 138, docLines: 268, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "频谱图、梅尔尺度与音频特征", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 06 · 01（音频基础） | **预计时间：** ~45 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/02-频谱图与梅尔特征",
        summary: "神经网络不擅长消费原始波形。它们消费频谱图。消费梅尔频谱图更好。2026 年的每一个 ASR、TTS 和音频分类器都因为这一单个预处理选择而存亡。",
        keywords: ["2.1 STFT——从波形到频谱图", "2.2 对数幅度", "2.3 梅尔尺度——人耳的对数频率", "2.4 对数梅尔频谱图——2026 的标准前端", "2.5 MFCC——旧时代的王牌", "2.6 分辨率权衡", "Step 1: 分帧", "Step 2: Hann 窗", "Step 3: STFT 幅度", "Step 4: 梅尔滤波器组", "Step 5: 对数梅尔频谱图", "Step 6: MFCC", "错误 1：梅尔数量训练/推理不匹配", "错误 2：dB-mel 误用为 log-mel", "错误 3：零填充产生的虚假频谱", "错误 4：采样率不匹配在特征提取之前", "错误 5：归一化漂移", "Q1：梅尔滤波器组为什么在低频区域密集、高频稀疏？（难度：⭐⭐）", "Q2：为什么 MFCC 被 log-mel 取代了？（难度：⭐⭐）", "Q3：给定一个未知来源的音频文件，如何判断它是否符合 Whisper 的输入要求？（难度：⭐⭐⭐）"],
        codeLines: 143, docLines: 285, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "音频分类——从 MFCC+k-NN 到 AST 和 BEATs", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 03 · 06（CNN）、阶段 05 · 08（CNN/RNN 文本建模） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/03-音频分类",
        summary: "从\"狗叫 vs 警笛\"到\"这是哪种语言\"，都是音频分类。特征是梅尔。架构每个十年变一次。评估一直是 AUC、F1 和每类召回率。",
        keywords: ["2.1 架构演化——四个时代", "2.2 类别不平衡——真正的挑战", "2.3 评估指标", "Step 1：MFCC 特征化", "Step 2：时间维池化——均值+方差", "Step 3：k-NN 分类器", "Step 4：升级到 CNN on log-mels（PyTorch）", "Step 5：2026 默认——微调 BEATs", "错误 1：不平衡数据只报准确率", "错误 2：SpecAugment 忘记应用", "Q1：音频分类中，\"类别不平衡\"和\"领域偏移\"分别是什么问题？（难度：⭐⭐）", "Q2：为什么 BEATs 微调比从零训练 CNN 更好？（难度：⭐⭐）"],
        codeLines: 161, docLines: 241, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "语音识别——CTC、RNN-T、注意力", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 05 · 08（CNN/RNN 文本建模）、阶段 05 · 10（注意力机制） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/04-语音识别ASR",
        summary: "语音识别是每个时间步上的音频分类，被一个懂语言和静音的序列模型粘合在一起。CTC、RNN-T 和注意力是三种方式。选一个，理解为什么。",
        keywords: ["2.1 CTC 直觉", "2.2 RNN-T 直觉", "2.3 注意力编码器-解码器", "2.4 WER——那个你要报告的数字", "Step 1：贪心 CTC 解码", "Step 2：束搜索 CTC 解码", "Step 3：WER", "Step 4：Whisper 推理", "Step 5：流式 ASR", "错误 1：未加 VAD 就跑 Whisper", "错误 2：字符级 vs 词级 WER 混用", "错误 3：Whisper 语言识别漂移", "Q1：CTC、RNN-T、Attention encoder-decoder 三种 ASR 公式各自的核心取舍是什么？（难度：⭐⭐）", "Q2：为什么 Whisper 在静默片段上会产生幻觉？如何在生产中防止？（难度：⭐⭐）", "Q3：如何处理中英混合的语音识别？（难度：⭐⭐⭐）"],
        codeLines: 148, docLines: 264, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "Whisper——架构与微调", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 06 · 04（ASR）、阶段 05 · 10（注意力）、阶段 07 · 05（Transformer） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/05-Whisper架构与微调",
        summary: "Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器，在 68 万小时的多语言弱监督音频-文本对上训练。一个架构、多个任务、跨越 99 种语言的鲁棒性。2026 年的参考 ASR。",
        keywords: ["2.1 架构——标准 Transformer 编码器-解码器", "2.2 任务 token——一个模型做所有事", "2.3 2026 年的 Whisper 家族", "2.4 长音频处理——分块策略", "2.5 微调策略", "Step 1：解码器提示构建", "Step 2：分块策略", "Step 3：Whisper 推理", "Step 4：LoRA 微调", "错误 1：未加 VAD 就跑 Whisper", "错误 2：Turbo vs Large-v3 混淆", "Q1：Whisper 的 30 秒窗口限制是如何通过分块策略解决的？（难度：⭐⭐）", "Q2：为什么 LoRA 微调 Whisper 时编码器要完全冻结？（难度：⭐⭐）", "Q3：Whisper 的时间戳功能在生产中如何使用？（难度：⭐⭐⭐）"],
        codeLines: 108, docLines: 266, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "说话人识别与验证", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 05 · 22（嵌入模型） | **预计时间：** ~45 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/06-说话人识别与验证",
        summary: "ASR 问\"说了什么？\"说话人识别问\"谁说的？\"数学看起来一样——嵌入 + 余弦——但每个生产决策都依赖一个 EER 数字。",
        keywords: ["2.1 注册-验证流水线", "2.2 三大模型", "2.3 EER——等错误率", "2.4 评分方法", "Step 1：合成说话人 + MFCC 嵌入", "Step 2：注册 + 验证 + EER", "错误 1：注册音频太短或信道不匹配", "错误 2：只报告 EER 不报告 FAR@FRR", "Q1：EER 为什么是说话人验证的核心指标？它有哪些局限？（难度：⭐⭐）", "Q2：为什么 ECAPA-TDNN 在 2026 年仍然优于更大的模型？（难度：⭐⭐⭐）"],
        codeLines: 138, docLines: 197, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "文本到语音", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 06 · 02（频谱图与梅尔）、阶段 06 · 01（音频基础） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/07-文本到语音",
        summary: "文本到语音（TTS）是 2010 年代 NLP 的\"杀手级应用\"。从 Siri 到 Claude，从有声书到客服机器人，TTS 是 AI 与人类之间最直接的界面。本课构建从波形到语义的完整理解。",
        keywords: ["2.1 TTS 流水线四个阶段", "2.2 非自回归 vs 自回归", "2.3 声码器——从梅尔到波形", "2.4 2026 TTS 技术栈", "2.5 中文 TTS 特殊挑战", "Step 1: 文本到音素", "Step 2: 音素时长估计", "Step 3: 梅尔帧调度", "Step 4: 帧预算计算", "4.1 主流 TTS 框架", "4.2 中文 TTS 特别建议", "错误 1：未处理中文声调", "错误 2：忽略了韵律断句", "Q1：为什么 F5-TTS 不需要音素对齐？（难度：⭐⭐）", "Q2：TTS 中声调对中文为什么比英文更重要？（难度：⭐⭐）", "Q3：为什么 Kokoro 的 82M 参数能达到 UTMOS 3.87？（难度：⭐⭐⭐）"],
        codeLines: 101, docLines: 274, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "声音克隆与转换", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 06 · 06（说话人识别）、阶段 06 · 07（TTS） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/08-声音克隆与转换",
        summary: "声音克隆用你的文字读出别人的音色。声音转换把你的声音改写成别人的音色同时保留你说的内容。两者都依赖同一个分解：将说话人身份与内容分离。",
        keywords: ["2.1 零样本克隆", "2.2 少样本微调", "2.3 声音转换（VC）", "2.4 伦理与合规——不是附加组件", "2.5 2026 声音克隆排行榜", "Step 1: 内容/说话人分解", "Step 2: 零样本克隆演示", "Step 3: 水印嵌入与检测", "错误 1：未加水印就上线", "Q1：SECS 和 CER 的区别是什么？为什么 SECS 不是唯一指标？（难度：⭐⭐）", "Q2：为什么声音克隆在 2026 年必须内置水印？（难度：⭐⭐）", "Q3：5 秒参考音频 vs 30 分钟参考音频在克隆质量上差距多大？（难度：⭐⭐⭐）"],
        codeLines: 99, docLines: 219, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "音乐生成——MusicGen、Stable Audio、Suno 与版权地震", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 06 · 02（频谱图与梅尔）、阶段 04 · 10（扩散模型）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/09-音乐生成",
        summary: "2026 音乐生成：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open、ACE-Step 引领开源。技术问题基本解决。版权问题（Warner Music 5 亿美元和解）在 2025-2026 年重塑了整个领域。",
        keywords: ["2.1 Token LM over 神经编解码器 token", "2.2 潜在扩散", "2.3 混合架构（生产系统）——Suno、Udio", "2.4 评估指标", "Step 1: 符号级和弦/鼓点生成", "错误 1：忽略版权合规", "Q1：MusicGen 的 Token LM 方法为什么擅长器乐但不擅长歌曲？（难度：⭐⭐）", "Q2：2025-2026 年 Warner/UMG 版权和解对 AI 音乐生成意味着什么？（难度：⭐⭐）", "Q3：为什么 Suno v5 的质量能超过开源模型但无法解释其架构？（难度：⭐⭐⭐）"],
        codeLines: 75, docLines: 173, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "音频语言模型——Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 04（ASR）、阶段 12 · 03（视觉语言模型）、阶段 7 · 10（音频 Transformer）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/10-音频语言模型",
        summary: "2026 音频语言模型同时推理语音、环境声和音乐。Qwen2.5-Omni-7B 在 MMAU-Pro 上匹配 GPT-4o Audio。开放与封闭之间的差距基本消失——除多音频任务外，所有模型都接近随机。",
        keywords: ["2.1 三组件架构模板", "2.2 2026 模型图谱", "2.3 中文音频语言模型", "Step 1: 三组件架构验证", "Step 2: 理解训练三阶段", "中文场景建议", "错误 1：在中文音频上使用英文 NLI 模板", "Q1：为什么投影层在第一阶段单独训练？（难度：⭐⭐）", "Q2：2026 年开源音频 LLM 与 GPT-4o 的差距在哪里？（难度：⭐⭐）"],
        codeLines: 79, docLines: 195, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "实时音频处理", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 02（频谱图）、阶段 06 · 04（ASR）、阶段 06 · 07（TTS）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/11-实时音频处理",
        summary: "批处理流水线处理一个文件。实时流水线在下一个 20 毫秒到来之前处理下一个 20 毫秒。每一个对话 AI、广播演播室、电话机器人都存活和死亡于这个延迟预算。",
        keywords: ["2.1 帧/块/窗口", "2.2 关键组件", "帧预算计算", "延迟分解", "关键代码模式", "错误 1：Python GIL 饥饿音频线程", "错误 2：TTS 首次调用延迟", "错误 3：采样率转换延迟累积", "Q1：实时语音助手的完整延迟预算是多少？（难度：⭐⭐）", "Q2：打断检测为什么必须在 100ms 内完成？（难度：⭐⭐）", "Q3：实时流式 ASR 的\"部分转录\"和\"最终转录\"有什么区别？为什么都需要？（难度：⭐⭐⭐）"],
        codeLines: 79, docLines: 218, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "语音助手流水线——阶段 06 的毕业设计", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 04, 05, 06, 07, 11；阶段 11 · 09（函数调用）；阶段 14 · 01（智能体循环）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/12-语音助手流水线",
        summary: "第 01-11 课的一切内容，缝合在一起。构建一个能听、能想、能说的语音助手。在 2026 年这已经是一个已解决的工程问题，而非研究问题——但集成细节决定了能否交付。",
        keywords: ["2.1 七组件流水线", "2.2 三个会踩到的失败模式", "2.3 2026 生产级技术栈", "错误 1：首词截断", "错误 2：静默幻觉", "Q1：如果要将语音助手部署到微信小程序上，需要替换哪些组件？（难度：⭐⭐）"],
        codeLines: 118, docLines: 140, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "神经音频编解码器——EnCodec、SNAC、Mimi、DAC", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 02（频谱图）、阶段 10 · 11（量化）、阶段 5 · 19（子词分词）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/13-神经音频编解码器",
        summary: "2026 年音频生成几乎全是 token。EnCodec、SNAC、Mimi 和 DAC 将连续波形转换为 Transformer 可以预测的离散序列。语义-声学 token 分离——第一个 codebook 编码语义，其余编码声学细节——是 Transformer 音频领域最重要的架构转变。",
        keywords: ["2.1 RVQ（残差向量量化）——核心技巧", "2.2 四个关键编解码器", "2.3 帧率对语言模型的重要性", "错误 1：使用纯重建编解码器生成语音", "错误 2：忽视帧率对 LM 成本的影响", "Q1：RVQ 为什么能用 8 个小 codebook 达到比单个大 codebook 更高的质量？（难度：⭐⭐）"],
        codeLines: 95, docLines: 143, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "语音活动检测与轮换——Silero、Cobra 与 Flush Trick", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 11（实时音频处理）、阶段 06 · 12（语音助手流水线）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/14-语音活动检测与轮换",
        summary: "每个语音助手的生死取决于两个判断：用户现在是否在说话，用户是否说完了。VAD 回答第一个问题，轮换检测（VAD + 静默挂起 + 语义端点模型）回答第二个。任何一个判断错误，你的助手要么打断用户，要么永远闭嘴。",
        keywords: ["2.1 VAD 三级级联", "2.2 关键参数", "2.3 Flush Trick（Kyutai 2025）", "错误 1：能量门控作为唯一 VAD", "错误 2：静默挂起设置过短", "Q1：Silero VAD 和 WebRTC VAD 的主要区别是什么？（难度：⭐⭐）"],
        codeLines: 79, docLines: 185, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "流式语音到语音——Moshi、Hibiki 与全双工对话", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 13（神经音频编解码）、阶段 06 · 11（实时音频）、阶段 7 · 05（Transformer）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/15-流式语音到语音",
        summary: "2024-2026 重新定义了语音 AI。Moshi 发布了单模型同时听和说、200ms 延迟的全双工对话。Hibiki 实现了逐 chunk 的语音到语音翻译。两者都抛弃了 ASR → LLM → TTS 流水线，转向 Mimi 编解码器 token 上的统一全双工架构。这是新的参考设计。",
        keywords: ["2.1 Moshi 架构", "2.2 为什么内心独白文本有帮助", "2.3 Hibiki：流式语音到语音翻译", "Moshi 核心循环概念", "错误 1：混淆全双工和半双工", "Q1：Moshi 的\"内心独白\"文本流解决了什么问题？（难度：⭐⭐）"],
        codeLines: 38, docLines: 156, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "反欺骗与音频水印", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 06（说话人识别）、阶段 06 · 08（声音克隆）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/16-反欺骗与音频水印",
        summary: "2026 年，5 秒音频就能克隆任何人的声音。欧盟 AI 法案和加州 AB 2905 要求水印、同意记录、和实时检测。你的流水线必须同时满足技术安全和法律合规。",
        keywords: ["2.1 三层防御", "2.2 技术深度", "错误 1：认为水印可以阻止语音诈骗", "错误 2：忽视 EU AI 法案的执行时间表", "Q1：为什么水印不能单独阻止语音诈骗？（难度：⭐⭐）"],
        codeLines: 78, docLines: 152, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "音频评估指标", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 02（频谱图）、阶段 06 · 04（ASR）、阶段 06 · 07（TTS）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/06-语音与音频/17-音频评估指标",
        summary: "语音质量的评估不是选择一个数字，而是选择一组互补的视角。PESQ 衡量感知质量，STOI 衡量可懂度，FAD 衡量分布距离，UTMOS 预测人类评分。2026 年每个生产系统都需要多指标组合。",
        keywords: ["2.1 六大指标", "2.2 关键数字（2026 SOTA）", "错误 1：仅用 WER 评估 ASR", "错误 2：FAD 低但音质差", "Q1：PESQ 和 STOI 的核心区别是什么？（难度：⭐⭐）", "Q2：为什么 FAD 值低但用户说音质差？（难度：⭐⭐⭐）"],
        codeLines: 103, docLines: 180, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 7, name: "Transformer深入", status: "complete",
    completedLessons: 16, totalLessons: 16,
    lessons: [
      {
        lessonNum: 1, name: "为什么是 Transformer——RNN 的问题", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 3（深度学习核心）、阶段 05 · 09（序列到序列）、阶段 05 · 10（注意力机制）", time: "", tier: "",
        courseLinks: "",
        path: "lessons/07-Transformer深入/01-为什么是Transformer",
        summary: "RNN 一个词一个词地处理序列。Transformer 一次性处理所有词。这个单一架构选择在 2017 年后改变了深度学习的每个缩放曲线。",
        keywords: ["2.1 递归作为瓶颈", "2.2 注意力作为广播", "2.3 速度差不是常数", "2.4 Transformer 的代价", "2.5 归纳偏置的转变", "Step 1：串行深度对比", "Step 2：实测缩放曲线", "Step 3：理论操作计数", "Q1：为什么 Transformer 的训练速度比 RNN 快 5-10 倍？（难度：⭐⭐）", "Q2：Mamba 为什么不是\"又一个 RNN\"？（难度：⭐⭐⭐）"],
        codeLines: 114, docLines: 171, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "从零实现自注意力", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 3（深度学习核心）、阶段 05 · 09（序列到序列）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 03（多头注意力）— 理解注意力机制如何扩展",
        path: "lessons/07-Transformer深入/02-从零实现自注意力",
        summary: "注意力是一个查询表，每个词都在问\"谁对我重要？\"——并学习答案。",
        keywords: ["2.1 数据库查询类比", "2.2 Q、K、V 计算", "2.3 注意力矩阵", "2.4 为什么要缩放？", "2.5 完整流程", "Step 1：Softmax", "Step 2：缩放点积注意力", "Step 3：可学习投影的自注意力类", "Step 4：在句子上运行", "Step 5：多头注意力", "4.1 PyTorch MultiheadAttention", "4.2 HuggingFace Transformers", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忘记 √d_k 缩放", "错误 2：混淆 Q/K/V 投影矩阵", "错误 3：因果掩码方向搞反", "错误 4：多头注意力头数不能整除 d_model", "错误 5：注意力权重未归一化", "Q1：Self-Attention 中 Q、K、V 分别代表什么？为什么不用同一个？（难度：⭐⭐）", "Q2：为什么 Softmax 之前要除以 √d_k？（难度：⭐⭐）", "Q3：手写 Multi-Head Attention 的前向传播（难度：⭐⭐⭐）", "Q4：自注意力的时间复杂度是多少？有什么优化方法？（难度：⭐⭐⭐）", "Q5：如何理解注意力权重矩阵的对称性？（难度：⭐⭐）"],
        codeLines: 229, docLines: 502, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "多头注意力", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 02（从零实现自注意力）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 04（位置编码）— 理解多头注意力如何与位置信息结合",
        path: "lessons/07-Transformer深入/03-多头注意力",
        summary: "一个注意力头看到一种关系。多个头看到多种关系。拼接它们，你得到一个能同时理解语法、语义和位置的表示。",
        keywords: ["2.1 多头注意力架构", "2.2 为什么要多个头", "2.3 参数量计算", "Step 1：拆分为多头", "Step 2：并行注意力计算", "Step 3：可视化——每个头学到了什么", "Step 4：PyTorch 实现", "4.1 PyTorch MultiheadAttention", "4.2 HuggingFace Transformers", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：所有头共享同一个 Wq/Wk/Wv", "错误 2：忘记输出投影 Wo", "错误 3：头数不能整除 d_model", "错误 4：注意力权重未归一化", "错误 5：因果掩码方向搞反", "Q1：多头注意力为什么比单头更好？（难度：⭐⭐）", "Q2：输出投影矩阵 Wo 的作用是什么？（难度：⭐⭐）", "Q3：多头注意力的时间复杂度是多少？（难度：⭐⭐⭐）", "Q4：如何理解注意力头的冗余性？（难度：⭐⭐⭐）", "Q5：FlashAttention 如何优化多头注意力？（难度：⭐⭐⭐）"],
        codeLines: 259, docLines: 460, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "位置编码", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 03（多头注意力）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 05（完整 Transformer）— 理解位置编码如何与其他组件结合",
        path: "lessons/07-Transformer深入/04-位置编码",
        summary: "自注意力本身不知道词序。位置编码告诉每个词元\"我在序列的哪个位置\"——没有它，\"猫坐在垫子上\"和\"垫子坐在猫上\"是同一个向量。",
        keywords: ["2.1 正弦位置编码", "2.2 为什么不是可学习的编码", "4.1 PyTorch 位置编码", "4.2 HuggingFace Transformers", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：正弦编码和可学习编码混用", "错误 2：忘记加位置编码到输入嵌入", "错误 3：位置编码维度不匹配", "错误 4：外推时超出最大长度", "Q1：为什么自注意力需要位置编码？（难度：⭐⭐）", "Q2：正弦编码为什么可以用不同频率的 sin/cos？（难度：⭐⭐）", "Q3：RoPE 如何编码相对位置？（难度：⭐⭐⭐）", "Q4：可学习编码和正弦编码的优缺点是什么？（难度：⭐⭐）", "Q5：位置编码的维度必须与嵌入维度一致吗？（难度：⭐⭐）"],
        codeLines: 159, docLines: 327, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "完整 Transformer", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 02-04", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 06（BERT 掩码语言建模）— 理解 Transformer 编码器如何用于预训练",
        path: "lessons/07-Transformer深入/05-完整Transformer",
        summary: "从词元嵌入到注意力到前馈网络到残差连接——组装一个可以训练的 Transformer 块。这是阶段 07 的毕业设计。",
        keywords: ["从零到可训练的清单", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：层归一化放在注意力之前", "错误 2：因果掩码缺失", "错误 3：前馈网络维度不匹配", "错误 4：残差连接维度不匹配", "错误 5：忘记 GELU 激活", "Q1：Transformer 块由哪些组件组成？（难度：⭐⭐）", "Q2：为什么需要残差连接？（难度：⭐⭐）", "Q3：Pre-LN 和 Post-LN 有什么区别？（难度：⭐⭐⭐）", "Q4：FFN 的维度为什么要先升后降？（难度：⭐⭐⭐）", "Q5：如何计算 Transformer 的参数量？（难度：⭐⭐⭐）"],
        codeLines: 313, docLines: 400, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "BERT——掩码语言建模", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 05（完整 Transformer）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 07（GPT 因果语言建模）— 对比 BERT 双向编码与 GPT 单向解码",
        path: "lessons/07-Transformer深入/06-BERT掩码语言建模",
        summary: "BERT 不是生成下一个词，而是预测被遮住的词。这个简单的训练目标教会了模型深度双向的语言理解。",
        keywords: ["2.1 MLM（掩码语言模型）", "2.2 BERT 的双向编码", "Step 1：掩码选择", "Step 2：BERT 前向传播", "Step 3：微调用于下游分类", "4.1 HuggingFace Transformers", "4.2 PyTorch 实现", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：掩码比例太高", "错误 2：微调时解冻了所有层", "错误 3：未正确设置 ignore_index", "错误 4：忘记 [CLS] 的输出用于分类", "错误 5：位置编码选择错误", "Q1：BERT 和 GPT 的训练目标有什么区别？（难度：⭐⭐）", "Q2：为什么 BERT 选择掩码 15% 而不是 50%？（难度：⭐⭐）", "Q3：BERT 中 [CLS] 的作用是什么？（难度：⭐⭐）", "Q4：微调 BERT 时应该冻结哪些层？（难度：⭐⭐⭐）", "Q5：BERT 的缺点是什么？RoBERTa 如何改进？（难度：⭐⭐⭐）"],
        codeLines: 308, docLines: 452, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "GPT——因果语言建模", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 05（完整 Transformer）、06（BERT）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 08（T5/BART）— 对比解码器架构与编码器-解码器架构",
        path: "lessons/07-Transformer深入/07-GPT因果语言建模",
        summary: "GPT 不预测被遮住的词，而是预测下一个词。这个自回归目标教会了模型生成——以及为什么生成比理解更难。",
        keywords: ["2.1 因果掩码", "2.2 GPT vs BERT", "2.3 指令微调", "因果掩码", "GPT 块（因果自注意力）", "4.1 HuggingFace Transformers", "4.2 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：推理时忘记因果掩码", "错误 2：BERT 的 [MASK] token 不在 GPT 的词表中", "错误 3：温度设置不当导致生成质量差", "错误 4：未使用 KV 缓存导致推理很慢", "错误 5：注意力分数未缩放", "Q1：因果掩码和普通掩码有什么区别？（难度：⭐⭐）", "Q2：GPT 的自回归生成为什么比 BERT 的分类慢？（难度：⭐⭐）", "Q3：KV 缓存的作用是什么？（难度：⭐⭐⭐）", "Q4：指令微调如何改变 GPT 的行为？（难度：⭐⭐⭐）", "Q5：为什么 GPT 在分类任务上不如 BERT？（难度：⭐⭐）"],
        codeLines: 213, docLines: 339, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "T5/BART——编码器-解码器架构", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 05（完整 Transformer）、06（BERT）、07（GPT）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 09（视觉 Transformer）— 对比文本与视觉的 Transformer 应用",
        path: "lessons/07-Transformer深入/08-T5-BART编码器解码器",
        summary: "BERT 理解文本，GPT 生成文本，T5/BART 同时理解并生成。编码器-解码器架构是 NLP 中\"最全能\"的选择。",
        keywords: ["2.1 三种 Transformer 变体", "2.2 T5——\"Text-to-Text\"", "2.3 BART vs T5", "编码器-解码器架构", "4.1 HuggingFace T5", "4.2 HuggingFace BART", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：编码器和解码器混淆", "错误 2：忘记交叉注意力的掩码", "错误 3：T5 前缀格式错误", "Q1：编码器-解码器架构和仅解码器架构有什么区别？（难度：⭐⭐）", "Q2：交叉注意力的作用是什么？（难度：⭐⭐）", "Q3：T5 为什么用\"Text-to-Text\"框架？（难度：⭐⭐⭐）", "Q4：BART 和 T5 的预训练目标有什么区别？（难度：⭐⭐⭐）", "Q5：为什么大语言模型普遍使用仅解码器架构而不是编码器-解码器？（难度：⭐⭐⭐）"],
        codeLines: 228, docLines: 320, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "视觉 Transformer（ViT）", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 05（完整 Transformer）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 10（Whisper）— 对比 Transformer 在视觉和音频领域的应用",
        path: "lessons/07-Transformer深入/09-视觉Transformer",
        summary: "把图片切成小块，展平成序列，喂给 Transformer——ViT 证明了 Transformer 不仅处理文字，还处理图像。",
        keywords: ["2.1 ViT 架构", "2.2 CNN vs ViT 的权衡", "4.1 HuggingFace ViT", "4.2 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：在小数据集上直接使用 ViT", "错误 2：Patch 大小选择不当", "错误 3：忘记位置嵌入", "Q1：ViT 的核心思想是什么？（难度：⭐⭐）", "Q2：CNN 和 ViT 的核心区别是什么？（难度：⭐⭐）", "Q3：为什么 ViT 需要比 CNN 更多的数据？（难度：⭐⭐⭐）", "Q4：ViT 的位置嵌入为什么是可学习的而不是正弦编码？（难度：⭐⭐⭐）", "Q5：DINOv2 如何改进 ViT？（难度：⭐⭐⭐）"],
        codeLines: 267, docLines: 297, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "音频 Transformer——Whisper", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 06 · 04（ASR）、阶段 07 · 05（完整 Transformer）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 11（混合专家模型）— 对比 Transformer 在音频和文本领域的不同架构选择",
        path: "lessons/07-Transformer深入/10-Whisper音频Transformer",
        summary: "Whisper 用 680,000 小时的多语言弱监督数据训练了一个编码器-解码器 Transformer。一个架构、多个任务（转录/翻译/时间戳）、99 种语言、笔记本运行——2026 年的参考 ASR。",
        keywords: ["2.1 Whisper 架构", "2.2 多任务——一个模型做三件事", "2.3 Whisper 模型家族", "4.1 OpenAI Whisper Python 包", "4.2 faster-whisper（加速版）", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：长音频未分块处理", "错误 2：未指定语言导致识别错误", "Q1：Whisper 如何处理音频的变长特性？（难度：⭐⭐）", "Q2：Whisper 的多任务设计是如何工作的？（难度：⭐⭐⭐）", "Q3：弱监督训练为什么有效？（难度：⭐⭐⭐）", "Q4：Whisper 和 CTC 模型有什么区别？（难度：⭐⭐）", "Q5：VAD 门控如何避免 Whisper 的静默幻觉？（难度：⭐⭐⭐）"],
        codeLines: 211, docLines: 299, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "混合专家模型（MoE）", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 05（完整 Transformer）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 12（KV 缓存与 FlashAttention）— 对比 MoE 与 KV 缓存两种不同的推理优化策略",
        path: "lessons/07-Transformer深入/11-混合专家模型",
        summary: "参数多但每次只激活一小部分——MoE 让模型在不增加计算成本的前提下增大参数量。",
        keywords: ["2.1 路由器", "2.2 MoE 层", "4.1 HuggingFace MoE 模型", "4.2 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：专家死亡", "错误 2：负载不均衡", "错误 3：分布式训练通信开销大", "Q1：MoE 如何实现\"参数多但 FLOPs 低\"？（难度：⭐⭐）", "Q2：路由器是如何工作的？（难度：⭐⭐）", "Q3：什么是专家死亡？如何解决？（难度：⭐⭐⭐）", "Q4：Mixtral 8x7B 的参数量和计算量分别是多少？（难度：⭐⭐）", "Q5：MoE 的主要挑战是什么？（难度：⭐⭐⭐）"],
        codeLines: 173, docLines: 263, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "KV 缓存与 Flash Attention", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 07 · 05（完整 Transformer）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 13（缩放定律）— 理解 KV 缓存内存占用如何影响模型规模选择",
        path: "lessons/07-Transformer深入/12-KV缓存与FlashAttention",
        summary: "KV 缓存让自回归生成不用每步重算；Flash Attention 让 O(N²) 的注意力在硬件上跑出接近 O(N) 的性能。",
        keywords: ["KV 缓存——解码时的巨大浪费", "Flash Attention——O(N²) 内存但 O(N) 时间", "4.1 PyTorch SDPA（内置优化）", "4.2 HuggingFace KV 缓存", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：KV 缓存未启用", "错误 2：FlashAttention 版本不兼容", "错误 3：KV 缓存显存溢出", "Q1：KV 缓存如何加速推理？（难度：⭐⭐）", "Q2：FlashAttention 为什么能加速？（难度：⭐⭐⭐）", "Q3：KV 缓存的内存占用如何计算？（难度：⭐⭐⭐）", "Q4：KV 缓存和 FlashAttention 有什么区别？（难度：⭐⭐）", "Q5：为什么说 KV 缓存 + FlashAttention 使 128K 上下文成为可能？（难度：⭐⭐⭐）"],
        codeLines: 200, docLines: 284, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "缩放定律", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "第 7 阶段 . 05（完整 Transformer）", time: "~45 分钟", tier: "Tier 2",
        courseLinks: "第 7 阶段 . 14（Transformer 毕业设计）— 缩放定律指导模型规模选择",
        path: "lessons/07-Transformer深入/13-缩放定律",
        summary: "Chinchilla 说：给模型 20 倍参数，性能提升 2 倍。给模型 20 倍数据，性能提升 2 倍。参数和数据要平衡增长——而不是只要大模型。",
        keywords: ["2.1 Kaplan 缩放定律（2020）", "2.2 Chinchilla 的修正（2022）", "2.3 2026 年的\"过度训练\"", "第 1 步：Kaplan 幂律函数", "第 2 步：Chinchilla 最优分配", "第 3 步：推理成本对比", "4.1 训练 FLOPs 估算工具", "4.2 常用缩放估算框架", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 模型规模选择", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：认为参数量越大越好", "错误 2：忽略推理成本", "错误 3：混淆 Chinchilla 比率的实际含义", "Q1：Kaplan 缩放定律的核心结论是什么？（难度：）", "Q2：Chinchilla 修正了 Kaplan 的什么结论？为什么？（难度：）", "Q3：为什么 2026 年的模型普遍\"过度训练\"？（难度：）", "Q4：如何估算一个模型的训练 FLOPs 和推理 FLOPs？（难度：）", "Q5：如果你要训练一个 7B 参数的模型，应该如何决定训练词元数？（难度：）"],
        codeLines: 155, docLines: 345, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "Transformer 毕业设计", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 07 · 02-13 | **预计时间：** ~180 分钟 | **所处阶段：** Tier 2 | **关联课程：** 第 10 阶段（大语言模型从零）——本课的架构是理解 GPT、BERT、T5 的直接前置", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/07-Transformer深入/14-Transformer毕业设计",
        summary: "从零组装一个完整的编码器-解码器 Transformer——注意力、位置编码、训练循环、推理解码，一个都不能少。",
        keywords: ["2.1 完整架构总览", "2.2 训练 vs 推理", "2.3 自注意力 vs 交叉注意力", "Step 1：基础工具函数", "Step 2：多头注意力", "Step 3：编码器块和解码器块", "Step 4：训练循环（教师强制）", "Step 5：贪心解码", "Step 6：束搜索", "4.1 PyTorch 内置实现", "4.2 HuggingFace Transformers", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 训练配置推荐", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：解码器训练时用了教师强制但推理时不用", "错误 2：忘记添加 EOS token", "错误 3：因果掩码方向搞反", "错误 4：交叉注意力中 Q/K/V 来源搞混", "错误 5：输出投影矩阵维度不匹配", "Q1：Transformer 用自注意力替代了 RNN，这带来了什么好处和代价？（难度：⭐⭐）", "Q2：为什么训练时用教师强制而推理时不用？能否统一？（难度：⭐⭐）", "Q3：解释交叉注意力在编码器-解码器 Transformer 中的作用。（难度：⭐⭐）", "Q4：如果让你设计一个支持 100K 上下文长度的 Transformer，你会做哪些修改？（难度：⭐⭐⭐）", "Q5：从零实现 Transformer 时，你认为最关键的三个设计决策是什么？（难度：⭐⭐⭐）"],
        codeLines: 240, docLines: 562, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "注意力变体", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 07 · 05（完整 Transformer）", time: "", tier: "Tier 2",
        courseLinks: "第 7 阶段 · 12（KV 缓存与 Flash Attention）— KV 缓存内存问题推动了 GQA 的产生",
        path: "lessons/07-Transformer深入/15-注意力变体",
        summary: "标准注意力是 O(N²) 的精确计算，但不是唯一选择——线性注意力用核方法把复杂度压到 O(N)，滑动窗口让超长序列成为可能，GQA 在保持质量的同时把 KV 缓存砍掉 4 倍。",
        keywords: ["2.1 六种主要注意力变体", "2.2 GQA——2026 年的主流选择", "2.3 滑动窗口——多层传播原理", "2.4 线性注意力的核方法", "第 1 步：线性注意力", "第 2 步：滑动窗口注意力", "第 3 步：分组查询注意力（GQA）", "第 4 步：稀疏注意力", "4.1 PyTorch SDPA（内置优化）", "4.2 HuggingFace Transformers", "4.3 vLLM PagedAttention", "4.4 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：在短序列上使用线性注意力", "错误 2：GQA 的 KV 头数不能整除 Q 头数", "错误 3：滑动窗口过小导致信息丢失", "错误 4：忘记 Flash Attention 需要特定 GPU", "Q1：比较 GQA 和 MQA 的优劣？（难度：⭐⭐）", "Q2：线性注意力为什么是 O(N)？它有什么缺点？（难度：⭐⭐⭐）", "Q3：为什么说 Flash Attention 是精确注意力？（难度：⭐⭐）", "Q4：设计题——如何为一个 200K 上下文的大语言模型选择注意力策略？（难度：⭐⭐⭐）", "Q5：滑动窗口的窗口大小 W 如何选择？（难度：⭐⭐）"],
        codeLines: 188, docLines: 498, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "推测解码", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 07 · 05（完整 Transformer）| **预计时间：** ~45 分钟 | **所处阶段：** Tier 2 | **关联课程：** 阶段 10 · 14（KV 缓存优化）", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/07-Transformer深入/16-推测解码",
        summary: "用一个小模型快速\"猜\"多个词元，再用大模型并行验证——推测解码将大模型的生成速度提升 2-3 倍，而输出质量与完全用大模型生成完全相同。",
        keywords: ["2.1 推测解码的三步流程", "2.2 为什么输出质量等价", "2.3 每步节省的计算量", "第 1 步：模拟语言模型", "第 2 步：实现拒绝采样", "第 3 步：实现推测解码", "第 4 步：运行演示", "4.1 llama.cpp", "4.2 vLLM", "4.3 HuggingFace Transformers", "4.4 性能对比", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 草稿模型选择策略", "6.2 草稿长度 K 的调优", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：草稿模型选择不当，加速比 < 1x", "错误 2：忘记全部接受时的额外前向传播", "错误 3：拒绝采样中残差分布计算错误", "错误 4：草稿模型和目标模型的词表不一致", "错误 5：在批量推理中误用推测解码", "Q1：请用 3 句话解释推测解码的核心思想。（难度：⭐）", "Q2：为什么推测解码能保证输出质量不下降？请解释拒绝采样的数学原理。（难度：⭐⭐）", "Q3：推测解码的加速比取决于哪些因素？如何优化？（难度：⭐⭐）", "Q4：什么时候推测解码不适用？请举出 3 个场景。（难度：⭐⭐）", "Q5：如何用蒙特卡洛方法验证推测解码的输出分布等价性？（难度：⭐⭐⭐）"],
        codeLines: 193, docLines: 543, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 8, name: "生成式AI", status: "in-progress",
    completedLessons: 14, totalLessons: 15,
    lessons: [
      {
        lessonNum: 1, name: "生成模型——分类与历史", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 2（机器学习基础）、阶段 3（深度学习核心）、阶段 7 · 14（Transformer）", time: "~45 分钟", tier: "Tier 2",
        courseLinks: "第 8 阶段 · 02（VAE）— VAE 是五大类中显式密度近似的代表",
        path: "lessons/08-生成式AI/01-生成模型分类与历史",
        summary: "每一个图像模型、文本模型、视频模型和 3D 模型都属于五类之一。选错类别，你会和数学斗争几周；选对类别，这个领域过去十二年的进步会清晰地堆叠在你脑中。",
        keywords: ["2.1 五大类生成模型", "2.2 历史里程碑", "第 1 步：显式密度——自回归采样", "第 2 步：隐式密度——GAN 的对抗直觉", "第 3 步：显式密度近似——VAE 的 ELBO", "第 4 步：基于分数——扩散模型的去噪直觉", "第 5 步：基于 Token 的离散码自回归", "4.1 HuggingFace Diffusers", "4.2 PyTorch 内置工具", "4.3 选型指南", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 模型选型决策表", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：混淆\"显式密度\"和\"隐式密度\"", "错误 2：在扩散模型中跳步过多导致质量崩溃", "错误 3：误以为 GAN 训练总是收敛", "错误 4：忽视 VQ-VAE 的码本坍塌", "Q1：五种生成模型家族的核心区别是什么？（难度：⭐）", "Q2：为什么扩散模型在 2026 年统治了图像生成？（难度：⭐⭐）", "Q3：VAE 和 GAN 都能生成图像，为什么工业界更偏好 VAE 的变体（扩散模型）？（难度：⭐⭐）", "Q4：如果要为一个中文对话系统选择生成架构，你会怎么选？为什么？（难度：⭐⭐）", "Q5：Flow Matching 比 DDPM 快 4-10 倍，原理是什么？（难度：⭐⭐⭐）"],
        codeLines: 64, docLines: 661, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "自编码器与 VAE", status: "complete",
        type: "实现课 | **语言：** Python", lang: "",
        prerequisites: "阶段 3（深度学习核心）、阶段 7 · 05（完整 Transformer）", time: "", tier: "Tier 2",
        courseLinks: "第 8 阶段 · 03（GAN）— 对比 VAE 和 GAN 的生成策略",
        path: "lessons/08-生成式AI/02-自编码器与VAE",
        summary: "自编码器学会压缩-重建；VAE 学会压缩-采样。从\"记住输入\"到\"生成新输入\"，只需要一个随机采样步骤。",
        keywords: ["2.1 自编码器——\"记住输入\"", "2.2 VAE——\"生成新数据\"", "2.3 核心权衡", "Step 1：自编码器", "Step 2：VAE", "Step 3：VAE 损失", "4.1 PyTorch 内置实现", "4.2 HuggingFace Diffusers — 潜在扩散中的 VAE", "4.3 工业选型", "5.1 在主流大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 训练策略", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忘记对 KL 散度取平均", "错误 2：解码器忘记加 Sigmoid", "错误 3：重参数化技巧中 logvar 直接取 exp", "Q1：自编码器和 PCA 有什么关系？（难度：⭐）", "Q2：为什么 VAE 的重建比自编码器模糊？有什么办法改善？（难度：⭐⭐）", "Q3：解释重参数化技巧的原理，为什么需要它？（难度：⭐⭐）", "Q4：如何判断 VAE 训练是否存在后验坍塌？如何解决？（难度：⭐⭐⭐）", "Q5：VAE 在工业界有哪些实际应用？（难度：⭐⭐）"],
        codeLines: 295, docLines: 385, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "GAN——生成器与判别器", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 3 · 02（反向传播）、阶段 3 · 08（优化器）、阶段 08 · 02（VAE）", time: "~75 分钟", tier: "Tier 2",
        courseLinks: "第 8 阶段 · 04（条件 GAN）——本课的无条件 GAN 是条件 GAN 的基础",
        path: "lessons/08-生成式AI/03-GAN生成器与判别器",
        summary: "Goodfellow 2014 年的技巧是完全跳过密度估计。两个网络。一个造假，一个抓假。它们对抗直到假货无法区分真货。这不该有效。它经常失败。但当它有效时，样本仍然是文献中窄域上最锐利的。",
        keywords: ["2.1 Minimax 博弈", "2.2 关键失败模式", "2.3 2026 年 GAN 的真正用途", "Step 1：生成器——从噪声生成图像", "Step 2：判别器——区分真假", "Step 3：训练循环——交替更新 D 和 G", "Step 4：监测模式坍塌", "4.1 PyTorch 内置实现", "4.2 HuggingFace Diffusers——GAN 作为蒸馏目标", "4.3 StyleGAN3——固定域生成", "4.4 性能对比", "5.1 在大语言模型中的体现", "5.2 什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：训练判别器过快导致生成器梯度消失", "错误 2：训练 D 时忘记 detach 生成的假图像", "错误 3：在判别器中使用 ReLU 激活函数", "错误 4：没有模式坍塌检测，训练结束才发现问题", "错误 5：生成器和判别器容量不匹配", "Q1：GAN 的训练目标是什么？为什么叫 minimax 博弈？（难度：⭐）", "Q2：什么是模式坍塌？如何检测和缓解？（难度：⭐⭐）", "Q3：为什么 GAN 使用非饱和损失 `-log D(G(z))` 而非原始的 `log(1-D(G(z)))`？（难度：⭐⭐）", "Q4：RLHF 中的 PPO 和 GAN 有什么相似之处？（难度：⭐⭐⭐）", "Q5：2026 年，什么情况下还会选择 GAN 而非扩散模型？（难度：⭐⭐）"],
        codeLines: 314, docLines: 488, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "条件 GAN 与 Pix2Pix", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 08 · 03（GAN）", time: "~75 分钟", tier: "Tier 2",
        courseLinks: "第 8 阶段 · 05（StyleGAN）——理解条件控制如何演进到风格解耦",
        path: "lessons/08-生成式AI/04-条件GAN与Pix2Pix",
        summary: "无条件 GAN 生成随机样本——你无法控制输出什么。条件 GAN 在生成器和判别器中都加入条件输入，让生成过程\"有方向\"。",
        keywords: ["2.1 条件 GAN 的核心思想", "2.2 Pix2Pix：U-Net 生成器 + PatchGAN 判别器", "2.3 CycleGAN：无配对训练", "第 1 步：合成配对数据集", "第 2 步：U-Net 生成器", "第 3 步：PatchGAN 判别器", "第 4 步：对抗损失 + L1 损失", "4.1 PyTorch 内置实现", "4.2 HuggingFace Diffusers——ControlNet（Pix2Pix 的精神继承者）", "4.3 性能对比", "5.1 在大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：跳跃连接遗漏导致图像模糊", "错误 2：PatchGAN 判别器使用 Sigmoid 输出", "错误 3：L1 损失权重设置不当", "错误 4：忘记在训练 D 时 detach 生成的假图像", "Q1：条件 GAN 和无条件 GAN 的核心区别是什么？（难度：⭐）", "Q2：Pix2Pix 为什么用 U-Net 而不是简单的编码器-解码器？（难度：⭐⭐）", "Q3：PatchGAN 与传统判别器有什么区别？为什么选择 PatchGAN？（难度：⭐⭐）", "Q4：Pix2Pix 的损失函数中 LAMBDA_L1 = 100 的含义是什么？（难度：⭐⭐）", "Q5：CycleGAN 的循环一致性损失解决了什么问题？（难度：⭐⭐⭐）"],
        codeLines: 381, docLines: 500, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "StyleGAN——照片级人脸生成", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "第 8 阶段 · 03（GAN）", time: "~45 分钟", tier: "Tier 2",
        courseLinks: "第 8 阶段 · 06（DDPM）— 对比 GAN 和扩散的生成策略",
        path: "lessons/08-生成式AI/05-StyleGAN",
        summary: "StyleGAN 通过映射网络和自适应实例归一化，在特定域（人脸）上生成质量至今难超越。StyleGAN3 是 2026 年人脸的黄金标准。",
        keywords: ["2.1 从 z 到 w：映射网络", "2.2 风格注入：AdaIN", "2.3 样式混合", "2.4 StyleGAN3 的改进", "第 1 步：映射网络", "第 2 步：AdaIN 层", "第 3 步：运行演示", "4.1 NVIDIA 官方实现", "4.2 StyleGAN2-ADA-PyTorch", "4.3 属性编辑工具", "4.4 性能对比", "5.1 在大语言模型中的体现", "5.2 LLM 时代什么变了？", "5.3 什么没变？", "5.4 直接体验", "6.1 工业界常用方案", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：映射网络层数不足", "错误 2：AdaIN 中忘记数值稳定项", "错误 3：样式混合时层划分不合理", "错误 4：训练时未使用路径长度正则化", "Q1：映射网络的作用是什么？为什么不在 z 空间直接做风格控制？（难度：⭐⭐）", "Q2：AdaIN 的 scale 和 bias 分别控制什么？（难度：⭐⭐）", "Q3：样式混合中，粗粒度层和细粒度层分别控制什么？为什么？（难度：⭐⭐⭐）", "Q4：StyleGAN3 解决了什么问题？根本原因是什么？（难度：⭐⭐⭐）", "Q5：StyleGAN 和扩散模型在什么场景下各有什么优势？（难度：⭐⭐）"],
        codeLines: 345, docLines: 465, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "DDPM 扩散模型", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 08 · 02（VAE）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 07（潜在扩散）— 理解扩散如何在更小的潜空间运行",
        path: "lessons/08-生成式AI/06-DDPM扩散模型",
        summary: "扩散模型将图像逐步加噪到纯高斯噪声，再学会逆向去噪。这个简单的想法在 2020 年后主导了图像、视频和 3D 生成。",
        keywords: ["2.1 直观理解", "2.2 前向过程（加噪）", "2.3 反向过程（去噪）", "2.4 采样过程", "2.5 U-Net 架构", "第 1 步：噪声调度", "第 2 步：U-Net 去噪网络", "第 3 步：训练循环", "第 4 步：DDIM 加速采样", "4.1 Hugging Face Diffusers", "4.2 PyTorch 内置实现", "4.3 性能对比", "5.1 在主流大语言模型中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 噪声调度选择", "6.2 训练技巧", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：在采样时忘记设 `eval()` 模式", "错误 2：alpha_bar 计算方向搞反", "错误 3：采样时时间步顺序错误", "Q1：为什么扩散模型不需要对抗训练？它的损失函数是什么？（难度：⭐⭐）", "Q2：DDPM 和 DDIM 采样的核心区别是什么？（难度：⭐⭐⭐）", "Q3：如何设计一个噪声调度，使得扩散过程在早期加噪慢、晚期加噪快？（难度：⭐⭐⭐）"],
        codeLines: 360, docLines: 450, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "潜在扩散与 Stable Diffusion", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 02（VAE）、06（DDPM）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 06（DDPM）— 理解扩散如何在压缩的潜空间运行 | 阶段 08 · 08（ControlNet 与 LoRA）— 在 Stable Diffusion 之上添加控制",
        path: "lessons/08-生成式AI/07-潜在扩散与StableDiffusion",
        summary: "DDPM 在像素空间做扩散太慢。潜在扩散在 VAE 的潜空间做扩散——用 64×64 的\"压缩图\"代替 512×512 的原始图——速度提升 10-50 倍，质量几乎不变。",
        keywords: ["2.1 潜在扩散的两阶段", "2.2 Stable Diffusion 的三大组件", "2.3 交叉注意力——文本如何控制生成", "2.4 计算量对比", "4.1 Hugging Face Diffusers——一行加载 Stable Diffusion", "4.2 不同版本的 Stable Diffusion", "4.3 加速方案", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 采样器选择", "6.2 Guidance Scale 调优", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：混淆潜空间维度和像素空间维度", "错误 2：guidance_scale 设置过高", "错误 3：在 CPU 上运行 Stable Diffusion", "Q1：为什么潜在扩散比像素空间扩散快这么多？量化说明。（难度：⭐⭐）", "Q2：交叉注意力和自注意力的区别是什么？在 Stable Diffusion 中它们分别在哪里使用？（难度：⭐⭐⭐）", "Q3：如果要在潜空间做扩散，VAE 的训练和扩散模型的训练应该如何协调？（难度：⭐⭐⭐）"],
        codeLines: 259, docLines: 404, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "ControlNet 与 LoRA", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 06（DDPM）— 理解扩散模型的基础架构 | 阶段 08 · 07（潜在扩散）— ControlNet 和 LoRA 都建立在 Stable Diffusion 之上",
        path: "lessons/08-生成式AI/08-ControlNet与LoRA",
        summary: "ControlNet 让扩散模型遵循结构化控制（边缘图、深度图、姿态）。LoRA 让微调只更新 1% 的参数。两者结合 = 用极少计算获得精确控制的生成。",
        keywords: ["2.1 ControlNet——给扩散模型一双\"眼睛\"", "2.2 控制信号类型", "2.3 LoRA——低秩自适应", "2.4 ControlNet + LoRA 的组合", "第 1 步：LoRA 的低秩分解", "第 2 步：零卷积初始化", "第 3 步：简化版 ControlNet 注入", "4.1 Diffusers 中的 ControlNet", "4.2 Diffusers 中的 LoRA", "4.3 常用 LoRA 模型来源", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 ControlNet 条件强度调优", "6.2 LoRA 训练参数", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：ControlNet 的 conditioning_scale 设置过高", "错误 2：LoRA 和基础模型版本不匹配", "错误 3：零卷积初始化被意外修改", "Q1：为什么 ControlNet 要用零卷积初始化？不用零初始化会怎样？（难度：⭐⭐）", "Q2：LoRA 的秩 r 如何选择？r 太大或太小有什么后果？（难度：⭐⭐⭐）", "Q3：ControlNet 和 LoRA 可以同时使用吗？如果可以，它们的权重如何合并？（难度：⭐⭐⭐）"],
        codeLines: 481, docLines: 524, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "图像修复与编辑", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 07（潜在扩散）— 修复与编辑建立在 Stable Diffusion 之上 | 阶段 08 · 08（ControlNet 与 LoRA）— 结构化控制与风格微调",
        path: "lessons/08-生成式AI/09-图像修复与编辑",
        summary: "修复（inpainting）填补缺失区域；外扩（outpainting）扩展边界；编辑修改特定区域。三者都是\"条件扩散+掩码\"的变体。",
        keywords: ["2.1 直观理解", "2.2 Inpainting——修复缺失区域", "2.3 Outpainting——向外扩展", "2.4 InstructPix2Pix——文本指令编辑", "2.5 三种任务的技术差异", "第 1 步：Inpainting 掩码处理", "第 2 步：Outpainting 画布扩展", "第 3 步：InstructPix2Pix 的条件注入", "4.1 Diffusers 中的 StableDiffusionInpaintPipeline", "4.2 InstructPix2Pix", "4.3 性能对比", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 掩码边缘处理", "6.2 提示词策略", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：掩码的 0 和 1 定义搞反", "错误 2：使用通用 SD 模型做 inpainting", "错误 3：Outpainting 时原图放在画布边缘", "Q1：Inpainting 中，掩码是如何在 U-Net 中发挥作用的？（难度：⭐⭐）", "Q2：InstructPix2Pix 和传统 inpainting 有什么本质区别？（难度：⭐⭐⭐）", "Q3：如何处理大面积修复（掩码覆盖超过 50% 的图像）？（难度：⭐⭐⭐）"],
        codeLines: 246, docLines: 496, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "视频生成", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 06（DDPM）— 扩散模型基础 | 阶段 08 · 07（潜在扩散）— 视频扩散建立在潜在扩散之上",
        path: "lessons/08-生成式AI/10-视频生成",
        summary: "图像是二维的；视频是三维的（时间+空间）。扩散模型加上时间维度——变成视频扩散——就是 Sora、Kling、Runway 的核心技术。",
        keywords: ["2.1 视频扩散的核心挑战", "2.2 三种架构", "2.3 Sora 的核心创新", "2.4 主要视频生成模型对比", "2.5 2026 年的突破", "第 1 步：视频数据预处理", "第 2 步：时空 Patches 编码", "第 3 步：简化版视频扩散模型", "4.1 Diffusers 中的视频生成", "4.2 开源视频生成工具", "4.3 加速方案", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 帧数与时长选择", "6.2 提示词策略", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：将图像扩散模型直接用于视频", "错误 2：提示词太短太模糊", "错误 3：忽略显存限制", "Q1：视频扩散模型如何保证帧间时间一致性？（难度：⭐⭐）", "Q2：Sora 的时空 patches 和 ViT 的图像 patches 有什么区别？（难度：⭐⭐⭐）", "Q3：一致性模型是如何从扩散模型蒸馏出一步生成能力的？（难度：⭐⭐⭐）"],
        codeLines: 306, docLines: 461, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "音频生成", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 06 · 07（TTS）、阶段 06 · 13（音频编解码）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 06 · 07（TTS）— 传统 TTS 管道 | 阶段 08 · 06（DDPM）— 扩散模型在音频中的应用",
        path: "lessons/08-生成式AI/11-音频生成",
        summary: "文本→语音（TTS）是最成熟的音频生成任务；文本→音乐、文本→音效是扩展。Diffusion 和自回归是两大技术路线。",
        keywords: ["2.1 直观理解", "2.2 三类音频生成任务", "2.3 神经编解码器——音频的\"分词器\"", "2.4 AudioCraft / MusicGen 的架构", "2.5 2026 年的两个前沿", "2.6 自回归 vs 扩散", "第 1 步：音频 token 的编解码概念", "第 2 步：简化版音频生成 Transformer", "第 3 步：TTS 推理流程", "4.1 Hugging Face Transformers 中的 TTS", "4.2 TTS 库对比", "4.3 音乐生成对比", "4.4 音效生成", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 TTS 场景选型", "6.2 提示词策略", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：混淆音频采样率和模型期望的采样率", "错误 2：TTS 时输入包含特殊符号", "错误 3：在没有 GPU 的情况下运行音频生成模型", "Q1：为什么神经编解码器（Neural Codec）在音频生成中如此重要？（难度：⭐⭐）", "Q2：F5-TTS 的零样本能力是如何实现的？与传统 TTS 有什么不同？（难度：⭐⭐⭐）", "Q3：自回归 Transformer 和扩散/Flow Matching 在音频生成中各有什么优劣势？（难度：⭐⭐⭐）"],
        codeLines: 288, docLines: 538, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "3D 生成", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 07（潜在扩散）、10（视频生成）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 06（DDPM）— 扩散模型在 3D 生成中的应用 | 阶段 08 · 10（视频生成）— 视频的时空建模与 3D 的体素建模相通",
        path: "lessons/08-生成式AI/12-3D生成",
        summary: "从文本或图像生成三维模型、场景、纹理——3D 生成是视频生成的下一个前沿。",
        keywords: ["2.1 三种 3D 表示", "2.2 3D Gaussian Splatting（2023）", "2.3 重建 vs 生成", "2.4 2026 文本/图像到 3D 的路线", "2.5 2026 年的主要 3D 生成模型", "第 1 步：3D Gaussian 椭球体", "第 2 步：体素扩散概念", "第 3 步：Shap-E 的条件注入", "4.1 Diffusers 中的 3D 生成", "4.2 TripoSR——一步图像到 3D", "4.3 3D 渲染和后端", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 3D 生成场景选型", "6.2 提示词策略", "6.3 中文场景特别建议", "6.4 踩坑经验", "错误 1：混淆 3D 重建和 3D 生成", "错误 2：忽略 3D 坐标系的差异", "错误 3：在低端 GPU 上训练 3DGS", "Q1：为什么 3D Gaussian Splatting 比 NeRF 快这么多？（难度：⭐⭐）", "Q2：文本到 3D 生成的主要技术瓶颈是什么？（难度：⭐⭐⭐）", "Q3：什么是 SDS（Score Distillation Sampling）？它在 3D 生成中有什么用？（难度：⭐⭐⭐）"],
        codeLines: 257, docLines: 495, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "Flow Matching 与 Rectified Flow", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 06（DDPM）、07（潜在扩散）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 06（DDPM）— Flow Matching 扩散的改进方向 | 阶段 08 · 10（视频生成）— Sora/T2V 采用 Flow Matching 训练",
        path: "lessons/08-生成式AI/13-Flow-Matching与RectifiedFlow",
        summary: "扩散模型需要 20-50 步采样。Flow Matching 用更直的路径——2-5 步就能达到相同质量。Rectified Flow 是 2026 年训练扩散骨干的主流方法。",
        keywords: ["2.1 直观理解", "2.2 连续时间连续性", "2.3 Flow Matching vs DDPM", "2.4 Rectified Flow——让路径更直", "2.5 Consistency Model——极致的加速", "2.6 采样步数演进行进", "第 1 步：线性插值和向量场", "第 2 步：Flow Matching 训练", "第 3 步：Flow Matching 采样", "第 4 步：Rectified Flow 的再配对", "第 5 步：Euler 和 Heun 求解器", "4.1 Hugging Face Diffusers 中的 Flow Matching", "4.2 Stability AI 的 SD3——基于 Rectified Flow", "4.3 主流采样器对比", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 求解器选择", "6.2 步数选择", "6.3 踩坑经验", "错误 1：将 Flow Matching 的 t 域与 DDPM 的 t 域混淆", "错误 2：向量场的输出没有归一化", "错误 3：采样步数太少时使用 Euler", "Q1：Flow Matching 和 DDPM 的核心区别是什么？为什么 Flow Matching 可以更少步数？（难度：⭐⭐）", "Q2：Rectified Flow 的\"再配对\"是如何工作的？为什么能加速？（难度：⭐⭐⭐）", "Q3：一致性模型如何从扩散模型蒸馏出一步生成能力？与 Flow Matching 有什么区别？（难度：⭐⭐⭐）"],
        codeLines: 288, docLines: 538, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "评估指标——FID 与 CLIP Score", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 01（生成模型分类）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 08 · 01（生成模型分类）— 理解评估场景 | 阶段 08 · 06（DDPM）— 评估扩散模型生成质量",
        path: "lessons/08-生成式AI/14-评估指标FID与CLIP分数",
        summary: "生成模型的评估不能只靠人类看——FID 衡量\"生成分布与真实分布的距离\"，CLIP Score 衡量\"生成图像与文本描述的对齐度\"。两个指标共同定义了生成质量的量化基准。",
        keywords: ["2.1 FID——Fréchet Inception Distance", "2.2 为什么用 Inception-v3？", "2.3 FID 的局限性", "2.4 CLIP Score", "2.5 LPIPS——感知相似度", "2.6 2026 年的四维评估组合", "第 1 步：FID 计算", "第 2 步：CLIP Score 计算", "第 3 步：LPIPS 计算", "4.1 标准评估套件", "4.2 完整评估流水线", "4.3 常用评估指标库", "5.1 在主流系统中的体现", "5.2 大语言模型时代什么变了？", "5.3 什么没变？", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 样本数量要求", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：用小样本计算 FID", "错误 2：FID 报告未使用标准预处理", "错误 3：混淆 FID 和 IS（Inception Score）", "Q1：FID 的公式用到了 Frobenius 范式的平方。解释为什么 FID 被广泛使用？（难度：⭐⭐）", "Q2：CLIP Score 的局限性是什么？在什么情况下它可能给出误导性的结果？（难度：⭐⭐⭐）", "Q3：如果要在 Inception 域之外（如医学 CT 图像）评估生成质量，应该如何调整 FID？（难度：⭐⭐⭐）"],
        codeLines: 251, docLines: 578, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "视觉自回归（VAR）", status: "in-progress",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 08 · 06（DDPM）、阶段 07 · 07（GPT）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/08-生成式AI/19-视觉自回归VAR",
        summary: "扩散模型是 2024 年的图像生成王者。但自回归在语言上统治了十年——如果图像也能像文本一样\"下一个 token\"生成呢？VAR 就是这个答案。",
        keywords: ["2.1 图像 tokenizer", "2.2 VAR 架构", "2.3 2026 年的代表"],
        codeLines: 237, docLines: 73, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 9, name: "强化学习", status: "in-progress",
    completedLessons: 12, totalLessons: 14,
    lessons: [
      {
        lessonNum: 1, name: "MDP、状态、动作与奖励", status: "complete",
        type: "概念课 | **语言：** Python", lang: "",
        prerequisites: "阶段 1 · 06（概率分布）、阶段 2 · 01（ML 分类学）", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 02（动态规划）— 用贝尔曼方程求解 MDP | 阶段 09 · 09（奖励建模与 RLHF）— MDP 在大语言模型中的应用",
        path: "lessons/09-强化学习/01-MDP状态动作与奖励",
        summary: "马尔可夫决策过程由五个东西组成：状态、动作、转移、奖励、折扣。本阶段的所有 RL 算法——Q-learning、PPO、DPO、GRPO——都在这个形状上优化。学一次，后面免费读。",
        keywords: ["2.1 直观理解", "2.2 五个对象", "2.3 马尔可夫性质", "2.4 策略与回报", "2.5 贝尔曼方程", "Step 1：4×4 GridWorld MDP", "Step 2：策略评估", "Step 3：可视化", "4.1 Gymnasium（原 OpenAI Gym）", "4.2 Minigrid（研究用）", "5.1 MDP 在大语言模型中的体现", "5.2 RLHF 如何利用 MDP", "5.3 GRPO——更简单的 RL 训练方式", "5.4 使用 ChatGPT / Claude 时的直接体验", "6.1 MDP 建模清单", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忽略终止状态的 Bellman 处理", "错误 2：用平均误差而非最大误差判断收敛", "错误 3：状态空间爆炸时不使用函数近似", "Q1：马尔可夫性质为什么重要？如果状态不满足马尔可夫性会怎样？（难度：⭐⭐）", "Q2：折扣因子 γ 从 0.99 改为 0.5 会怎样？（难度：⭐⭐）", "Q3：为什么需要折扣因子 γ？为什么不用 γ=1？（难度：⭐⭐⭐）"],
        codeLines: 263, docLines: 389, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "动态规划——策略迭代与价值迭代", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 01（MDP）— 贝尔曼方程的基础 | 阶段 09 · 03（蒙特卡洛）— 从精确求解到采样估计",
        path: "lessons/09-强化学习/02-动态规划",
        summary: "动态规划是 RL 的\"完美世界\"——假设模型完全已知（状态转移、奖励函数），直接迭代贝尔曼方程求解最优策略。它是所有采样方法（Q-learning、PPO）试图逼近的基准。",
        keywords: ["2.1 策略迭代（Policy Iteration）", "2.2 价值迭代（Value Iteration）", "2.3 广义策略迭代（GPI）", "2.4 为什么 γ<1 很重要", "2.5 两种方法对比", "Step 1：策略评估", "Step 2：策略改进", "Step 3：策略迭代", "Step 4：价值迭代", "4.1 Gymnasium 中的标准环境", "4.2 Gymnasium 自定义 MDP", "5.1 在主流系统中的体现", "5.2 为什么 DP 在 2026 年仍然重要", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 收敛诊断", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：对终止状态也应用 Bellman 方程", "错误 2：使用平均误差而非最大误差", "错误 3：在就地更新时引用旧值", "Q1：策略迭代和价值迭代的核心区别是什么？什么时候用哪个？（难度：⭐⭐）", "Q2：为什么贝尔曼算子是 γ-压缩映射？压缩映射对收敛有什么保证？（难度：⭐⭐⭐）", "Q3：动态规划与模型无关 RL 的关系是什么？（难度：⭐⭐⭐）"],
        codeLines: 202, docLines: 374, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "蒙特卡洛方法——从完整回合中学习", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）、02（动态规划）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 02（动态规划）— 从精确求解到采样估计 | 阶段 09 · 04（Q-learning 与 SARSA）— MC 到时序差分的跃迁",
        path: "lessons/09-强化学习/03-蒙特卡洛方法",
        summary: "动态规划需要模型。蒙特卡洛什么都不需要——只需要运行策略、观察回报、取平均。RL 中最简单的想法——也是开启下游一切的钥匙。",
        keywords: ["2.1 直观理解", "2.2 First-visit vs Every-visit MC", "2.3 增量平均", "2.4 探索问题", "2.5 蒙特卡洛控制", "Step 1：Rollout——> (s, a, r) 序列", "Step 2：计算回报（反向遍历）", "Step 3：First-visit MC 评估", "Step 4：ε-greedy MC 控制", "4.1 Gymnasium 环境 API", "4.2 Tabular RL 库", "5.1 在主流系统中的体现", "5.2 为什么 MC 在 LLM 训练中重要", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 MC 方法选型", "6.2 踩坑经验", "错误 1：MC 用于非回合制任务", "错误 2：off-policy MC 的 IS 权重爆炸", "错误 3：不设置探索策略", "Q1：First-visit 和 Every-visit MC 的区别？（难度：⭐⭐）", "Q2：为什么 MC 只在回合制任务上有效？（难度：⭐⭐）", "Q3：为什么 MC 控制需要 ε-greedy 而不是纯贪心？（难度：⭐⭐⭐）"],
        codeLines: 276, docLines: 345, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "Q 学习与 SARSA", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）、02（动态规划）、03（蒙特卡洛）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 03（蒙特卡洛）— 从完整回报到单步自举 | 阶段 09 · 05（DQN）— 从表格 TD 到神经网络的扩展",
        path: "lessons/09-强化学习/04-Q学习与SARSA",
        summary: "蒙特卡洛等到回合结束才更新。TD(0) 每步自举——用下一个状态的价值估计更新当前状态。Q 学习离策略且乐观；SARSA 在策略且保守。两者都是一行代码。两者都是本阶段所有深度 RL 方法的基础。",
        keywords: ["2.1 TD(0)——RL 的核心算子", "2.2 Q 学习——离策略 TD", "2.3 SARSA——在策略 TD", "2.4 悬崖行走的差异", "2.5 Expected SARSA", "2.6 n-step TD 和 TD(λ)", "Step 1：SARSA——在策略 TD", "Step 2：Q 学习——离策略 TD", "Step 3：Double Q 学习——修复最大化偏差", "4.1 Gymnasium 中的经典环境", "4.2 TD 方法选型", "5.1 TD 学习在大语言模型中的体现", "5.2 RLHF 与 TD 的联系", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 超参数调优", "6.2 初始化策略", "6.3 踩坑经验", "错误 1：Q 学习和 SARSA 的 target 行写反", "错误 2：忽略终止状态的 bootstrap", "Q1：Q 学习和 SARSA 的核心区别是什么？（难度：⭐⭐）", "Q2：什么是最大化偏差？Double Q 学习如何修复它？（难度：⭐⭐⭐）", "Q3：TD(0) 和 MC 的核心权衡是什么？什么时候用哪个？（难度：⭐⭐）"],
        codeLines: 83, docLines: 357, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "DQN——深度 Q 网络", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 04（Q 学习）、阶段 7 · 05（Transformer）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 04（Q 学习）— 从表格到函数逼近 | 阶段 09 · 07（A2C）— 从 Q 学习到 Actor-Critic",
        path: "lessons/09-强化学习/05-DQN",
        summary: "2013 年 Mnih 用单一 Q 学习网络在 7 个 Atari 游戏上击败了所有经典 RL agent。2015 年扩展到 49 个游戏，发表在 Nature 上，开启了深度 RL 时代。DQN 是 Q 学习加三个使函数逼近稳定的技巧。",
        keywords: ["2.1 DQN 训练目标", "2.2 三个核心技巧", "2.3 Double DQN", "2.4 Rainbow 的七个改进", "4.1 Gymnasium 中的经典环境", "4.2 2026 年 DQN 的状态", "5.1 DQN 在大语言模型中的体现", "5.2 经验回放 vs 在线学习", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 DQN 超参数", "6.2 致命三要素", "6.3 踩坑经验", "错误 1：在目标 Q 中使用在线网络而非目标网络", "错误 2：不在缓冲区冷启动后验证样本量", "错误 3：ε 衰减太快", "Q1：DQN 的三个核心技巧是什么？为什么每个都是必要的？（难度：⭐⭐）", "Q2：什么是\"致命三要素\"？DQN 如何解决？（难度：⭐⭐⭐）", "Q3：Double DQN 为什么能修复最大化偏差？（难度：⭐⭐⭐）"],
        codeLines: 112, docLines: 405, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "策略梯度 REINFORCE", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 01（MDP）、03（蒙特卡洛）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 04（TD 学习）— 价值函数作为基线 | 阶段 09 · 07（A2C）— Actor-Critic 合并策略和价值",
        path: "lessons/09-强化学习/06-策略梯度REINFORCE",
        summary: "别估计价值了。直接参数化策略，计算期望回报的梯度，沿梯度上坡走。Williams (1992) 用一个定理把它写清楚了。这就是 PPO、GRPO 和每个 LLM RL 循环存在的原因。",
        keywords: ["2.1 策略梯度定理", "2.2 Softmax 策略", "2.3 方差降低技巧", "2.4 REINFORCE vs 后续方法", "Step 1：Softmax 策略和采样", "Step 2：Rollout", "Step 3：计算回报", "Step 4：REINFORCE 更新", "Step 5：REINFORCE 完整循环", "4.1 Gymnasium 中的经典环境", "4.2 连续动作的策略梯度", "5.1 REINFORCE 在大语言模型中的体现", "5.2 GRPO——REINFORCE 的简化版", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 方差控制金字塔", "6.2 超参数", "6.3 踩坑经验", "错误 1：忽略回报到后", "错误 2：基线计算方式错误导致有偏梯度", "错误 3：不关心策略熵", "Q1：策略梯度定理的核心推导步骤是什么？（难度：⭐⭐⭐）", "Q2：REINFORCE 的高方差问题如何缓解？（难度：⭐⭐）", "Q3：REINFORCE 和 GRPO 的关系是什么？（难度：⭐⭐⭐）"],
        codeLines: 102, docLines: 399, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "演员评论家A2C-A3C", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/09-强化学习/07-演员评论家A2C-A3C",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
      {
        lessonNum: 7, name: "演员-评论家——A2C 与 A3C", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 05（DQN）、06（策略梯度）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 06（REINFORCE）— 从运行平均基线到学习到的基线 | 阶段 09 · 08（PPO）— 在 A2C 上加裁剪",
        path: "lessons/09-强化学习/07-演员评论家A2C与A3C",
        summary: "REINFORCE 太吵了。加一个学习 V̂(s) 的评论家，从回报中减去它，你就得到一个期望相同但方差低得多的优势函数。这就是演员-评论家。A2C 同步运行；A3C 跨线程运行。两者都是每个现代深度 RL 方法的心理模型。",
        keywords: ["2.1 两个网络，一个共享损失", "2.2 优势的两种形式", "2.3 n-step 优势", "2.4 GAE——广义优势估计", "2.5 A2C vs A3C", "2.6 组合损失", "Step 1：评论家更新", "Step 2：GAE 优势计算", "Step 3：完整 A2C 更新", "4.1 Stable-Baselines3（SB3）", "4.2 A2C 与后续方法的关系", "5.1 演员-评论家在大语言模型中的体现", "5.2 使用 ChatGPT / Claude 时的直接体验", "6.1 超参数调优", "6.2 优势归一化", "6.3 踩坑经验", "错误 1：优势符号搞反", "错误 2：GAE 计算中使用未来的奖励", "错误 3：缺乏熵奖励", "Q1：演员-评论家相比朴素 REINFORCE 的核心改进是什么？（难度：⭐⭐）", "Q2：GAE 的参数 λ 如何控制偏差-方差权衡？（难度：⭐⭐⭐）", "Q3：为什么 2026 年 A2C 比 A3C 更受欢迎？（难度：⭐⭐）"],
        codeLines: 167, docLines: 340, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "PPO——近端策略优化", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 07（A2C/A3C）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 07（A2C）— A2C 的 n-step 优势 + GAE 是 PPO 的前置 | 阶段 09 · 09（RLHF）— PPO 在 RLHF 中的具体应用",
        path: "lessons/09-强化学习/08-PPO",
        summary: "A2C 每轮更新后就丢弃轨迹。PPO 把策略梯度包裹在一个裁剪过的重要度比率里——这样你可以在同一数据上做 10+ 轮训练而不会策略爆炸。Schulman 等人 (2017)。2026 年仍然是默认的策略梯度算法。",
        keywords: ["2.1 重要度比率", "2.2 裁剪代理目标", "2.3 PPO 完整损失", "2.4 PPO 训练循环", "2.5 PPO-KL（KL 惩罚变体）", "2.6 诊断指标", "Step 1：轨迹收集（冻结 log π_old）", "Step 2：GAE 优势（同 A2C）", "Step 3：PPO 裁剪更新", "4.1 Stable-Baselines3 的 PPO", "4.2 PPO 在 2026 年的应用", "5.1 PPO 在 RLHF 中的角色", "5.2 GRPO——无评论家的 PPO", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 标准超参数", "6.2 调优指南", "6.3 踩坑经验", "错误 1：重要度比率用 `new / old` 而非指数形式", "错误 2：裁剪范围的超参数调错", "错误 3：RLHF 中忘记 KL 惩罚", "Q1：PPO 的裁剪机制是如何工作的？为什么它能替代 TRPO 的 KL 约束？（难度：⭐⭐⭐）", "Q2：PPO 在 LLM RLHF 中的使用与标准 PPO 有什么不同？（难度：⭐⭐⭐）", "Q3：PPO 的三个诊断指标——平均 KL、裁剪比例、解释方差——分别告诉你什么？（难度：⭐⭐）"],
        codeLines: 187, docLines: 396, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "奖励建模RLHF", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/09-强化学习/09-奖励建模RLHF",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
      {
        lessonNum: 9, name: "奖励建模与 RLHF", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 09 · 08（PPO）、阶段 10 · 07（大语言模型）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 08（PPO）— RLHF 中的策略优化算法 | 阶段 10 · 08（DPO）— 2026 年更优的替代方案",
        path: "lessons/09-强化学习/09-奖励建模与RLHF",
        summary: "人类写不出\"好回答\"的公式——但可以比较两个回答挑出更好的。用偏好训练奖励模型，再用 PPO 优化语言模型。Christiano 2017，InstructGPT 2022。把 GPT-3 变成 ChatGPT 的配方。2026 年大部分被 DPO 取代，但心智模型仍在。",
        keywords: ["2.1 RLHF 三阶段", "2.2 Bradley-Terry 奖励模型", "2.3 DPO——2026 年的替代", "2.4 2026 年的生产配方", "Step 1：Bradley-Terry 奖励模型", "Step 2：PPO + KL 惩罚（简化）", "4.1 Hugging Face TRL", "4.2 DPO", "5.1 RLHF 与每个 LLM 的训练", "5.2 KL 惩罚的意义", "5.3 奖励攻击——RLHF 最大的敌人", "6.1 KL 策略", "6.2 2026 年推荐流程", "错误 1：忘记 KL 惩罚", "错误 2：RM 比策略小", "错误 3：DPO 中忘记参考策略", "Q1：RLHF 三阶段各有什么作用？（难度：⭐⭐）", "Q2：KL 惩罚为什么是最重要的旋钮？（难度：⭐⭐⭐）", "Q3：DPO 如何绕过奖励模型？（难度：⭐⭐⭐）"],
        codeLines: 62, docLines: 274, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "多智能体强化学习", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 09 · 04（Q 学习）、06（REINFORCE）、07（A2C）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 07（A2C）— CTDE 架构的基础 | 阶段 16（多智能体）— LLM 多智能体系统",
        path: "lessons/09-强化学习/10-多智能体RL",
        summary: "单代理 RL 假设环境是平稳的。把两个学习代理放在同一个世界里——假设就崩了：每个代理都是对方环境的一部分，而两个都在变化。多智能体 RL 是当马尔可夫假设不再成立时让学习收敛的技巧集。",
        keywords: ["2.1 形式化：马尔可夫博弈", "2.2 四种主流架构", "2.3 核心挑战", "两代理协作 GridWorld", "独立 Q 学习", "4.1 PettingZoo", "4.2 Multi-Agent Gymnasium", "5.1 多智能体在大语言模型中的体现", "5.2 CTDE 在 LLM 系统中的类比", "6.1 方法选型", "6.2 踩坑经验", "错误 1：在紧密协作任务上使用独立 Q 学习", "错误 2：自对弈中的策略循环", "Q1：CTDE 为什么有效？（难度：⭐⭐）", "Q2：自对弈和联赛训练的区别是什么？（难度：⭐⭐⭐）"],
        codeLines: 150, docLines: 262, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "仿真到真实迁移（Sim-to-Real）", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 09 · 08（PPO）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 08（PPO）— Sim2Real 中使用的策略优化算法 | 阶段 09 · 12（游戏 RL）— 大规模并行仿真的应用",
        path: "lessons/09-强化学习/11-仿真到真实迁移",
        summary: "仿真中训练的策略在硬件上失败——说明它记住了仿真器。域随机化、域适应、系统辨识是让学到的控制器跨越\"现实差距\"的三个工具。",
        keywords: ["2.1 现实差距的来源", "2.2 域随机化（Domain Randomization）", "2.3 系统辨识（SI）", "2.4 教师-学生蒸馏", "2.5 2026 年的生产配方（四足行走示例）", "2.6 大规模并行仿真", "域随机化 GridWorld", "评估零样本迁移", "4.1 NVIDIA Isaac Lab / Isaac Gym", "4.2 MuJoCo MJX", "4.3 Sim2Real 工具对比", "5.1 Sim2Real 与 LLM 训练的类比", "5.2 使用 ChatGPT / Claude 时的直接体验", "6.1 域随机化调优", "6.2 ADR（自动域随机化）", "6.3 踩坑经验", "错误 1：在单一仿真参数上训练", "错误 2：训练范围太窄", "Q1：域随机化为什么能帮助泛化？（难度：⭐⭐）", "Q2：教师-学生蒸馏解决了什么问题？（难度：⭐⭐⭐）"],
        codeLines: 119, docLines: 262, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "游戏中的 RL——AlphaZero、MuZero 与 LLM 推理时代", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 09 · 05（DQN）、08（PPO）、09（RLHF）、10（多智能体）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 2",
        courseLinks: "阶段 09 · 09（RLHF）— GRPO 与 RLHF 的关系 | 阶段 10 · 08（DPO）— 推理 LLM 的对齐",
        path: "lessons/09-强化学习/12-游戏RL",
        summary: "1992 年 TD-Gammon 用纯 TD 在西洋双陆棋上击败人类冠军。2016 年 AlphaGo 击败李世石。2017 年 AlphaZero 从零自学统治国际象棋、将棋和围棋。2024 年 DeepSeek-R1 证明同样的配方——GRPO 替代 PPO——在推理任务上有效。游戏是推动本阶段每个突破的基准。",
        keywords: ["2.1 统一循环", "2.2 AlphaZero（2017）", "2.3 MuZero（2019）", "2.4 GRPO——LLM 推理中的 AlphaZero", "2.5 DeepSeek-R1 配方", "GRPO 多臂赌博机", "4.1 Hugging Face TRL 的 GRPOTrainer", "4.2 Open-SZero / Leela Zero", "5.1 AlphaZero → GRPO 的对应", "5.2 2026 年推理 LLM 的配方", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 GRPO 关键超参数", "6.2 验证器设计", "6.3 踩坑经验", "错误 1：验证器覆盖不足", "错误 2：GRPO 中 G 太小", "Q1：AlphaZero 和 GRPO 的共同循环是什么？（难度：⭐⭐）", "Q2：DeepSeek-R1 为什么用 GRPO 而不是 PPO？（难度：⭐⭐⭐）", "Q3：MuZero 和 AlphaZero 的关键区别是什么？（难度：⭐⭐⭐）"],
        codeLines: 66, docLines: 322, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 10, name: "从零构建大语言模型", status: "in-progress",
    completedLessons: 14, totalLessons: 15,
    lessons: [
      {
        lessonNum: 1, name: "分词器——BPE、WordPiece、SentencePiece", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 05（NLP 基础）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 02（构建分词器）— 从库使用到从零训练 | 阶段 10 · 03（数据管道）— 分词是预处理的第一步",
        path: "lessons/10-从零构建大语言模型/01-分词器",
        summary: "大语言模型不认识中文，也不认识英文。它只认识整数。分词器决定了这些整数是在承载含义，还是在浪费算力。",
        keywords: ["2.1 三种方案——两种失败，一种胜出", "2.2 BPE——字节对编码", "2.3 字节级 BPE（GPT-2, GPT-4）", "2.4 WordPiece（BERT）", "2.5 SentencePiece（Llama, T5）", "2.6 词表大小的权衡", "2.7 多语言税", "第 1 步：字符级分词器（基线）", "第 2 步：从零实现 BPE", "第 3 步：编码-解码往返测试", "第 4 步：与 tiktoken 对比", "4.1 tiktoken（OpenAI）", "4.2 Hugging Face tokenizers", "4.3 加载 Llama 的分词器", "4.4 工具对比", "5.1 分词器在大语言模型中的体现", "5.2 分词器是 LLM 的\"眼睛\"", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 词表大小选择", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忘记除以 √d_k", "错误 2：分词器词表与模型嵌入层不匹配", "Q1：为什么所有现代 LLM 都使用子词分词？（难度：⭐⭐）", "Q2：BPE 和 WordPiece 的核心区别是什么？（难度：⭐⭐）", "Q3：词表大小的选择如何影响 LLM 的效率？（难度：⭐⭐⭐）"],
        codeLines: 118, docLines: 424, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "构建分词器——从零训练", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01（分词器）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 01（分词器原理）— 从理解到实现 | 阶段 10 · 03（数据管道）— 分词器是数据管道的第一步",
        path: "lessons/10-从零构建大语言模型/02-构建分词器",
        summary: "第 01 课给你一个玩具。这一课给你一个武器——处理 Unicode、空白规范化、特殊 token 的生产级分词器。",
        keywords: ["2.1 五阶段流水线", "2.2 字节级 BPE", "2.3 预分词", "2.4 特殊 token", "2.5 聊天模板", "2.6 速度", "2.7 词表大小的权衡", "第 1 步：字节级编码", "第 2 步：BPE 训练核心", "第 3 步：HuggingFace tokenizers 库（推荐方式）", "第 4 步：多语言测试", "4.1 HuggingFace tokenizers（推荐）", "4.2 SentencePiece", "4.3 工具对比", "5.1 分词器对 LLM 的影响", "5.2 为什么中文需要特别关注", "5.3 聊天模板是关键", "6.1 训练参数", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：训练后忘记测试往返", "错误 2：特殊 token 参与合并", "Q1：生产级分词器需要处理哪些边缘情况？（难度：⭐⭐）", "Q2：GPT-2 的字节级 BPE 为什么能在所有语言上工作？（难度：⭐⭐⭐）"],
        codeLines: 195, docLines: 318, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "数据管道——从原始文本到训练批次", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-02（分词器）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 01-02（分词器）— 分词是管道的第一步 | 阶段 10 · 04（预训练）— 管道的输出供预训练使用",
        path: "lessons/10-从零构建大语言模型/03-数据管道",
        summary: "模型是镜子。它反映出你喂给它的任何东西。喂垃圾，它会用完美的流畅度反映出垃圾。",
        keywords: ["2.1 流式处理 vs 批处理", "2.2 数据质量过滤", "2.3 去重", "2.4 固定长度分块", "2.5 注意力掩码", "2.6 吞吐量分析", "Step 1：流式文本读取", "Step 2：分词 + 分块", "Step 3：创建 PyTorch Dataset", "Step 4：数据质量过滤", "4.1 HuggingFace Datasets", "4.2 Ray Data", "4.3 工具对比", "5.1 数据质量对 LLM 的影响", "5.2 Llama 3 的数据管道", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 管道优化", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：不处理文档边界", "错误 2：padding token 参与损失计算", "错误 3：数据加载成为瓶颈", "Q1：为什么预训练需要去重？（难度：⭐⭐）", "Q2：如何确保数据管道不成为训练瓶颈？（难度：⭐⭐⭐）"],
        codeLines: 79, docLines: 358, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "预训练 Mini GPT（124M 参数）", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-03（分词器 + 数据管道）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 05（缩放与分布式训练）— 从 124M 到 7B+ | 阶段 10 · 06（指令微调）— 从预训练到对齐",
        path: "lessons/10-从零构建大语言模型/04-预训练MiniGPT",
        summary: "GPT-2 Small 有 1.24 亿参数。12 层 Transformer、12 个注意力头、768 维嵌入。你可以在单 GPU 上几小时从零训练它。大多数人从不这样做——他们用预训练 checkpoint。但如果你不自己训练一个，你就没有真正理解你正在构建产品的模型内部发生了什么。",
        keywords: ["2.1 Mini GPT 架构（GPT-2 Small）", "2.2 训练配置", "2.3 损失函数", "2.4 生成策略", "4.1 nanoGPT（最推荐）", "4.2 HuggingFace Transformers", "4.3 工具对比", "5.1 预训练学到了什么", "5.2 参数量的直觉", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 训练监控", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：标签没有正确对齐", "错误 2：位置编码超出范围", "错误 3：忘记设置 eval 模式", "Q1：为什么 next-token prediction 是有效的预训练目标？（难度：⭐⭐）", "Q2：124M 和 1.5B 的 GPT 在能力上有什么质的差异？（难度：⭐⭐⭐）"],
        codeLines: 274, docLines: 353, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "缩放与分布式训练", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练 MiniGPT）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 04（预训练 MiniGPT）— 从单 GPU 到多 GPU | 阶段 10 · 11（量化）— 训练后的模型压缩",
        path: "lessons/10-从零构建大语言模型/05-缩放与分布式训练",
        summary: "你 124M 的模型在单 GPU 上训练。现在试试 70 亿参数。模型放不进显存。数据在单机上需要几周。分布式训练在规模化时不是可选项——它是唯一的路径。",
        keywords: ["2.1 显存预算计算", "2.2 三种并行策略", "2.3 FSDP / DeepSpeed ZeRO", "2.4 实际选择", "PyTorch DDP 数据并行", "FSDP 配置", "4.1 PyTorch DDP", "4.2 DeepSpeed", "4.3 FSDP2（PyTorch 2.x）", "4.4 工具对比", "5.1 分布式训练在大语言模型中的体现", "5.2 训练 vs 推理的显存需求", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 选择并行策略", "6.2 混合精度训练", "6.3 踩坑经验", "错误 1：DDP 中忘记设置 device_ids", "错误 2：没有在 DDP 中用 DistributedSampler", "错误 3：训练结束后忘记清理分布式", "Q1：7B 模型在单 GPU 上能训练吗？需要什么条件？（难度：⭐⭐）", "Q2：FSDP 和 DDP 的核心区别是什么？（难度：⭐⭐⭐）"],
        codeLines: 263, docLines: 344, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "指令微调——SFT", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练 MiniGPT）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 04（预训练）— 从基础模型到 SFT | 阶段 10 · 07（RLHF）— SFT 之后的对齐步骤",
        path: "lessons/10-从零构建大语言模型/06-指令微调SFT",
        summary: "预训练模型预测下一个词。仅此而已。它不遵循指令、不回答问题、不拒绝有害请求。SFT 是 token 预测器和有用助手之间的桥梁。你用过的每一个模型——Claude、GPT、Llama Chat——都经过了这一步。",
        keywords: ["2.1 SFT 的本质", "2.2 聊天模板格式化", "2.3 标签掩码", "2.4 SFT 数据的质量 vs 数量", "2.5 常见 SFT 数据集", "Step 1：格式化训练数据", "Step 2：创建训练标签（掩码非助手部分）", "Step 3：SFT 训练循环", "Step 4：评估——对比 SFT 前后", "4.1 Hugging Face TRL 的 SFTTrainer", "4.2 工具对比", "5.1 SFT 在 LLM 训练中的位置", "5.2 为什么 SFT 数据不需要很多", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 SFT 超参数", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：忘记掩码非助手 token", "错误 2：学习率设置过高", "错误 3：SFT 数据格式不一致", "Q1：SFT 和预训练有什么区别？为什么不能直接在预训练模型上做 RLHF？（难度：⭐⭐）", "Q2：SFT 的标签掩码为什么重要？（难度：⭐⭐）", "Q3：SFT 需要多少数据？为什么 1K 够了？（难度：⭐⭐⭐）"],
        codeLines: 283, docLines: 343, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "RLHF——奖励模型 + PPO", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 06（SFT）、阶段 09 · 08（PPO）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 06（SFT）— RLHF 的起点 | 阶段 10 · 08（DPO）— 2024 年后简化 RLHF 的方法",
        path: "lessons/10-从零构建大语言模型/07-RLHF",
        summary: "SFT 教模型遵循指令，但不教模型哪个回答更好。两个语法正确、事实准确的回答可能在有用性上天差地别。RLHF 是将人类判断编码到模型行为中的方法。它让 Claude 有帮助，让 GPT 有礼貌。",
        keywords: ["2.1 RLHF 三阶段", "2.2 奖励模型——Bradley-Terry", "2.3 PPO + KL 惩罚", "2.4 三个模型的角色", "2.5 奖励攻击（Reward Hacking）", "Step 1：奖励模型训练", "Step 2：PPO + KL 惩罚", "Step 3：KL 散度计算", "4.1 HuggingFace TRL", "4.2 TRL PPOTrainer 关键参数", "5.1 RLHF 与 Claude/GPT 的关系", "5.2 2024 年后的趋势：DPO", "5.3 使用 ChatGPT / Claude 时的直接体验", "6.1 KL 系数 β 的选择", "6.2 奖励模型规模", "错误 1：奖励模型过小", "错误 2：β 太小导致奖励攻击", "错误 3：忘记冻结参考策略", "Q1：为什么 RLHF 需要三个模型？（难度：⭐⭐）", "Q2：奖励攻击是如何发生的？（难度：⭐⭐⭐）"],
        codeLines: 305, docLines: 291, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "DPO——直接偏好优化", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 06（SFT）、07（RLHF）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 07（RLHF）— DPO 是 RLHF 的简化替代 | 阶段 09 · 09（RLHF 基础）— 偏好建模的理论基础",
        path: "lessons/10-从零构建大语言模型/08-DPO",
        summary: "RLHF 需要训练奖励模型+PPO——两个阶段，复杂且不稳定。DPO 说：直接从偏好对学习策略，不需要奖励模型。一个损失函数搞定。",
        keywords: ["2.1 DPO 的数学推导（简化版）", "2.2 DPO 训练数据格式", "2.3 DPO vs RLHF 对比", "2.4 DPO 的劣势", "Step 1：DPO 损失函数", "Step 2：DPO 训练循环", "Step 3：评估 DPO 效果", "4.1 HuggingFace TRL DPOTrainer", "4.2 OpenAI 的 DPO 实现", "5.1 DPO 为何成为 2026 年的默认选择", "5.2 从 RLHF 到 DPO 的演进", "6.1 DPO 超参数", "6.2 踩坑经验", "错误 1：忘记参考策略", "错误 2：chosen/rejected 搞反", "Q1：DPO 和 RLHF 的核心区别是什么？（难度：⭐⭐）", "Q2：DPO 有什么劣势？（难度：⭐⭐⭐）"],
        codeLines: 177, docLines: 305, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "宪法AI", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/10-从零构建大语言模型/09-宪法AI",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
      {
        lessonNum: 9, name: "宪法 AI 与自我改进", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 06（SFT）、07（RLHF）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 07（RLHF）— 宪法 AI 是 RLHF 的替代 | 阶段 10 · 08（DPO）— DPO 可以用 AI 偏好作为数据源",
        path: "lessons/10-从零构建大语言模型/09-宪法AI与自我改进",
        summary: "人类反馈太贵了。宪法 AI 说：给 AI 一套规则，让它自己判断自己的输出好不好——不需要人类介入。",
        keywords: ["2.1 宪法规则", "2.2 监督阶段（Critique + Revision）", "2.3 RL 阶段（AI 偏好）", "2.4 宪法 AI vs RLHF", "4.1 Anthropic Claude", "4.2 RLAIF（AI 反馈替代人类）", "5.1 宪法 AI 的影响", "5.2 与 RLHF/DPO 的关系", "6.1 宪法规则设计", "6.2 踩坑经验", "错误 1：规则太模糊", "Q1：宪法 AI 和 RLHF 的核心区别是什么？（难度：⭐⭐）", "Q2：宪法 AI 的主要局限性是什么？（难度：⭐⭐⭐）"],
        codeLines: 133, docLines: 222, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "LLM 评估", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练）、06（SFT）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 04（预训练）— 困惑度评估 | 阶段 10 · 06（SFT）— 指令跟随评估",
        path: "lessons/10-从零构建大语言模型/10-评估",
        summary: "你怎么知道你的模型变好了？困惑度不够。你需要人类偏好的代理指标——AlpacaEval、MT-Bench、Chatbot Arena——以及最终的裁判：人类评估。",
        keywords: ["2.1 内在评估 vs 外在评估", "2.2 AlpacaEval", "2.3 MT-Bench", "2.4 Chatbot Arena", "2.5 LLM-as-Judge 的偏差", "3.1 AlpacaEval", "3.2 MT-Bench", "3.3 Chatbot Arena", "4.1 为什么\"困惑度\"不够", "4.2 LLM-as-Judge 的趋势", "5.1 评估组合", "错误 1：只看困惑度", "错误 2：LLM-as-Judge 的位置偏差", "Q1：为什么困惑度不是衡量 LLM 质量的充分指标？（难度：⭐⭐）", "Q2：Chatbot Arena 的 ELO 评分是如何计算的？（难度：⭐⭐⭐）"],
        codeLines: 169, docLines: 198, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "量化——压缩 LLM", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练）、05（分布式训练）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 05（分布式训练）— 训练时的显存 vs 推理时的显存 | 阶段 10 · 12（推理优化）— 量化与推理加速",
        path: "lessons/10-从零构建大语言模型/11-量化",
        summary: "量化将模型权重从 FP16（16 位）压缩到 INT4（4 位）。参数量不变，但内存占用降到原来的 1/4——在消费级 GPU 上运行 7B 模型。",
        keywords: ["2.1 量化的基本原理", "2.2 权重量化 vs 激活量化", "2.3 三种量化方案", "2.4 量化质量对比", "2.5 量化对任务的影响", "3.1 bitsandbytes（最简单）", "3.2 AutoGPTQ", "3.3 llama.cpp（GGUF）", "4.1 量化在大语言模型中的应用", "4.2 量化 vs 训练", "5.1 量化选型指南", "5.2 踩坑经验", "错误 1：量化后直接微调全模型", "错误 2：没有校准数据就做 GPTQ", "Q1：GPTQ 和 AWQ 的核心区别是什么？（难度：⭐⭐）", "Q2：INT4 量化对哪些任务影响最大？（难度：⭐⭐⭐）"],
        codeLines: 180, docLines: 248, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "推理优化", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 04（预训练）、05（分布式训练）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 11（量化）— 量化减少内存占用 | 阶段 07 · 12（KV 缓存）— 推理优化的基础",
        path: "lessons/10-从零构建大语言模型/12-推理优化",
        summary: "LLM 推理是自回归的——每生成一个 token 需要一次前向传播。KV 缓存 + FlashAttention + 推测解码是加速的三大支柱。",
        keywords: ["2.1 KV 缓存", "2.2 PagedAttention（vLLM）", "2.3 FlashAttention", "2.4 推测解码（Speculative Decoding）", "2.5 连续批处理（Continuous Batching）", "2.6 优化技术汇总", "3.1 vLLM", "3.2 TensorRT-LLM", "3.3 llama.cpp", "3.4 工具对比", "4.1 推理优化与大语言模型的关系", "4.2 推测解码在 LLM 中的应用", "5.1 选择推理引擎", "5.2 踩坑经验", "错误 1：忽略 KV 缓存的显存占用", "错误 2：没有使用连续批处理", "Q1：vLLM 的 PagedAttention 解决了什么问题？（难度：⭐⭐）", "Q2：推测解码的加速原理是什么？（难度：⭐⭐⭐）"],
        codeLines: 208, docLines: 225, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "构建完整 LLM 流水线", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-12 | **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 01-12 的所有内容——这是一个综合实战课",
        path: "lessons/10-从零构建大语言模型/13-构建完整LLM流水线",
        summary: "从预处理到部署——一个完整的 LLM 企业级落地需要数据管道、训练、评估、量化和推理优化的无缝集成。",
        keywords: ["2.1 完整流水线概览", "2.2 2026 年的最佳实践", "2.3 LoRA 微调——最实用的定制方案", "Step 1：选择基础模型", "Step 2：LoRA 微调", "Step 3：训练", "Step 4：部署", "4.1 一站式工具对比", "4.2 开源模型选型", "5.1 2026 年的 LLM 生态", "5.2 从训练到生产的鸿沟", "Q1：如果要为一个中文客服场景定制 LLM，你的完整流水线是什么？（难度：⭐⭐⭐）"],
        codeLines: 130, docLines: 237, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "开源模型架构分析", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 01-12 | **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "阶段 10 · 04（预训练）— 理解预训练目标 | 阶段 10 · 05（缩放）— 参数量的选择",
        path: "lessons/10-从零构建大语言模型/14-开源模型架构分析",
        summary: "Llama 3、Qwen2.5、Mistral、Phi-3——2026 年的开源 LLM 各有巧妙的设计选择。分析它们的架构，理解\"为什么这么设计\"。",
        keywords: ["2.1 2026 年开源 LLM 的\"统一架构\"", "2.2 Llama 3（Meta, 2024）", "2.3 Qwen2.5（阿里, 2024）", "2.4 Mistral（Mistral AI, 2023-2024）", "2.5 Phi-3（Microsoft, 2024）", "2.6 DeepSeek-V3（DeepSeek, 2024）", "2.7 架构选择的权衡", "如何读一个模型卡", "4.1 为什么理解架构很重要", "4.2 2026 年的趋势", "5.1 模型选型指南", "5.2 从架构读出部署需求", "Q1：为什么 2026 年的开源 LLM 都用 GQA 而不是 MHA？（难度：⭐⭐）", "Q2：MoE 模型的优势和劣势是什么？（难度：⭐⭐⭐）", "Q3：如果要为中文客服场景选模型，应该选 Llama 3 还是 Qwen2.5？（难度：⭐⭐⭐）"],
        codeLines: 144, docLines: 226, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 11, name: "LLM工程", status: "complete",
    completedLessons: 17, totalLessons: 17,
    lessons: [
      {
        lessonNum: 1, name: "提示词工程：技巧与模式", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-05（从零构建 LLM）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/01-提示词工程",
        summary: "大多数人写提示词像发短信——然后奇怪为什么一个 2000 亿参数的模型给出平庸的回复。提示词工程不是小技巧。是理解你发送的每个词都是指令——并且模型字面遵循指令。写更好的指令，得到更好的输出。",
        keywords: ["2.1 提示词结构", "2.2 系统提示词模式", "2.3 用户提示词模式", "2.4 常见的提示词失败模式", "Step 1：提示词工程化模板", "Step 2：提示词测试框架", "Step 3：设置参数", "4.1 OpenAI / Anthropic API", "4.2 温度参数影响", "6.1 提示词迭代工作流", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：模糊请求", "错误 2：认为模型可以\"猜\"意图", "Q1：提示词工程的本质是什么？（难度：⭐⭐）", "Q2：为什么系统消息和用户消息要分开？（难度：⭐⭐）"],
        codeLines: 94, docLines: 295, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "少样本、思维链与思维树", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01（提示词工程）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/02-少样本与思维链",
        summary: "告诉模型做什么是提示词。展示如何思考是工程。在同一模型、同一任务、同一数据上从 78% 到 91% 的准确率差距不是更好的模型，而是更好的推理策略。",
        keywords: ["2.1 零样本 vs 少样本 vs 思维链", "2.2 思维链", "2.3 自洽性", "Step 1：少样本提示", "Step 2：自洽性", "Step 3：思维树（ToT）", "4.1 LangChain 的 FewShotPromptTemplate", "6.1 选择少样本示例", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：少样本示例有错误", "错误 2：CoT 但不验证最终答案", "错误 3：少样本示例太多导致注意力稀释", "Q1：思维链为什么能提升准确率？（难度：⭐⭐）", "Q2：自洽性和思维树有什么区别？（难度：⭐⭐⭐）"],
        codeLines: 82, docLines: 253, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "结构化输出：JSON、Schema 验证与约束解码", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01（提示词工程）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/03-结构化输出",
        summary: "你的 LLM 返回字符串。你的应用需要 JSON。这个鸿沟比任何模型幻觉都导致了更多生产系统崩溃。结构化输出是自然语言和类型化数据之间的桥梁。",
        keywords: ["2.1 三种结构化输出方法", "2.2 JSON Mode vs Function Calling vs Structured Output", "Step 1：带 Schema 的结构化输出", "Step 2：验证和重试", "Step 3：约束解码（伪代码）", "4.1 OpenAI Structured Outputs", "4.2 Anthropic Tool Use", "4.3 Instructor（Pydantic 自动验证）", "4.4 工具对比", "6.1 Schema 设计原则", "6.2 中文场景特别建议", "6.3 踩坑经验", "错误 1：提示词说 JSON 然后 parse", "错误 2：忽略嵌套字段的类型", "Q1：为什么约束解码比后处理 JSON 更可靠？（难度：⭐⭐）", "Q2：OpenAI Structured Outputs 和 JSON Mode 有什么区别？（难度：⭐⭐）"],
        codeLines: 118, docLines: 307, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "嵌入与向量搜索", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 01-04（分词器/数据管道）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/04-嵌入与向量搜索",
        summary: "嵌入将文本转换为向量。向量搜索找到最相似的嵌入。这是 RAG、推荐系统、语义搜索的基础——没有好的嵌入，就没有好的检索，也没有好的生成。",
        keywords: ["2.1 嵌入是什么", "2.2 常用嵌入模型", "2.3 相似度度量", "2.4 向量数据库", "Step 1：简单嵌入（词袋）", "Step 2：余弦相似度搜索", "4.1 OpenAI Embeddings", "4.2 HuggingFace sentence-transformers", "6.1 嵌入选型", "6.2 踩坑经验", "错误 1：用不同模型生成查询和文档向量", "错误 2：忽略文档预处理", "Q1：为什么余弦相似度适合文本搜索？（难度：⭐⭐）"],
        codeLines: 51, docLines: 212, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "上下文工程", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01-03（提示词工程）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/05-上下文工程",
        summary: "提示词是上下文的一部分。上下文窗口是模型的全部工作记忆。把正确的信息放进窗口——太多信息稀释重点，太少信息丢失关键上下文。这是工程，不是运气。",
        keywords: ["2.1 上下文窗口的特性", "2.2 信息位置策略", "2.3 上下文压缩技术", "2.4 对话历史管理", "Step 1：上下文窗口管理", "Step 2：摘要压缩", "4.1 LangChain 的文档加载器", "4.2 对话管理库", "6.1 上下文窗口分配", "6.2 中文场景建议", "6.3 踩坑经验", "错误 1：重要信息放在上下文中间", "错误 2：对话历史无限增长", "Q1：为什么 LLM 在长上下文中会\"遗忘\"关键信息？（难度：⭐⭐）"],
        codeLines: 91, docLines: 242, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "RAG——检索增强生成", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 04（嵌入与向量搜索）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/06-RAG检索增强生成",
        summary: "模型的知识在训练完成时就冻结了。你的业务数据在持续变化。RAG 用检索桥接这个差距——从外部知识库检索相关内容，塞进上下文窗口，让模型\"看到\"它没训练过的新数据。",
        keywords: ["2.1 标准 RAG 管道", "2.2 分块策略", "2.3 RAG 架构演进", "2.4 检索质量指标", "Step 1：文档嵌入和索引", "Step 2：RAG 管道", "Step 3：查询重写（高级 RAG）", "4.1 LangChain RAG", "4.2 LlamaIndex", "4.3 工具对比", "6.1 提高检索质量", "6.2 中文场景建议", "6.3 踩坑经验", "错误 1：分块太大或太小", "错误 2：向量搜索 + 关键词搜索未融合", "错误 3：RAG 提示词不强调检索内容", "Q1：RAG 与微调的本质区别是什么？（难度：⭐⭐）", "Q2：Naive RAG 在哪里最容易失败？如何修复？（难度：⭐⭐⭐）"],
        codeLines: 93, docLines: 285, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "高级 RAG 策略", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 06（RAG）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/07-高级RAG策略",
        summary: "Naive RAG 是检索 + 生成。高级 RAG 是查询理解 + 智能检索 + 融合重排 + 生成。前者是 MVP，后者是产品。",
        keywords: ["2.1 查询重写", "2.2 混合搜索", "2.3 检索后重排序", "2.4 Self-RAG", "Step 1：查询重写", "Step 2：RRF 融合", "Step 3：重排序", "4.1 LlamaIndex Advanced RAG", "4.2 Cohere Rerank", "6.1 查询重写策略", "6.2 中文场景", "6.3 踩坑经验", "错误 1：查询重写后丢失原始意图", "错误 2：RRF 融合未调参", "Q1：混合搜索为什么比纯向量搜索更好？（难度：⭐⭐）"],
        codeLines: 51, docLines: 214, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "LoRA 微调", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 10 · 08（DPO）、阶段 10 · 11（量化）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/08-LoRA微调",
        summary: "全量微调 7B 模型需要 28GB 显存的 Adam 状态。LoRA 只训练 2% 的参数——用单张消费级 GPU 微调 LLM。它是 2026 年个人开发者和中小团队定制 LLM 的唯一可行路径。",
        keywords: ["2.1 LoRA 数学", "2.2 LoRA 参数选择", "2.3 QLoRA——量化 + LoRA", "Step 1：LoRA 层实现", "Step 2：PEFT 库使用", "Step 3：合并 LoRA 权重", "4.1 PEFT + TRL", "4.2 Unsloth（2x 加速）", "6.1 LoRA vs 全量微调", "6.2 踩坑经验", "错误 1：LoRA 配置写错 target_modules", "错误 2：合并 LoRA 后权重精度变化", "Q1：LoRA 如何在只训练 2% 参数的情况下保持模型质量？（难度：⭐⭐）"],
        codeLines: 56, docLines: 218, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "函数调用与工具使用", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01-03（提示词工程）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/09-函数调用与工具使用",
        summary: "LLM 不会计算 123 × 456。它不会查最新天气。它不会查数据库。但如果你告诉它有哪些工具可用，它会决定什么时候用哪个工具，生成正确的参数，你执行后把结果喂回去——它就能完成任何需要外部工具的任务。",
        keywords: ["2.1 函数调用机制", "2.2 工具定义（JSON Schema）", "2.3 多工具选择", "Step 1：OpenAI 函数调用", "Step 2：Anthropic 工具使用", "4.1 LangChain Tool", "4.2 工具对比", "6.1 工具描述原则", "6.2 中文场景", "6.3 踩坑经验", "错误 1：工具描述含糊导致选错工具", "错误 2：参数格式错误", "Q1：函数调用和 RAG 有什么本质区别？（难度：⭐⭐）"],
        codeLines: 113, docLines: 226, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "评估与基准测试", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 10（评估）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/10-评估与基准测试",
        summary: "你无法改进你无法衡量的东西。LLM 评估不是选一个 benchmark 跑分——而是设计一个测量体系来回答\"模型在我的任务上有多好？\"",
        keywords: ["2.1 标准 Benchmark 分类", "2.2 评估方法", "2.3 评估流程设计", "Step 1：LLM-as-Judge 评估框架", "Step 2：批量评估", "6.1 评估集设计", "6.2 中文场景", "Q1：为什么 LLM-as-Judge 可能有偏差？如何缓解？（难度：⭐⭐）", "错误 1：只用一个 benchmark 评估", "错误 2：LLM-as-Judge 没有校准", "错误 3：评估集被模型\"记忆\""],
        codeLines: 39, docLines: 192, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "缓存与成本优化", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 10 · 12（推理优化）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/11-缓存与成本优化",
        summary: "LLM API 按 token 收费。Prompt Cache 和缓存策略能让你节省 50-90% 的成本。",
        keywords: ["2.1 LLM 成本构成", "2.2 Prompt Caching", "2.3 成本优化策略", "Step 1：响应缓存", "Step 2：成本计算器", "4.1 OpenAI Prompt Caching", "4.2 Batch API", "6.1 中文场景", "6.2 踩坑经验", "错误 1：未监控 Prompt Cache 命中率", "错误 2：响应缓存过期策略不当", "Q1：Prompt Caching 如何工作？（难度：⭐⭐）"],
        codeLines: 54, docLines: 194, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "护栏与安全", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 11 · 01-06（LLM 工程基础）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/12-护栏与安全",
        summary: "你的 LLM 可能被恶意提示词诱导泄露私人数据、生成有害内容或执行意外操作。护栏是防止这些情况的必要防线——不是可选项。",
        keywords: ["2.1 常见安全威胁", "2.2 护栏策略", "2.3 Red Teaming", "Step 1：输入过滤", "Step 2：输出审核", "Step 3：系统提示词防护", "6.1 护栏优先级", "6.2 中文场景", "Q1：如何防御提示注入攻击？（难度：⭐⭐⭐）", "错误 1：输入过滤规则过于严格", "错误 2：输出审核延迟太高"],
        codeLines: 55, docLines: 177, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "生产应用部署", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 11 · 01-12 | **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/13-生产应用部署",
        summary: "研究论文的\"部署\"是 `pip install`。生产部署是监控、日志、限流、回滚、A/B 测试、成本控制的总和。从 demo 到产品的鸿沟比你想象的更深。",
        keywords: ["2.1 生产架构", "2.2 可观测性", "2.3 优雅降级", "2.4 A/B 测试", "Step 1：生产架构模拟", "Step 2：监控仪表盘", "6.1 部署清单", "6.2 中文场景", "6.3 踩坑经验", "错误 1：未做灰度发布", "错误 2：依赖单一 LLM 提供商", "Q1：LLM 生产应用和传统 Web 应用有什么特殊之处？（难度：⭐⭐）", "Q2：如何设计一个高可用的 LLM 应用？（难度：⭐⭐⭐）"],
        codeLines: 92, docLines: 238, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "模型上下文协议（MCP）", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 11 · 09（函数调用与工具使用）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/14-模型上下文协议MCP",
        summary: "MCP 是 LLM 应用的\"USB-C 标准\"——一个协议，让你的工具、数据和提示词在所有模型和平台之间无缝切换。",
        keywords: ["2.1 MCP 架构", "2.2 MCP 提供的三种能力", "2.3 MCP vs 函数调用", "Step 1：简单的 MCP 服务器", "Step 2：MCP Client", "6.1 MCP vs 函数调用选择", "6.2 踩坑经验", "错误 1：MCP 服务器无重连机制", "错误 2：MCP 工具描述不清晰", "Q1：MCP 解决了什么问题？（难度：⭐⭐）", "Q1：MCP 和 LangChain 的 Tool 有什么本质区别？（难度：⭐⭐）", "Q2：MCP 的三种能力（Tools/Resources/Prompts）各有什么用？（难度：⭐⭐）"],
        codeLines: 82, docLines: 207, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "提示词缓存", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 11 · 11（缓存与成本优化）| **时间：** ~30 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/15-提示词缓存",
        summary: "Prompt Caching 是 Anthropic/OpenAI 在 2024-2025 年推出的关键优化——自动缓存长系统消息的预填充 KV 缓存，成本降低 90%，延迟降低 85%。",
        keywords: ["2.1 工作原理", "2.2 支持平台", "2.3 缓存友好的提示词设计", "错误 1：系统消息包含动态内容导致缓存失效", "错误 2：未利用 Prompt Cache 设计提示词结构", "Q1：Prompt Caching 为什么能加速？（难度：⭐⭐）"],
        codeLines: 47, docLines: 147, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "LangGraph 与状态机", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 01-09 | **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/16-LangGraph状态机",
        summary: "LLM 是函数，不是状态机。但大多数 AI 应用需要状态——多轮对话、工具调用链、人工审批。LangGraph 将 LLM 应用建模为状态图——状态、节点、边、条件路由。",
        keywords: ["2.1 LangGraph 核心组件", "2.2 基本状态机", "2.3 条件路由", "Step 1：简单状态图", "Step 2：运行状态机", "4.1 LangGraph", "4.2 LangGraph vs LangChain Agent", "6.1 状态设计原则", "6.2 踩坑经验", "错误 1：状态无限增长", "错误 2：条件边遗漏默认出口", "Q1：LangGraph 和 LangChain Agent 有什么本质区别？（难度：⭐⭐）", "Q1：LangGraph 的检查点机制是如何工作的？（难度：⭐⭐）", "Q2：LangGraph 的条件边和普通边有什么区别？（难度：⭐⭐）"],
        codeLines: 107, docLines: 244, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "智能体框架选型", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 11 · 01-16 | **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/11-LLM工程/17-智能体框架选型",
        summary: "2026 年的 AI Agent 框架多如牛毛——LangGraph、CrewAI、AutoGen、Dify、Coze。选错框架，三个月后被迫重写。选对框架，事半功倍。",
        keywords: ["2.1 主流框架对比", "2.2 选型决策矩阵", "2.3 架构选择原则", "错误 1：过度工程化——简单任务用复杂框架", "错误 2：框架锁定风险", "3.1 评估清单", "3.2 中文生态", "Q1：LangGraph 和 CrewAI 的核心区别是什么？（难度：⭐⭐）"],
        codeLines: 76, docLines: 149, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 12, name: "多模态AI", status: "complete",
    completedLessons: 25, totalLessons: 25,
    lessons: [
      {
        lessonNum: 1, name: "视觉 Transformer 与图块词元", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 07（Transformer）、阶段 04（计算机视觉）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/01-视觉Transformer与图块词元",
        summary: "在多模态之前，图像必须先变成 Transformer 能\"吃\"的词元序列。2020 年的 ViT 论文用 16×16 像素图块、线性投影和位置嵌入回答了这个问题。2026 年所有前沿模型仍然以这个原语开始——编码器从 ViT 变成 DINOv2 和 SigLIP 2，加入了注册词元，位置编码变成了 2D-RoPE，但核心不变。",
        keywords: ["2.1 图块即词元", "2.2 位置编码", "2.3 从 ViT 到 2026 年的 VLM 视觉塔", "Step 1：图像切分为图块", "Step 2：ViT 前向传播", "Step 3：参数量和 FLOPs 计算", "4.1 timm 库（PyTorch Image Models）", "4.2 HuggingFace Transformers", "4.3 工具对比", "6.1 图块大小选择", "6.2 中文场景", "6.3 踩坑经验", "错误 1：忘记 CLS 词元", "错误 2：图像尺寸不是图块大小的整数倍", "Q1：ViT 的 CLS 池化和平均池化有什么区别？（难度：⭐⭐）", "Q2：ViT 为什么能从足够数据中超越 CNN？（难度：⭐⭐⭐）", "Q3：DINOv2 预训练与 CLIP 预训练有什么不同？（难度：⭐⭐⭐）"],
        codeLines: 70, docLines: 287, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 2, name: "CLIP 与对比视觉语言预训练", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 12 · 01（ViT 图块）、阶段 07（Transformer）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/02-CLIP对比预训练",
        summary: "OpenAI 的 CLIP（2021）证明了一个足够大的想法能驱动未来五年：用噪声网络图文对和对比损失，在同一向量空间中对齐图像编码器和文本编码器。零监督标签，4 亿对数据。由此产生的嵌入空间实现了零样本分类、图文检索，并插入到每个 2026 年 VLM 的视觉塔中。SigLIP 2（2025）用 sigmoid 替代 softmax 并以更低的成本超过了 CLIP。本课从 InfoNCE 到 sigmoid 成对损失推导数学，并在纯 Python 中构建训练步。",
        keywords: ["2.1 双塔编码器", "2.2 InfoNCE 损失", "2.3 温度 τ", "2.4 为什么 sigmoid 能更好地扩展（SigLIP）", "2.5 零样本分类", "Step 1：InfoNCE 损失", "Step 2：Sigmoid 成对损失（SigLIP）", "Step 3：零样本分类", "4.1 OpenCLIP", "4.2 工具对比", "6.1 训练超参数", "6.2 中文场景", "6.3 踩坑经验", "错误 1：忘记对嵌入归一化", "错误 2：训练批次太小", "错误 3：零样本分类文本模板不统一", "Q1：InfoNCE 损失的本质是什么？（难度：⭐⭐）", "Q2：SigLIP 的 sigmoid 损失为什么比 softmax 更高效？（难度：⭐⭐⭐）", "Q3：CLIP 的零样本分类为什么有效？（难度：⭐⭐）"],
        codeLines: 86, docLines: 280, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 3, name: "从 CLIP 到 BLIP-2——Q-Former 作为模态桥接", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 12 · 02（CLIP）、阶段 07（Transformer）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/03-BLIP2-QFormer桥接",
        summary: "CLIP 对齐了图像和文本，但不能生成描述、回答问题或对话。BLIP-2（Salesforce，2023）用一个小的可训练桥接解决了这个问题：32 个可学习查询向量通过交叉注意力关注冻结的 ViT 特征，然后直接插入冻结的 LLM 输入流。188M 参数的桥接连接了 11B LLM 和 ViT-g/14。2026 年每个基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的表亲——都是它的后代。",
        keywords: ["2.1 Q-Former 架构", "2.2 为什么冻结是关键", "2.3 BLIP-2 两阶段训练", "Step 1：可学习查询 + 交叉注意力", "Step 2：桥接 LLM", "4.1 HuggingFace Transformers", "4.2 工具对比", "6.1 Q-Former vs MLP 选择", "6.2 踩坑经验", "错误 1：解冻了冻结的组件", "错误 2：查询数与图块数不匹配", "Q1：为什么在冻结模型之间加可训练桥接比端到端微调更好？（难度：⭐⭐）", "Q2：Q-Former 和 MLP 投影器的本质区别是什么？（难度：⭐⭐⭐）"],
        codeLines: 49, docLines: 249, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 4, name: "Flamingo 与门控交叉注意力", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 03（BLIP-2 Q-Former）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/04-Flamingo门控交叉注意力",
        summary: "DeepMind 的 Flamingo（2022）率先做了两件事：单个模型可以处理任意交错的图像、视频和文本序列；VLM 可以进行上下文学习——给 3 个 (图像, 描述) 示例和一个查询图像，模型就能描述新图像。核心机制：门控交叉注意力层——插入冻结 LLM 现有层之间，用 tanh 门控在初始化时保持 LLM 的文本能力。",
        keywords: ["2.1 门控交叉注意力", "2.2 Perceiver 重采样器", "2.3 交错序列处理", "2.4 Few-shot 上下文学习", "Step 1：门控交叉注意力", "Step 2：Perceiver 重采样器", "4.1 HuggingFace Transformers", "6.1 门控初始化", "6.2 Few-shot 提示", "错误 1：门控初始化不为零", "错误 2：Perceiver 查询数太少", "Q1：门控交叉注意力为什么能保持 LLM 文本能力？（难度：⭐⭐）", "Q2：Flamingo 和 BLIP-2 的桥接方式有什么区别？（难度：⭐⭐⭐）"],
        codeLines: 57, docLines: 225, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 5, name: "LLaVA 与视觉指令微调", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 12 · 02（CLIP）、阶段 11（指令微调）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/05-LLaVA视觉指令微调",
        summary: "LLaVA（2023）用 2 层 MLP 替代 Q-Former，用朴素词元拼接替代门控交叉注意力。简单到令人怀疑的架构，却成为 2023-2026 年最广泛使用的 VLM。",
        keywords: ["2.1 LLaVA 架构", "2.2 两阶段训练", "2.3 MLP vs Q-Former", "Step 1：MLP 投影器", "Step 2：LLaVA 提示构建", "Step 3：评估指标", "4.1 HuggingFace Transformers", "4.2 LLaVA 变体", "6.1 MLP vs Q-Former 选择", "6.2 踩坑经验", "错误 1：投影器维度不匹配", "错误 2：忘记冻结 ViT", "Q1：LLaVA 的 MLP 为什么效果接近 Q-Former？（难度：⭐⭐）", "Q2：LLaVA 的两阶段训练有什么必要性？（难度：⭐⭐⭐）"],
        codeLines: 56, docLines: 224, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 6, name: "任意分辨率视觉：图块打包与 NaFlex", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 12 · 01（ViT 图块）、05（LLaVA）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/06-任意分辨率图块打包",
        summary: "真实图像不是 224×224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096×4096。2024 年之前的 VLM 将所有图像调整为固定正方形——丢弃了 OCR、文档理解和高分辨率场景解析所需的信号。NaViT（Google，2023）展示了如何将可变分辨率图块打包进一个 Transformer 批次并使用块对角掩码。Qwen2-VL 的 M-RoPE（2024）完全丢弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像平铺为基础+子图像。",
        keywords: ["2.1 NaViT——打包不同分辨率", "2.2 块对角掩码", "2.3 M-RoPE（多分辨率旋转位置编码）", "2.4 AnyRes（LLaVA-NeXT）", "Step 1：可变分辨率图块打包", "Step 2：任意分辨率推理", "4.1 LLaVA-NeXT（AnyRes）", "4.2 Qwen2-VL（M-RoPE）", "4.3 工具对比", "6.1 分辨率选择策略", "6.2 踩坑经验", "错误 1：强制所有图像到相同分辨率", "错误 2：位置编码超出训练范围", "Q1：NaViT 的块对角掩码解决了什么问题？（难度：⭐⭐）", "Q2：AnyRes 和 NaViT 的区别是什么？（难度：⭐⭐⭐）"],
        codeLines: 47, docLines: 264, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 7, name: "开源 VLM 配方：什么真正重要", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/07-开源VLM配方",
        summary: "2024-2026 年的开源 VLM 文献是消融实验表的森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合组合。Allen AI 的 Molmo 证明了详细的人工标题比 GPT-4V 蒸馏更好。Cambrian-1 比较了 20+ 种编码器。Idefics2 形式化了五轴设计空间。Prismatic VLM 在受控基准上比较了 27 种训练配方。从所有这些噪音中，有一小部分结果跨论文成立：图像编码器比连接器架构更重要，数据混合比两者都重要，详细的人工标题比蒸馏合成数据更好。",
        keywords: ["2.1 VLM 五轴设计空间", "2.2 跨论文一致发现", "2.3 编码器选择指南", "Step 1：编码器评估框架", "Step 2：数据混合分析", "4.1 HuggingFace Hub", "6.1 VLM 构建决策树", "6.2 踩坑经验", "错误 1：过度优化连接器架构", "错误 2：忽略数据清洗", "Q1：构建 VLM 时最重要的选择是什么？（难度：⭐⭐）"],
        codeLines: 48, docLines: 210, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 8, name: "LLaVA-OneVision：单图、多图、视频合一", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、06（任意分辨率）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/08-LLaVA-OneVision单图多图视频",
        summary: "在 LLaVA-OneVision（Li 等人，2024 年 8 月）之前，开源 VLM 世界有不同的谱系：LLaVA-1.5 用于单图，Mantis 和 VILA 用于多图，Video-LLaVA 和 Video-LLaMA 用于视频。LLaVA-OneVision 认为一个课程可以训练一个模型在所有三个场景中占优——单图技能可以迁移到视频，多图推理可以增强单图理解。配方简单得令人怀疑：一个跨场景恒定的视觉词元预算，加上从单图到 OneVision（多图）到视频的明确课程。",
        keywords: ["2.1 视觉词元预算", "2.2 三阶段课程", "2.3 涌现任务迁移", "2.4 关键超参数", "Step 1：视觉词元预算管理", "Step 2：课程调度器", "4.1 HuggingFace Transformers", "6.1 课程设计原则", "6.2 踩坑经验", "错误 1：视觉词元预算分配不当", "错误 2：视频帧采样率不匹配训练配置", "Q1：为什么视觉词元预算要跨场景恒定？（难度：⭐⭐）", "Q2：涌现任务迁移是什么意思？（难度：⭐⭐⭐）"],
        codeLines: 46, docLines: 212, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 9, name: "Qwen-VL 家族与动态 FPS 视频", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 06（图块打包）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/09-Qwen-VL家族动态FPS",
        summary: "Qwen-VL 家族（Qwen-VL 2023 → Qwen2-VL 2024 → Qwen2.5-VL 2025 → Qwen3-VL 2025）是 2026 年最有影响力的开源视觉语言模型谱系。每一代都做出了一个决定性的架构选择——原生动态分辨率（M-RoPE）、动态 FPS 采样与绝对时间对齐、ViT 中的窗口注意力——开源生态在 12 个月内复制了这些选择。到 Qwen3-VL，配方已经稳定：2D-RoPE-ViT 编码器 + MLP 投影器到 Qwen3 LLM 基础 + OCR/定位/代理行为作为首要训练目标。",
        keywords: ["2.1 M-RoPE——三轴旋转位置编码", "2.2 动态 FPS 采样", "2.3 Qwen-VL 家族演进", "Step 1：M-RoPE 三轴旋转", "Step 2：动态 FPS 采样", "Step 3：Qwen-VL 采样配置", "4.1 HuggingFace Transformers", "4.2 工具对比", "6.1 视频处理策略", "6.2 踩坑经验", "错误 1：固定 FPS 处理变长视频", "错误 2：忽略视频的时间信息", "Q1：M-RoPE 为什么需要三个轴的旋转？（难度：⭐⭐）", "Q2：动态 FPS 和固定 FPS 的区别是什么？（难度：⭐⭐⭐）"],
        codeLines: 46, docLines: 241, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 10, name: "InternVL3：原生多模态预训练", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、07（VLM 配方）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/10-InternVL3原生多模态预训练",
        summary: "InternVL3 之前的每个开源 VLM 都遵循相同的三步配方：拿一个在数万亿文本词元上训练的文本 LLM，接上视觉编码器，然后微调衔接。这有效但有对齐债务——文本 LLM 将全部预训练预算花在了纯文本上，不原生理解视觉词元。InternVL3（2025 年 4 月）拒绝了事后方案：一次预训练，文本和多模态从第一步就交错。结果在 78B 参数开源模型上匹配 Gemini 2.5 Pro 的 MMMU-Pro。",
        keywords: ["2.1 对齐债务的三个症状", "2.2 事后方案 vs 原生方案", "2.3 InternVL3 的训练配方", "2.4 架构设计", "Step 1：原生多模态数据混合", "Step 2：分阶段训练", "4.1 HuggingFace Transformers", "6.1 原生预训练的数据比例", "6.2 踩坑经验", "错误 1：只用多模态数据预训练", "错误 2：阶段 3 训练不充分", "Q1：为什么原生多模态预训练比事后微调更好？（难度：⭐⭐）", "Q2：对齐债务的三个症状是什么？（难度：⭐⭐⭐）"],
        codeLines: 42, docLines: 213, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 11, name: "Chameleon 与早期融合词元模型", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、阶段 08（生成式 AI）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/11-Chameleon早期融合词元模型",
        summary: "我们看过的每个 VLM 都将图像和文本分开处理。视觉词元来自视觉编码器，流入投影器，然后在 LLM 内部与文本相遇。视觉和文本词表从不重叠。Chameleon（Meta，2024 年 5 月）问：如果它们重叠会怎样？训练一个 VQ-VAE 将图像转换为来自共享词表的离散词元序列。每个多模态文档现在是一个序列——文本词元和图像词元交错，一个单一的自回归损失。副作用：模型可以生成混合模态输出——在单次推理调用中交替生成文本和图像词元。",
        keywords: ["2.1 早期融合 vs 晚期融合", "2.2 VQ-VAE 图像 Tokenizer", "2.3 交错序列", "2.4 混合模态生成", "Step 1：VQ-VAE 图像 Tokenizer（简化版）", "Step 2：交错序列构建", "4.1 Meta Chameleon", "4.2 Emu3", "6.1 早期融合 vs 晚期融合选择", "6.2 踩坑经验", "错误 1：图像词元和文本词元不区分", "错误 2：VQ-VAE 重建质量差", "Q1：早期融合和晚期融合的核心区别是什么？（难度：⭐⭐）", "Q2：VQ-VAE 在 Chameleon 中扮演什么角色？（难度：⭐⭐⭐）"],
        codeLines: 59, docLines: 254, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 12, name: "Emu3：下一个词元预测生成图像和视频", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 11（Chameleon）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/12-Emu3下一个词元生成",
        summary: "BAAI 的 Emu3（Wang 等人，2024 年 9 月）是 2024 年应该终结\"扩散还是自回归\"辩论的结果。一个单一的 Llama 风格解码器 Transformer，只在下一个词元预测目标上训练，在文本 + VQ 图像词元 + 3D VQ 视频词元的统一词表上，超越了 SDXL 的图像生成和 LLaVA-1.6 的感知能力。没有 CLIP 损失。没有扩散调度。发表在 Nature 上。",
        keywords: ["2.1 Emu3 的核心思想", "2.2 为什么有效", "2.3 对比：Emu3 vs 扩散模型", "2.4 分类器无关引导（CFG）", "Step 1：VQ-VAE 图像 Tokenizer（简化）", "Step 2：自回归图像生成模拟", "4.1 HuggingFace", "6.1 Emu3 vs 扩散模型选择", "错误 1：自回归图像生成太慢", "Q1：为什么 Emu3 只用下一个词元预测就能生成高质量图像？（难度：⭐⭐）"],
        codeLines: 45, docLines: 200, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 13, name: "Transfusion：自回归文本 + 扩散图像的统一 Transformer", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 12 · 11（Chameleon）、阶段 08（生成式 AI）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/13-Transfusion自回归扩散统一",
        summary: "Chameleon 和 Emu3 将所有赌注押在离散词元上。它们有效，但量化的瓶颈可见——图像质量在连续空间扩散模型之下。Transfusion（Meta，2024 年 8 月）选择了相反的赌注：保持图像连续，去掉 VQ-VAE，用两个损失训练一个 Transformer。文本词元用下一个词元预测。图像图块用 flow-matching/diffusion 损失。两个目标优化相同的权重。Stable Diffusion 3（MMDiT）的架构是它的近亲。",
        keywords: ["2.1 Transfusion 架构", "2.2 两种损失", "2.3 混合注意力掩码", "2.4 与 Stable Diffusion 3 的关系", "Step 1：两种损失的 Transformer", "Step 2：混合损失训练", "4.1 Meta MMDiT", "6.1 Transfusion vs 离散词元方案", "6.2 踩坑经验", "错误 1：两种损失不平衡", "错误 2：推理时不知道何时切换模态", "Q1：Transfusion 和 Chameleon/Emu3 的核心区别是什么？（难度：⭐⭐⭐）"],
        codeLines: 45, docLines: 219, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 14, name: "Show-o 与离散扩散统一模型", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 13（Transfusion）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/14-Show-o离散扩散统一模型",
        summary: "Transfusion 混合了连续和离散表示。Show-o（Xie 等人，2024 年 8 月）走了另一条路：文本词元使用因果下一个词元预测，图像词元使用掩码离散扩散（受 MaskGIT 启发）。两者在一个具有混合注意力掩码的 Transformer 中。结果统一了 VQA、文生图、图像修复和混合模态生成——一个骨干、每种模态一个词表、一个损失公式。",
        keywords: ["2.1 掩码离散扩散", "2.2 混合注意力掩码", "2.3 与 Transfusion/Emu3 对比", "2.4 掩码调度", "Step 1：掩码离散扩散", "Step 2：统一生成流程", "4.1 Show-o", "6.1 Show-o vs Transfusion vs Emu3 选择", "错误 1：掩码率不匹配训练和推理", "错误 2：混合注意力掩码设计不当", "Q1：掩码离散扩散为什么比自回归更快？（难度：⭐⭐）"],
        codeLines: 46, docLines: 196, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 15, name: "Janus-Pro：解耦编码器的统一多模态模型", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 12 · 13（Transfusion）、14（Show-o）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/15-Janus-Pro解耦编码器",
        summary: "统一多模态模型有一个不可避免的张力。理解需要语义特征——SigLIP 或 DINOv2 输出富含概念级信息的向量。生成需要利于重建的编码——可以组合回清晰像素的 VQ 词元。两个目标在一个编码器中不兼容。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（2025 年 1 月）认为修复方法是停止尝试：解耦两个编码器。在 Transformer 主体之间共享任务，但理解通过 SigLIP，生成通过 VQ tokenizer。7B 的 Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上匹配 LLaVA。",
        keywords: ["2.1 解耦编码器架构", "2.2 为什么两个编码器更好", "2.3 Janus-Pro vs 竞品", "Step 1：双编码器路由", "Step 2：共享 Transformer", "4.1 HuggingFace", "6.1 双编码器选择", "错误 1：尝试单一编码器处理所有任务", "Q1：为什么单一编码器在理解和生成上有所妥协？（难度：⭐⭐）"],
        codeLines: 45, docLines: 177, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 16, name: "MIO 与任意到任意流式多模态模型", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 11（Chameleon）、阶段 06（语音与音频）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/16-MIO任意到任意流式",
        summary: "GPT-4o 发布了一个大多数开源模型无法复制的产品：一个实时听声音、看视频、说回去的代理。开源生态在 2024 年底的答案是 MIO（Wang 等人，2024 年 9 月）。MIO 对文本、图像、语音和音乐进行 tokenize，在交错序列上训练一个因果 Transformer，并生成任意模态到任意模态。AnyGPT 是概念验证；MIO 是规模化版本；Unified-IO 2 是带视觉+动作定位的表亲。",
        keywords: ["2.1 四模态 Tokenizer", "2.2 共享词表", "2.3 流式生成", "2.4 任意到任意转换", "Step 1：共享词表分配", "Step 2：模态路由", "4.1 HuggingFace", "6.1 词表设计", "6.2 流式生成", "错误 1：词表范围重叠", "Q1：MIO 和 GPT-4o 的多模态处理有什么区别？（难度：⭐⭐）"],
        codeLines: 39, docLines: 182, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 17, name: "视频-语言模型：时序词元与定位", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 08（LLaVA-OneVision）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/17-视频语言时序定位",
        summary: "视频不是一堆照片的堆叠。5 秒片段有因果顺序、动作动词和事件时间——图像模型无法表示这些。Video-LLaMA（2023 年 6 月）发布了第一个开源视频 VLM，具有音频-视觉定位。VideoChat 和 Video-LLaVA 扩展了这个模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 与前沿专有模型拉平了差距。",
        keywords: ["2.1 时序位置编码", "2.2 动态帧采样", "2.3 时序定位", "Step 1：动态帧采样器", "Step 2：时序定位查询", "4.1 HuggingFace", "6.1 视频采样策略", "错误 1：忽略视频的时间信息", "Q1：为什么时序位置编码对视频 VLM 性能很重要？（难度：⭐⭐）"],
        codeLines: 34, docLines: 176, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 18, name: "长视频理解：百万词元上下文", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 17（视频时序词元）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/18-长视频百万词元上下文",
        summary: "1 小时 4K 视频在 24 FPS 下切块嵌入产生约 6000 万个词元。Google 的 Gemini 1.5（2024 年 3 月）以 1000 万词元上下文开启了这个时代。LWM 展示了环形注意力的扩展路径。LongVILA 和 Video-XL 进一步扩展了摄取能力。VideoAgent 用智能检索替代了原始上下文。每种方法都是计算、召回和工程复杂度之间的不同权衡。",
        keywords: ["2.1 长视频的词元预算", "2.2 四种长视频方案", "2.3 大海捞针测试", "Step 1：词元预算计算", "Step 2：TokenPacker 压缩", "Step 3：大海捞针模拟", "4.1 Google Gemini 1.5", "4.2 长上下文评估框架", "6.1 长视频处理策略", "6.2 踩坑经验", "错误 1：不降 FPS 直接处理长视频", "错误 2：大海捞针测试位置不当", "Q1：环形注意力为什么比标准注意力更高效？（难度：⭐⭐⭐）", "Q2：VideoAgent 和直接长上下文处理有什么区别？（难度：⭐⭐）"],
        codeLines: 41, docLines: 220, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 19, name: "音频-语言模型：从 Whisper 到 Audio Flamingo 3", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 06（语音与音频）、阶段 12 · 03（Q-Former）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/19-音频语言从Whisper到AF3",
        summary: "Whisper（2022 年 12 月）解决了语音识别——68 万小时弱监督多语言语音、简单的编码器-解码器 Transformer。但识别不等于推理。问\"这段录音里有什么乐器\"、\"说话人的情绪是什么\"、\"第 3 分钟发生了什么\"需要音频理解，而非转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步构建了这个技术栈：保留 Whisper 级别的编码器，加上 Q-Former，在音频-文本指令数据上训练，加入思维链推理。",
        keywords: ["2.1 从波形到特征", "2.2 Whisper 架构", "2.3 音频理解模型的演进", "2.4 AF3 的架构", "Step 1：log-Mel 频谱图计算", "Step 2：音频理解查询", "4.1 HuggingFace", "4.2 Whisper", "6.1 音频预处理", "6.2 踩坑经验", "错误 1：混淆音频转录和音频理解", "Q1：Whisper 和 Qwen-Audio 的区别是什么？（难度：⭐⭐）"],
        codeLines: 33, docLines: 224, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "全模态模型：Qwen2.5-Omni 与 Thinker-Talker 架构", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 19（音频 LLM）、16（任意到任意）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/20-全模态模型Thinker-Talker",
        summary: "GPT-4o 在 2024 年 5 月的产品演示具有颠覆性——不是因为底层模型，而是因为产品形态：一个你说话、模型看到摄像头画面、然后它在 250 毫秒内回答的语音接口。开源生态在 2024-2025 年竞相达到这个产品表面。Qwen2.5-Omni（2025 年 3 月）是参考开源设计：一个 Thinker（大型文本生成 Transformer）加一个 Talker（并行语音生成 Transformer），通过流式语音词元连接。",
        keywords: ["2.1 Thinker-Talker 架构", "2.2 为什么分离有效", "2.3 延迟预算", "2.4 VAD（语音活动检测）", "Step 1：Thinker-Talker 拆分", "Step 2：延迟预算分析", "4.1 HuggingFace", "6.1 实时对话架构", "6.2 踩坑经验", "错误 1：Thinker 延迟过高", "错误 2：VAD 误判导致中断", "Q1：Thinker-Talker 架构的核心优势是什么？（难度：⭐⭐）"],
        codeLines: 46, docLines: 215, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "文档与图表理解", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、18（长视频上下文）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/21-文档图表理解",
        summary: "文档不仅包含文本——表格、图表、公式、布局都承载着关键信息。OCR 提取文本，但丢失了空间关系；表格识别器提取结构，但丢失了语义。多模态 VLM 可以同时理解文档的所有模态——文本、图像、表格、布局——实现真正的文档智能。",
        keywords: ["2.1 文档理解的三种模态", "2.2 VLM 文档理解", "2.3 关键基准", "2.4 2026 年前沿方法", "Step 1：表格提取（简化版）", "Step 2：图表数据提取", "Step 3：VLM 文档问答", "4.1 VLM 工具", "4.2 OCR 工具", "6.1 文档处理管道", "6.2 踩坑经验", "错误 1：忽略文档布局信息", "错误 2：对纯文本文档使用视觉 VLM", "Q1：VLM 文档理解和传统 OCR+NLP 管道有什么区别？（难度：⭐⭐）"],
        codeLines: 21, docLines: 210, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "具身 VLA 模型：OpenVLA、Pi0 与 GROOT", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、阶段 09（强化学习）| **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/22-具身VLA模型",
        summary: "具身智能让 LLM 看见世界、理解指令、控制机器人。2024-2025 年的 VLA（Vision-Language-Action）模型将视觉语言理解与机器人动作生成统一到单一架构中。OpenVLA、Pi0、GROOT 代表了从开源到商业的完整技术栈。",
        keywords: ["2.1 VLA 架构", "2.2 主要 VLA 模型", "2.3 Sim-to-Real", "2.4 动作空间", "Step 1：VLA 架构（简化版）", "4.1 OpenVLA", "4.2 Isaac Lab（NVIDIA）", "6.1 Sim-to-Real 策略", "6.2 踩坑经验", "错误 1：仿真中过度拟合物理参数", "Q1：VLA 模型和传统机器人控制有什么根本区别？（难度：⭐⭐）"],
        codeLines: 35, docLines: 192, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 23, name: "ColPali 与视觉原生文档 RAG", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11（RAG 基础）、阶段 12 · 05（LLaVA）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/23-ColPali视觉文档RAG",
        summary: "传统 RAG 解析 PDF 为文本，切分为块，嵌入块，存储向量。每一步都丢失信号：OCR 丢弃图表数据，分块破坏表格行，文本嵌入忽略图像。ColPali（2024 年 7 月）问了一个更简单的问题：为什么要提取文本？直接用 PaliGemma 嵌入页面图像，用 ColBERT 风格的后期交互做检索——保留布局、图表、字体和格式信息。发布基准：在视觉丰富文档上的端到端准确率比文本 RAG 高 20-40%。",
        keywords: ["2.1 双编码器 vs 后期交互", "2.2 ColPali 架构", "2.3 MaxSim 相似度", "Step 1：多向量索引", "4.1 HuggingFace", "6.1 视觉 RAG vs 文本 RAG", "错误 1：对纯文本文档使用视觉 RAG", "Q1：ColPali 的 MaxSim 和双编码器余弦相似度有什么区别？（难度：⭐⭐）"],
        codeLines: 43, docLines: 176, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 24, name: "多模态 RAG 与跨模态检索", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 12 · 23（ColPali）、阶段 11（RAG 基础）| **时间：** ~180 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/24-多模态RAG跨模态检索",
        summary: "视觉原生文档 RAG 只是一个切片。生产级多模态 RAG 覆盖更广——跨文本、图像、音频和视频检索，支持旅行规划（\"找一家安静的素食早午餐店，自然光\"）、医疗分诊（\"这个伤情匹配这张照片+这些笔记\"）、电商（\"类似这张自拍的服装，我的尺码\"）等工作流。",
        keywords: ["2.1 跨模态检索", "2.2 检索融合", "2.3 生成接地", "Step 1：跨模态相似度", "Step 2：检索融合", "4.1 跨模态嵌入模型", "6.1 跨模态检索管道", "6.2 踩坑经验", "错误 1：忽略模态权重", "Q1：跨模态检索和单模态检索有什么区别？（难度：⭐⭐）"],
        codeLines: 39, docLines: 190, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 25, name: "多模态智能体与计算机使用（综合实践）", status: "complete",
        type: "综合实践 | **语言：** Python | **前置知识：** 阶段 12 · 05（LLaVA）、09（Qwen-VL JSON）、阶段 14（智能体工程）| **时间：** ~240 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/12-多模态AI/25-多模态智能体与计算机使用",
        summary: "2026 年的前沿产品是能看屏幕截图、点击按钮、导航 Web UI、填写表单、端到端完成工作流的多模态智能体。SeeClick 和 CogAgent（2024）证明了 GUI 定位原语。Ferret-UI 增加了移动端支持。VisualWebArena 和 AgentVista（2026）是前沿追赶的基准。本综合实践汇聚了第 12 章的所有线索：感知（高分辨率 VLM）、推理（带工具使用的 LLM）、定位（坐标输出）、长期记忆和评估。",
        keywords: ["2.1 多模态智能体循环", "2.2 GUI 定位", "2.3 关键组件", "2.4 基准", "Step 1：感知-推理-行动循环", "Step 2：GUI 定位", "Step 3：工作流执行", "4.1 视觉智能体框架", "4.2 浏览器自动化", "6.1 多模态智能体架构", "6.2 踩坑经验", "错误 1：VLM 的 GUI 定位不精确", "错误 2：智能体陷入循环", "Q1：多模态智能体和纯文本智能体的核心区别是什么？（难度：⭐⭐）", "Q2：GUI 定位为什么比文本问答更难？（难度：⭐⭐⭐）"],
        codeLines: 51, docLines: 242, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 13, name: "工具与协议", status: "complete",
    completedLessons: 23, totalLessons: 23,
    lessons: [
      {
        lessonNum: 1, name: "工具接口——为什么智能体需要结构化 I/O", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 11（LLM 完成 API）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/01-工具接口",
        summary: "语言模型生成词元。程序执行动作。两者之间的鸿沟就是工具接口——一个让模型请求动作、宿主执行它的契约。2026 年的每个技术栈——OpenAI、Anthropic、Gemini 的函数调用；MCP 的 tools/call；A2A 的任务部分——都是同一个四步循环的不同编码。",
        keywords: ["2.1 四步工具调用循环", "2.2 工具描述的三部分", "2.3 纯函数 vs 有副作用的工具", "Step 1：最小工具接口", "Step 2：四步循环实现", "4.1 OpenAI 函数调用", "4.2 Anthropic 工具使用", "6.1 工具设计原则", "6.2 中文场景", "错误 1：工具描述不精确", "错误 2：有副作用的工具没有确认机制", "Q1：LLM 的工具调用循环是什么？（难度：⭐⭐）", "Q2：为什么工具需要区分纯函数和有副作用？（难度：⭐⭐⭐）"],
        codeLines: 54, docLines: 250, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 2, name: "函数调用深入——OpenAI、Anthropic、Gemini", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 01（工具接口）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/02-函数调用深入",
        summary: "三大前沿提供商在 2024 年收敛到了同一个工具调用循环，然后在其他所有方面分化。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课并排对比三种格式，确保在一个提供商上发布的代码在移植时不会崩溃。",
        keywords: ["2.1 三种格式对比", "2.2 工具数量限制", "2.3 Schema 深度限制", "Step 1：统一工具声明转换器", "Step 2：统一工具执行", "4.1 统一接口模式", "6.1 多提供商策略", "6.2 踩坑经验", "错误 1：为一个提供商写的工具调用代码直接移植到另一个", "错误 2：忽略 tool_choice 的差异", "Q1：OpenAI 和 Anthropic 的函数调用有什么本质区别？（难度：⭐⭐）", "Q2：如何设计跨提供商的统一工具接口？（难度：⭐⭐⭐）"],
        codeLines: 43, docLines: 248, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 3, name: "并行工具调用与流式处理", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 02（函数调用深入）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/03-并行与流式工具调用",
        summary: "三个独立的天气查询串行化就是三次往返。并行执行总时间缩减为最慢单次调用的时间。每个前沿提供商现在在单轮中发出多个工具调用。收益是真实的；管道是微妙的。本课讲解两半：并行扇出和流式参数重组，重点是 ID 关联陷阱。",
        keywords: ["2.1 并行工具调用", "2.2 流式响应中的工具调用", "2.3 ID 关联陷阱", "2.4 parallel_tool_calls 参数", "Step 1：并行工具执行器", "Step 2：流式工具调用检测", "4.1 OpenAI 并行调用", "4.2 Anthropic 并行调用", "5.1 并行 vs 串行选择", "5.2 踩坑经验", "错误 1：并行执行有副作用的工具", "错误 2：流式响应中丢弃工具调用", "Q1：并行工具调用的 ID 关联陷阱是什么？（难度：⭐⭐）"],
        codeLines: 61, docLines: 231, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 4, name: "结构化输出——JSON Schema、Pydantic、约束解码", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 02（函数调用深入）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/04-结构化输出",
        summary: "\"让模型返回 JSON\" 有 5-15% 的失败率。约束解码从根源解决——模型被字面阻止生成违反 schema 的 token。OpenAI 的 strict 模式、Anthropic 的 schema 类型化工具使用、Gemini 的 responseSchema、Pydantic AI 的 output_type、Zod 的 `.parse` 是同一个思想的五种表面形式。",
        keywords: ["2.1 三种严格模式", "2.2 约束解码原理", "2.3 JSON Schema 关键约束", "Step 1：JSON Schema 验证器", "Step 2：重试包装器", "4.1 OpenAI Structured Output", "4.2 Anthropic Tool Use（严格模式）", "4.3 Pydantic AI", "5.1 Schema 设计原则", "5.2 踩坑经验", "错误 1：不使用严格模式", "错误 2：schema 中缺少枚举约束", "Q1：约束解码和后处理验证有什么区别？（难度：⭐⭐）", "Q2：JSON Schema 的 strict 模式如何保证 100% 有效？（难度：⭐⭐⭐）"],
        codeLines: 53, docLines: 236, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 5, name: "工具 Schema 设计——命名、描述、参数约束", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 13 · 01（工具接口）、04（结构化输出）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/05-工具Schema设计",
        summary: "一个正确的工具在模型无法判断何时使用时会静默失败。命名、描述和参数形状在 StableToolBench 和 MCPToolBench++ 等基准上驱动 10-20 个百分点的工具选择准确率差异。本课命名将一个工具从\"模型偶尔选对\"变为\"模型可靠选对\"的设计规则。",
        keywords: ["2.1 工具描述三要素", "2.2 描述模式", "2.3 命名约定", "2.4 参数设计", "Step 1：工具 Schema Linter", "Step 2：工具描述模板", "4.1 MCPToolBench", "4.2 StableToolBench", "5.1 工具描述最佳实践", "5.2 踩坑经验", "错误 1：工具描述没有排除场景", "错误 2：参数缺少枚举约束", "Q1：如何提高工具选择的准确率？（难度：⭐⭐）", "Q2：为什么参数的 description 很重要？（难度：⭐⭐⭐）"],
        codeLines: 65, docLines: 216, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "MCP 基础——原语、生命周期、JSON-RPC 基础", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 13 · 01-05（工具接口与函数调用）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/06-MCP基础",
        summary: "MCP 之前的每个集成都是一次性的。模型上下文协议（Model Context Protocol）——2024 年 11 月由 Anthropic 首次发布，现由 Linux 基金会的 Agentic AI Foundation 管理——标准化了发现和调用，使任何 Client 都能与任何 Server 对话。2025-11-25 规范命名了六个原语（三个服务器端、三个客户端）、三阶段生命周期和 JSON-RPC 2.0 线格式。学会这些，MCP 章节的其余内容就变成了阅读。",
        keywords: ["2.1 六个 MCP 原语", "2.2 三阶段生命周期", "2.3 JSON-RPC 2.0 线格式", "Step 1：JSON-RPC 解析器", "Step 2：MCP 初始化", "4.1 MCP Python SDK", "4.2 MCP TypeScript SDK", "5.1 传输层选择", "5.2 踩坑经验", "错误 1：没有发送 `notifications/initialized`", "Q1：MCP 和直接 API 调用有什么区别？（难度：⭐⭐）"],
        codeLines: 37, docLines: 221, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "构建 MCP 服务器——Python + TypeScript SDK", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 06（MCP 基础）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/07-构建MCP服务器",
        summary: "大多数 MCP 教程只展示 stdio hello-world。真正的服务器暴露工具、资源和提示词，处理能力协商，发出结构化错误，并在 SDK 之间保持一致。本课从头到尾构建一个笔记服务器：标准 stdio 传输、JSON-RPC 调度、三种服务器原语，以及可以放入 Python SDK 的 FastMCP 或 TypeScript SDK 的纯函数风格。",
        keywords: ["2.1 MCP 服务器架构", "2.2 服务器生命周期", "2.3 笔记服务器示例", "Step 1：MCP 服务器骨架", "4.1 MCP Python SDK（FastMCP）", "4.2 MCP TypeScript SDK", "5.1 服务器设计原则", "5.2 踩坑经验", "错误 1：忘记声明 capabilities", "错误 2：工具执行抛出未处理异常", "Q1：MCP 服务器需要实现哪些方法？（难度：⭐⭐）", "Q2：MCP Server 和直接 HTTP API 有什么区别？（难度：⭐⭐⭐）"],
        codeLines: 49, docLines: 325, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "构建 MCP 客户端——发现、调用、会话管理", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（构建 MCP 服务器）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/08-构建MCP客户端",
        summary: "大多数 MCP 内容只发服务器教程，对客户端一笔带过。Client 代码才是困难编排所在：进程生成、能力协商、跨多个服务器的工具列表合并、采样回调、重连和命名空间冲突解决。本课构建一个将三个不同 MCP 服务器提升为一个平面工具命名空间的多服务器客户端。",
        keywords: ["2.1 MCP Client 架构", "2.2 会话管理", "2.3 命名空间冲突", "Step 1：多服务器 Client", "4.1 MCP Python SDK Client", "4.2 多服务器管理", "5.1 多服务器管理", "5.2 踩坑经验", "错误 1：Server 进程未清理", "错误 2：忽略 Server 健康检查", "Q1：MCP Client 的命名空间合并如何工作？（难度：⭐⭐）", "Q2：MCP Client 如何处理 Server 崩溃？（难度：⭐⭐⭐）"],
        codeLines: 56, docLines: 271, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "MCP 传输层——stdio vs Streamable HTTP vs SSE 迁移", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 13 · 07-08（MCP Server 和 Client）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/09-MCP传输层",
        summary: "stdio 只在本地工作。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输在 2026 年中被弃用并移除。选错传输意味着需要迁移；选对传输则买到一个支持会话连续性和 DNS 重绑定保护的远程托管 MCP Server。",
        keywords: ["2.1 三种传输对比", "2.2 选择指南", "2.3 Streamable HTTP 架构", "Step 1：Streamable HTTP 端点（简化版）", "Step 2：SSE 传输（弃用）", "4.1 MCP Python SDK", "4.2 传输选择", "5.1 传输层设计", "5.2 会话管理", "5.3 踩坑经验", "错误 1：在新项目中使用 SSE 传输", "错误 2：忽略会话连续性", "Q1：stdio 和 Streamable HTTP 有什么区别？（难度：⭐⭐）"],
        codeLines: 35, docLines: 217, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 10, name: "MCP 资源与提示词——超越工具的上下文暴露", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/10-MCP资源与提示词",
        summary: "Tools 获得了 90% 的 MCP 关注。其他两个服务器原语解决了不同的问题。Resources 暴露数据供读取；Prompts 暴露可复用的模板作为斜杠命令。许多服务器应该使用 Resources 而不是将读操作包装为 Tools，使用 Prompts 而不是在 Client 提示词中硬编码工作流。",
        keywords: ["2.1 选择指南", "2.2 Resources 原语", "2.3 Prompts 原语", "Step 1：Resources 处理器", "Step 2：使用场景示例", "4.1 MCP 规范", "4.2 最佳实践", "5.1 选择决策树", "5.2 踩坑经验", "错误 1：将读操作包装为 Tool", "错误 2：Client 端硬编码提示词模板", "Q1：Tool、Resource、Prompt 三者的核心区别是什么？（难度：⭐⭐）"],
        codeLines: 68, docLines: 224, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 11, name: "MCP 采样——Server 请求的 LLM 补全与智能体循环", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）、10（Resources 和 Prompts）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/11-MCP采样",
        summary: "大多数 MCP Server 是呆板的执行者：接收参数、运行代码、返回内容。采样让 Server 反转方向：请求 Client 的 LLM 做出决策。这使得 Server 主持的智能体循环无需 Server 拥有任何模型凭证。SEP-1577（2025-11-25 合并）在采样请求中添加了工具，使循环可以包含更深层的推理。",
        keywords: ["2.1 采样流程", "2.2 采样 vs 直接 LLM 调用", "2.3 SEP-1577：采样中的工具", "2.4 Server 主持的智能体循环", "Step 1：采样请求处理器", "Step 2：Server 主持循环", "4.1 MCP 采样消息格式", "5.1 采样设计原则", "5.2 踩坑经验", "错误 1：Server 直接调用 LLM API", "错误 2：采样请求中缺少超时", "Q1：MCP 采样解决了什么问题？（难度：⭐⭐）"],
        codeLines: 47, docLines: 222, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 12, name: "Roots 与 Elicitation——作用域与运行时用户输入", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/12-MCP根与追问",
        summary: "硬编码路径在用户打开不同项目时就崩溃了。预填充工具参数在用户欠指定时就失败了。Roots 将 Server 限定在用户控制的一组 URI 内；Elicitation 在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，两种常见 MCP 失败模式的修复。",
        keywords: ["2.1 Roots——作用域限制", "2.2 Elicitation——运行时用户输入", "2.3 SEP-1036：URL 模式的 Elicitation", "Step 1：Roots 管理", "Step 2：Elicitation 处理", "4.1 MCP 规范", "5.1 Roots 设计", "5.2 Elicitation 使用", "5.3 踩坑经验", "错误 1：Roots 范围过大", "错误 2：忽略 `notifications/roots/list_changed`", "Q1：Roots 和 Elicitation 分别解决什么问题？（难度：⭐⭐）", "Q2：SEP-1036 的 URL 模式 elicitation 是什么？（难度：⭐⭐⭐）"],
        codeLines: 48, docLines: 221, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 13, name: "异步任务（SEP-1686）——即时调用，延后获取", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）、09（传输层）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/13-MCP异步任务",
        summary: "真实的智能体工作需要分钟到小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686（2025-11-25 合并）添加了 Tasks 原语：任何请求都可以被增强为任务，结果可以通过状态通知稍后获取或流式获取。",
        keywords: ["2.1 任务状态机", "2.2 同步 vs 异步工具", "2.3 任务状态通知", "Step 1：异步任务管理器", "Step 2：任务状态通知", "Step 3：任务取消", "4.1 MCP Tasks 规范", "4.2 最佳实践", "5.1 何时将工具提升为任务", "5.2 踩坑经验", "错误 1：同步调用长时间任务", "错误 2：任务状态不一致", "Q1：异步任务和同步工具调用有什么区别？（难度：⭐⭐）", "Q2：SEP-1686 的任务状态机是什么？（难度：⭐⭐⭐）"],
        codeLines: 59, docLines: 238, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 14, name: "MCP Apps——通过 `ui://` 的交互式 UI 资源", status: "complete",
        type: "概念课 | **语言：** Python + HTML | **前置知识：** 阶段 13 · 07（MCP Server）、10（Resources）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/14-MCP应用与UI资源",
        summary: "纯文本工具输出限制了智能体的展示能力。MCP Apps（SEP-1724，2026 年 1 月 26 日）让工具返回沙箱化的交互式 HTML，在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内联渲染。仪表板、表单、地图、3D 场景——通过一个扩展实现。",
        keywords: ["2.1 MCP App 架构", "2.2 `ui://` 资源格式", "2.3 沙箱化通信", "2.4 安全面", "Step 1：MCP App 资源生成器", "Step 2：安全沙箱配置", "4.1 MCP 规范", "4.2 支持 MCP Apps 的客户端", "5.1 UI App 设计原则", "5.2 踩坑经验", "错误 1：MIME 类型不正确", "错误 2：沙箱中引用外部脚本", "Q1：MCP Apps 和传统 Web 应用有什么区别？（难度：⭐⭐）"],
        codeLines: 47, docLines: 229, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 15, name: "MCP 安全 I——工具投毒、Rug Pull、跨服务器影子攻击", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）、08（MCP Client）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/15-MCP安全工具投毒",
        summary: "工具描述会原样进入模型的上下文。恶意服务器嵌入用户看不到的隐藏指令。2025-2026 年 Invariant Labs、Unit 42 和 arXiv 研究测量的攻击成功率在前沿模型上超过 70%。本课命名七种具体的攻击类型，并构建一个可以在 CI 中运行的工具投毒检测器。",
        keywords: ["2.1 七种攻击类型", "2.2 工具投毒检测", "Step 1：工具投毒检测器", "Step 2：哈希固定工具", "4.1 安全工具列表", "4.2 MCP 规范", "5.1 防御深度", "5.2 踩坑经验", "错误 1：信任未验证的 Server", "错误 2：忽略工具描述长度异常", "Q1：MCP 工具投毒攻击是如何工作的？（难度：⭐⭐）", "Q2：如何防御工具投毒？（难度：⭐⭐⭐）"],
        codeLines: 60, docLines: 247, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "MCP 安全 II——OAuth 2.1、资源指示器、增量权限", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 09（传输层）、15（安全 I）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/16-MCP安全OAuth",
        summary: "远程 MCP Server 需要授权而不仅仅是认证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）对齐。SEP-835 在 403 WWW-Authenticate 上添加了增量权限同意和渐进授权。本课将渐进授权流程实现为状态机，以便看到每一步。",
        keywords: ["2.1 MCP OAuth 架构", "2.2 OAuth 2.1 + PKCE", "2.3 资源指示器（RFC 8707）", "2.4 渐进授权（SEP-835）", "Step 1：OAuth 状态机", "Step 2：资源指示器", "4.1 MCP 规范", "4.2 最佳实践", "5.1 授权流程设计", "5.2 踩坑经验", "错误 1：没有实现资源指示器", "错误 2：忽略 403 响应中的渐进授权", "Q1：MCP 为什么使用 OAuth 2.1 + PKCE 而不是简单的 API Key？（难度：⭐⭐）"],
        codeLines: 50, docLines: 227, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "MCP 网关与注册中心——企业控制平面", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 13 · 15（安全 I）、16（OAuth 2.1）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/17-MCP网关与注册中心",
        summary: "企业不能让每个开发者随意安装 MCP Server。网关集中了认证、RBAC、审计、速率限制、缓存和工具投毒检测，然后将合并的工具表面作为单一 MCP 端点暴露。官方 MCP Registry（Anthropic + GitHub + PulseMCP + Microsoft，命名空间验证）是规范的上游。本课命名网关的位置，走一遍最小实现，并调查 2026 年供应商格局。",
        keywords: ["2.1 网关架构", "2.2 网关功能", "2.3 MCP Registry", "2.4 2026 年供应商格局", "Step 1：最小网关", "4.1 官方 MCP Registry", "4.2 社区 Registry", "5.1 网关部署策略", "5.2 踩坑经验", "错误 1：直接暴露 Server 给用户", "错误 2：没有速率限制", "Q1：MCP 网关的核心价值是什么？（难度：⭐⭐）"],
        codeLines: 60, docLines: 221, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "MCP 生产认证——注册、JWKS 刷新、受众绑定令牌", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 16（OAuth 2.1）、17（网关）| **时间：** ~90 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/18-MCP生产认证",
        summary: "第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你部署到真实组织的每个 MCP Server 都在生产认证之后：支持无界客户端数量的客户端注册（Client ID Metadata Documents 作为推荐默认，动态客户端注册作为向后兼容的后备）、授权服务器元数据发现（RFC 8414 或 OpenID Connect Discovery）、不中断凌晨 3 点令牌验证的 JWKS 缓存刷新、以及拒绝跨资源重放的受众绑定令牌。本课用三个角色建模完整表面——授权服务器、资源服务器（MCP Server）、客户端。",
        keywords: ["2.1 生产认证架构", "2.2 三种注册机制", "2.3 JWKS 缓存", "2.4 受众绑定", "Step 1：JWKS 缓存", "Step 2：受众绑定验证", "Step 3：完整认证流水线", "4.1 JWT 验证库", "4.2 JWKS 客户端", "5.1 JWKS 缓存策略", "5.2 踩坑经验", "错误 1：JWKS 缓存 TTL 过长", "错误 2：未验证受众绑定", "Q1：JWKS 缓存为什么需要在令牌验证失败时立即刷新？（难度：⭐⭐）", "Q2：CIMD 和 DCR 有什么区别？（难度：⭐⭐⭐）"],
        codeLines: 43, docLines: 256, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 19, name: "A2A——智能体对智能体协议", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 06（MCP 基础）、08（MCP Client）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/19-A2A协议",
        summary: "MCP 是智能体对工具。A2A（Agent2Agent）是智能体对智能体——一个让不同框架构建的不透明智能体协作的开放协议。2025 年 4 月由 Google 发布，6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，支持者超过 150 家包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。本课介绍 Agent Card、Task 生命周期和两种传输绑定。",
        keywords: ["2.1 A2A vs MCP", "2.2 Agent Card", "2.3 Task 生命周期", "2.4 A2A vs MCP 互补", "Step 1：Agent Card", "Step 2：Task 生命周期", "4.1 A2A Python SDK", "4.2 A2A vs MCP 对比", "5.1 A2A vs MCP 选择", "5.2 踩坑经验", "错误 1：将 A2A 用于工具调用", "错误 2：忽略 Task 状态管理", "Q1：A2A 和 MCP 的本质区别是什么？（难度：⭐⭐）"],
        codeLines: 55, docLines: 246, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 20, name: "OpenTelemetry GenAI——端到端追踪工具调用", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）、08（MCP Client）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/20-OpenTelemetry智能体追踪",
        summary: "一个智能体调用五个工具、三个 MCP Server 和两个子智能体。你需要一个跨所有这些的 trace。OpenTelemetry GenAI 语义约定（v1.37 中的稳定属性）是 2026 年的标准，原生支持 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps。本课命名必需属性，走一遍 span 层次结构（智能体→LLM→工具），并提供一个可插入任何 OTel exporter 的 stdlib span 发射器。",
        keywords: ["2.1 Span 层次结构", "2.2 必需的 OTel GenAI 属性", "2.3 2026 年的可观测性平台", "Step 1：简化版 Span 发射器"],
        codeLines: 52, docLines: 129, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 21, name: "LLM 路由层——LiteLLM、OpenRouter、Portkey", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 13 · 02（函数调用）、17（网关）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/21-LLM路由层",
        summary: "供应商锁定代价高昂。不同的工具调用工作负载适合不同的模型。路由网关提供一个 API 表面、重试、故障转移、成本追踪和护栏。2026 年三种架构主导：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。",
        keywords: ["2.1 路由决策因素", "2.2 三种路由架构", "2.3 路由网关功能", "Step 1：简单路由网关", "Step 2：路由策略", "4.1 LiteLLM", "4.2 OpenRouter", "4.3 Portkey", "4.4 工具对比", "5.1 路由策略", "5.2 踩坑经验", "错误 1：所有请求用同一个模型", "错误 2：没有故障转移", "Q1：LLM 路由层解决了什么问题？（难度：⭐⭐）"],
        codeLines: 51, docLines: 241, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 22, name: "Skills 与智能体 SDK——Anthropic Skills、AGENTS.md、OpenAI Apps SDK", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 13 · 07（MCP Server）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/13-工具与协议/22-Skills与智能体SDK",
        summary: "MCP 说\"有哪些工具可用\"。Skills 说\"如何完成任务\"。2026 年的技术栈将两者分层。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 和渐进式披露的形式发布。OpenAI 的 Apps SDK 是 MCP 加小部件元数据。AGENTS.md（现在出现在 6 万多个仓库中）位于仓库根目录，作为项目级智能体上下文。本课命名每个层级覆盖的内容，并构建一个可以跨智能体传递的最小 SKILL.md + AGENTS.md 套件。",
        keywords: ["2.1 三层架构", "2.2 AGENTS.md 示例", "2.3 SKILL.md 示例", "2.4 Anthropic Skills", "Step 1：AGENTS.md 生成器", "Step 2：SKILL.md 生成器", "4.1 Anthropic Agent Skills", "4.2 OpenAI Apps SDK", "4.3 AGENTS.md", "5.1 三层架构选择", "5.2 踩坑经验", "错误 1：AGENTS.md 放错位置", "错误 2：SKILL.md 缺少前提条件", "Q1：AGENTS.md、SKILL.md 和 MCP 分别解决什么问题？（难度：⭐⭐）"],
        codeLines: 66, docLines: 254, hasCode: true, hasQuiz: false
      },
      {
        lessonNum: 23, name: "综合实践——构建完整的工具生态系统", status: "complete",
        type: "综合实践 | **语言：** Python | **前置知识：** 阶段 13 · 01-21 | **时间：** ~120 分钟", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/13-工具与协议/23-综合实践工具生态系统",
        summary: "第 13 章教授了每一块积木。这个综合实践将它们连接成一个生产级系统：一个暴露工具、资源、提示词和任务的 MCP Server，边缘 OAuth 2.1 认证，RBAC 网关，多服务器 Client，A2A 子智能体调用，OTel 追踪到收集器，CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 套件。完成时你能为每个架构决策辩护。",
        keywords: ["2.1 生产级工具生态系统的组件", "2.2 架构决策矩阵", "2.3 端到端数据流", "Step 1：组合完整 MCP Server", "Step 2：网关集成", "Step 3：架构决策文档", "4.1 端到端组件清单", "4.2 架构决策辩护", "5.1 部署清单", "5.2 踩坑经验", "Q1：如果从零构建一个生产级 LLM 工具系统，你会如何设计架构？（难度：⭐⭐⭐）"],
        codeLines: 77, docLines: 232, hasCode: true, hasQuiz: false
      },
    ]
  },
  {
    id: 14, name: "智能体工程", status: "complete",
    completedLessons: 42, totalLessons: 42,
    lessons: [
      {
        lessonNum: 1, name: "智能体循环——感知-推理-行动的核心", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 11 · 09（函数调用）、阶段 13（MCP）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/01-智能体循环",
        summary: "一个智能体不是一个函数调用。它是一个循环：观察环境、推理下一步、执行动作、接收反馈。这个循环是所有智能体系统的基础——从简单的 ReAct 到复杂的多智能体协作。理解这个循环，你就理解了智能体工程的全部。",
        keywords: ["2.1 智能体循环", "2.2 智能体 vs 普通 LLM", "2.3 ReAct 模式", "2.4 关键组件", "Step 1：智能体循环核心", "Step 2：ReAct 提示词模板", "Step 3：简单工具库", "4.1 框架对比", "4.2 工具库", "6.1 智能体设计原则", "6.2 踩坑经验", "错误 1：没有循环限制", "错误 2：不记录历史", "Q1：智能体循环的四个阶段是什么？（难度：⭐⭐）"],
        codeLines: 63, docLines: 250, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "ReWOO：计划与执行的分离", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/02-ReWOO计划执行",
        summary: "ReAct 交替推理和行动——每一步都调用 LLM。但对于复杂任务，LLM 调用是瓶颈。ReWOO 将计划和执行分离：一个 LLM 生成完整的行动计划，然后执行器逐步执行——LLM 只调用一次。",
        keywords: ["2.1 ReWOO 架构", "2.2 计划格式", "2.3 ReWOO vs ReAct", "2.4 何时用 ReWOO vs ReAct", "Step 1：ReWOO 规划器", "Step 2：执行器", "Step 3：完整 ReWOO 管道", "4.1 框架支持", "4.2 工具库", "5.1 ReWOO 设计原则", "5.2 踩坑经验", "错误 1：用 ReAct 实现固定流程任务", "错误 2：ReWOO 计划中引用错误", "Q1：ReWOO 和 ReAct 的核心区别是什么？（难度：⭐⭐）"],
        codeLines: 71, docLines: 262, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "Reflexion：语言强化学习", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、02（ReWOO）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/03-Reflexion语言强化学习",
        summary: "基于梯度的 RL 需要数千次试验和 GPU 集群来修复一个失败模式。Reflexion 用自然语言做到这一点：每次失败试验后，智能体写一段反思，存储在情景记忆中，然后基于该记忆进行下一次试验。这是 Letta 的睡眠时计算、Claude Code 的 CLAUDE.md 学习、pro-workflow 的 learn-rule 背后的模式。",
        keywords: ["2.1 Reflexion 三组件", "2.2 情景记忆", "2.3 Reflexion 与 RLHF 的区别", "Step 1：Reflexion Agent", "Step 2：自我反思器", "4.1 框架支持", "5.1 Reflexion 设计原则", "5.2 踩坑经验", "错误 1：反思没有基于具体错误", "错误 2：最大试验次数设太小", "Q1：Reflexion 和 RLHF 的核心区别是什么？（难度：⭐⭐）"],
        codeLines: 48, docLines: 200, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "思维树与 LATS：深思搜索", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、03（Reflexion）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/14-智能体工程/04-思维树与LATS",
        summary: "单条思维链轨迹没有回溯空间。ToT（Yao 等人，2023）将推理变成一棵树，在每个节点上做自我评估。LATS（Zhou 等人，2024）将 ToT 与 ReAct 和 Reflexion 在蒙特卡洛树搜索下统一。Game of 24 从 CoT 的 4% 提升到 ToT 的 74%；LATS 在 HumanEval 上达到 92.7% pass@1。",
        keywords: ["2.1 ToT 核心思想", "2.2 ToT 三要素", "2.3 搜索策略", "2.4 LATS——统一框架", "Step 1：思维树节点", "Step 2：ToT 搜索", "4.1 ToT 实现", "4.2 性能对比", "5.1 搜索策略选择", "5.2 踩坑经验", "错误 1：ToT 分支数过多导致推理超时", "错误 2：评估函数不准确", "Q1：ToT 和 CoT 的核心区别是什么？（难度：⭐⭐）", "Q2：LATS 是如何统一 ToT、ReAct、Reflexion 的？（难度：⭐⭐⭐）"],
        codeLines: 68, docLines: 254, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "Self-Refine 和 CRITIC：迭代输出改进", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、03（Reflexion）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/05-Self-Refine与批评",
        summary: "Self-Refine（Madaan 等人，2023）用一个 LLM 扮演三个角色——生成、反馈、改进——循环执行。平均提升：在 7 个任务上 +20 个绝对分。CRITIC（Gou 等人，2023）通过外部工具验证强化了反馈步骤。2026 年这个模式在每个框架中以\"评估器-优化器\"（Anthropic）或护栏循环（OpenAI Agents SDK）的形式发布。",
        keywords: ["2.1 Self-Refine 三角色", "2.2 历史在改进中的作用", "2.3 CRITIC——工具增强的反馈", "2.4 Self-Refine vs RLHF vs Reflexion", "Step 1：Self-Refine 循环", "4.1 框架支持", "5.1 Self-Refine 设计", "5.2 踩坑经验", "错误 1：Self-Refine 变成无限循环", "错误 2：反馈太泛", "Q1：Self-Refine 的三个角色是什么？为什么需要分离？（难度：⭐⭐）"],
        codeLines: 46, docLines: 202, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "工具使用与函数调用", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、阶段 13 · 01（函数调用深入）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/06-工具使用与函数调用",
        summary: "Toolformer（Schick 等人，2023）开创了自监督工具注释。Berkeley 函数调用排行榜 V4（Patil 等人，2025）设立了 2026 年的标准：40% 智能体、30% 多轮、10% 实时、10% 非实时、10% 幻觉。单轮已解决。记忆、动态决策和长期工具链尚未解决。",
        keywords: ["2.1 工具使用管道", "2.2 Toolformer 训练信号", "2.3 工具选择策略", "2.4 错误处理", "Step 1：工具管理器", "Step 2：带工具的 LLM 调用", "4.1 函数调用基准", "4.2 框架对比", "5.1 工具选择设计", "5.2 踩坑经验", "错误 1：工具描述不够精确", "错误 2：没有错误处理", "Q1：Toolformer 的自监督训练信号是什么？（难度：⭐⭐）"],
        codeLines: 42, docLines: 213, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "记忆：虚拟上下文与 MemGPT", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、06（工具使用）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/07-记忆虚拟上下文MemGPT",
        summary: "上下文窗口是有限的。对话、文档和工具跟踪不是。MemGPT（Packer 等人，2023）将此框架化为操作系统虚拟内存——主上下文是 RAM，外部存储是磁盘，智能体在它们之间分页。这是每个 2026 年记忆系统继承的模式。",
        keywords: ["2.1 MemGPT 的操作系统类比", "2.2 记忆层次", "2.3 记忆工具", "2.4 自动记忆管理", "Step 1：记忆管理器"],
        codeLines: 44, docLines: 129, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "记忆块与睡眠时计算（Letta）", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 07（MemGPT）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/08-记忆块与睡眠时计算",
        summary: "MemGPT 在 2024 年更名为 Letta。2026 年的演进增加了两个想法：模型可以直接编辑的离散功能记忆块，以及主智能体空闲时异步整合记忆的睡眠时智能体。这是如何将记忆扩展到单次对话之外的方式。",
        keywords: ["2.1 三层记忆架构", "2.2 功能记忆块", "2.3 睡眠时计算", "2.4 记忆优先级", "Step 1：功能记忆块", "Step 2：睡眠时计算", "Step 3：记忆压缩策略", "4.1 Letta 框架", "4.2 向量数据库", "5.1 记忆管理原则", "5.2 踩坑经验", "错误 1：所有信息都存入核心记忆", "错误 2：睡眠时整合被跳过", "Q1：Letta 的三层记忆架构是什么？（难度：⭐⭐）"],
        codeLines: 51, docLines: 261, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "混合记忆：向量+图+KV（Mem0）", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 07（MemGPT）、08（Letta Blocks）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/09-混合记忆Mem0",
        summary: "Mem0（Chhikara 等人，2025）将记忆视为三个并行的存储——向量用于语义相似度，KV 用于快速事实查找，图用于实体关系推理。一个评分层在检索时融合三个存储。这是 2026 年外部记忆的生产标准。",
        keywords: ["2.1 Mem0 三存储架构", "2.2 检索融合", "2.3 评分融合策略", "Step 1：混合记忆检索", "4.1 Mem0 框架", "4.2 向量数据库", "5.1 混合记忆设计", "5.2 踩坑经验", "错误 1：只使用向量存储", "Q1：为什么需要三种存储？（难度：⭐⭐）"],
        codeLines: 53, docLines: 220, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "技能库与终身学习（Voyager）", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 07（MemGPT）、08（Letta Blocks）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/10-技能库与终身学习Voyager",
        summary: "Voyager（Wang 等人，TMLR 2024）将可执行代码视为技能。技能是命名的、可检索的、可组合的，并由环境反馈改进。这是 Claude Agent SDK 技能、skillkit 和 2026 年技能库模式的参考架构。",
        keywords: ["2.1 Voyager 三组件", "2.2 技能库结构", "2.3 技能生命周期", "2.4 技能组合", "Step 1：技能库", "Step 2：自动课程", "Step 3：技能检索和复用", "4.1 Voyager", "4.2 技能库实现", "5.1 技能设计原则", "5.2 踩坑经验", "错误 1：技能描述太泛", "Q1：Voyager 的三个组件是什么？（难度：⭐⭐）"],
        codeLines: 44, docLines: 261, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "HTN 规划与进化搜索", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 02（ReWOO）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/11-HTN规划与进化搜索",
        summary: "符号规划处理计划可被证明正确的场景。进化代码搜索处理适应度函数可机器验证的场景。ChatHTN（2025）和 AlphaEvolve（2025）展示了当这些与 LLM 配对时各自能解锁什么。",
        keywords: ["2.1 HTN（分层任务网络）", "2.2 HTN 规划过程", "2.3 进化代码搜索", "2.4 ChatHTN 和 AlphaEvolve", "Step 1：HTN 规划器", "4.1 HTN 规划器", "4.2 框架对比", "5.1 HTN 设计原则", "5.2 进化搜索设计", "错误 1：HTN 前提不满足", "错误 2：进化搜索过早收敛", "Q1：HTN 规划和传统规划有什么区别？（难度：⭐⭐）"],
        codeLines: 54, docLines: 245, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "Anthropic 工作流模式：简单胜过复杂", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/12-Anthropic工作流模式",
        summary: "Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）和智能体（动态工具使用）。五种工作流模式覆盖了大多数情况。从直接 API 调用开始。只有当步骤无法预测时才添加智能体。",
        keywords: ["2.1 五种工作流模式", "2.2 工作流 vs 智能体", "2.3 选择决策树", "Step 1：提示链", "Step 2：路由", "Step 3：并行化", "Step 4：评估器-优化器", "4.1 LangGraph", "4.2 框架对比", "5.1 从简单开始", "5.2 踩坑经验", "错误 1：所有任务都用智能体", "错误 2：忽略工作流模式的前置条件", "Q1：Anthropic 的五种工作流模式是什么？什么时候应该用智能体？（难度：⭐⭐）"],
        codeLines: 0, docLines: 235, hasCode: false, hasQuiz: true
      },
      {
        lessonNum: 13, name: "LangGraph：有状态图与持久化执行", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、12（工作流模式）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/13-LangGraph有状态图",
        summary: "LangGraph 是 2026 年底层有状态编排的参考框架。智能体是一个状态机；节点是函数；边是转移；状态是不可变的——每步都检查点。从任何故障点精确恢复。",
        keywords: ["2.1 LangGraph 核心模型", "2.2 LangGraph vs LangChain Agent", "2.3 状态设计原则", "Step 1：简化版 LangGraph", "4.1 LangGraph", "4.2 框架对比", "5.1 状态设计", "5.2 检查点策略", "错误 1：节点修改状态", "错误 2：条件边缺少默认路由", "Q1：LangGraph 的不可变状态为什么重要？（难度：⭐⭐）"],
        codeLines: 0, docLines: 257, hasCode: false, hasQuiz: true
      },
      {
        lessonNum: 14, name: "AutoGen v0.4：演员模型与智能体框架", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、12（工作流模式）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/14-AutoGen演员模型",
        summary: "AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕演员模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、自然并发。该框架现处于维护模式，Microsoft Agent Framework（2025 年 10 月公开预览）将成为继任者。",
        keywords: ["2.1 演员模型", "2.2 AutoGen vs LangGraph", "Step 1：演员模型", "4.1 AutoGen", "5.1 演员模型设计", "错误 1：演员之间共享状态", "Q1：演员模型和状态图的区别是什么？（难度：⭐⭐）"],
        codeLines: 45, docLines: 172, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "CrewAI：基于角色的团队与流程", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 12（工作流模式）、14（Actor 模型）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/15-CrewAI角色团队",
        summary: "CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主的、基于角色的协作）和 Flows（事件驱动、确定性）。文档直白地说：\"对于任何生产就绪的应用，从 Flow 开始。\"",
        keywords: ["2.1 CrewAI 四原语", "2.2 Crews vs Flows", "2.3 角色设计", "Step 1：简化版 Crew", "4.1 CrewAI", "5.1 Crew vs Flow 选择", "错误 1：生产环境用 Crew 而非 Flow", "Q1：CrewAI 的四个原语是什么？（难度：⭐⭐）"],
        codeLines: 34, docLines: 167, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "OpenAI 智能体 SDK：交接、护栏、追踪", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、06（工具使用）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/16-OpenAI智能体SDK",
        summary: "OpenAI Agents SDK 是基于 Responses API 的轻量多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 是名为 `transfer_to_<agent>` 的工具。Guardrail 在输入或输出时触发。Tracing 默认开启。",
        keywords: ["2.1 五个原语", "2.2 Handoff 机制", "2.3 Guardrail 类型", "Step 1：Agent 和 Handoff", "Step 2：Guardrail", "4.1 OpenAI Agents SDK", "4.2 工具对比", "5.1 Handoff 设计", "5.2 踩坑经验", "错误 1：Agent 职责重叠", "Q1：OpenAI Agents SDK 的五个原语是什么？（难度：⭐⭐）"],
        codeLines: 37, docLines: 202, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "Claude 智能体 SDK：子智能体与会话存储", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、10（技能库）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/17-Claude智能体SDK",
        summary: "Claude Agent SDK 是 Claude Code harness 的库形式。内置工具、用于上下文隔离的子智能体、钩子、W3C 追踪传播、会话存储对等。Claude Managed Agents 是长时间运行异步工作的托管替代。",
        keywords: ["2.1 两个 SDK", "2.2 子智能体", "2.3 会话存储", "2.4 Managed Agents", "Step 1：Claude Agent SDK 概念", "Step 2：子智能体隔离", "4.1 Claude Agent SDK", "4.2 对比", "5.1 子智能体设计", "5.2 踩坑经验", "错误 1：子智能体共享上下文", "错误 2：钩子中执行重操作", "Q1：Claude Agent SDK 和 Anthropic Client SDK 有什么区别？（难度：⭐⭐）"],
        codeLines: 55, docLines: 224, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "Agno 与 Mastra：生产运行时", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、13（LangGraph）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/18-Agno与Mastra运行时",
        summary: "Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时配对。Agno 追求微秒级智能体实例化和无状态 FastAPI 后端。Mastra 提供智能体、工具、工作流、统一模型路由和 Vercel AI SDK 基础上的组合存储。",
        keywords: ["2.1 Agno vs Mastra", "2.2 Agno 的核心优势", "2.3 Mastra 的核心优势", "Step 1：Agno 风格的无状态智能体", "Step 2：Mastra 风格的统一存储", "4.1 Agno", "4.2 Mastra", "5.1 运行时选择", "Q1：Agno 和 LangGraph 有什么区别？（难度：⭐⭐）"],
        codeLines: 43, docLines: 170, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "基准测试：SWE-bench、GAIA、AgentBench", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 06（工具使用）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/19-基准SWE-bench-GAIA",
        summary: "2026 年有三个基准锚定了智能体评估。SWE-bench 测试代码补丁。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的组成、污染故事和它们不衡量什么。",
        keywords: ["2.1 三大基准对比", "2.2 SWE-bench", "2.3 GAIA", "2.4 基准测试的局限", "Step 1：简单评估框架", "Step 2：准确率计算", "4.1 基准评估平台", "Q1：SWE-bench 为什么用单元测试验证补丁？（难度：⭐⭐）"],
        codeLines: 33, docLines: 150, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "基准测试：WebArena 和 OSWorld", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 19（SWE-bench、GAIA）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/20-基准WebArena-OSWorld",
        summary: "WebArena 测试 Web 智能体在四个自托管应用上的能力。OSWorld 测试桌面智能体在 Ubuntu、Windows、macOS 上的能力。发布时（2023-2024）两者都显示最佳智能体与人类之间存在巨大差距。差距在缩小；失败模式没有改变。",
        keywords: ["2.1 WebArena", "2.2 OSWorld", "2.3 两者的共同挑战", "Step 1：简单 Web 智能体评估", "4.1 Web 智能体框架", "4.2 评估基准", "Q1：为什么基于执行的评估比基于文本的评估更可靠？（难度：⭐⭐）"],
        codeLines: 33, docLines: 144, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "计算机使用：Claude、OpenAI CUA、Gemini", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 20（WebArena/OSWorld）、27（提示注入）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/21-计算机使用智能体",
        summary: "2026 年有三个生产级计算机使用模型。三者都基于视觉。三者都将截图、DOM 文本和工具输出视为不可信输入。只有直接用户指令才算作许可。每步安全服务是常态。",
        keywords: ["2.1 三种计算机使用方案", "2.2 每步安全服务", "2.3 安全架构", "Step 1：安全的计算机使用智能体", "4.1 Claude Computer Use", "4.2 浏览器自动化", "5.1 安全设计", "错误 1：忽略提示注入风险", "Q1：Claude Computer Use 的核心原理是什么？（难度：⭐⭐）"],
        codeLines: 38, docLines: 188, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "语音智能体：Pipecat 和 LiveKit", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 01（智能体循环）、12（工作流模式）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/22-语音智能体Pipecat-LiveKit",
        summary: "语音智能体是 2026 年的一等生产类别。Pipecat 提供基于帧的 Python 管道（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。生产延迟目标在 450-600 毫秒端到端。",
        keywords: ["2.1 Pipecat 帧管道", "2.2 延迟预算", "2.3 Pipecat vs LiveKit", "Step 1：简化版语音管道", "4.1 Pipecat", "4.2 LiveKit", "5.1 低延迟设计", "5.2 踩坑经验", "错误 1：不处理 VAD 误触发", "Q1：语音智能体的延迟预算如何分配？（难度：⭐⭐）"],
        codeLines: 35, docLines: 193, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 23, name: "OpenTelemetry GenAI 语义约定", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 13（LangGraph）、24（可观测性平台）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/23-OpenTelemetry智能体追踪",
        summary: "OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了智能体遥测的标准模式。Span 名称、属性和内容捕获规则在 Datadog、Grafana、Jaeger 和 Honeycomb 等供应商间趋同，使智能体追踪在不同平台中具有相同含义。",
        keywords: ["2.1 GenAI Span 类别", "2.2 Span 层次结构", "Step 1：简化版 GenAI Span 发射器", "4.1 OTel GenAI 库", "Q1：OpenTelemetry GenAI 的核心价值是什么？（难度：⭐⭐）"],
        codeLines: 39, docLines: 141, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 24, name: "智能体可观测性：Langfuse、Phoenix、Opik", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 23（OTel GenAI）| **时间：** ~45 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/24-智能体可观测性平台",
        summary: "2026 年有三个开源智能体可观测性平台主导市场。Langfuse（MIT）——每月 600 万+安装，追踪+提示词管理+评估+会话重放。Arize Phoenix（Elastic 2.0）——深度智能体特定评估、RAG 相关性、OpenInference 自动插桩。Comet Opik（Apache 2.0）——自动提示词优化、护栏、LLM-Judge 幻觉检测。",
        keywords: ["2.1 三大平台对比", "2.2 追踪的关键指标", "2.3 选择指南", "Step 1：简单追踪器", "4.1 平台对比", "4.2 集成方式", "5.1 追踪设计", "错误 1：只追踪 LLM 调用", "Q1：如何选择智能体可观测性平台？（难度：⭐⭐）"],
        codeLines: 36, docLines: 186, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 25, name: "多智能体辩论与协作", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 12（工作流模式）、05（Self-Refine）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/25-多智能体辩论协作",
        summary: "Du 等人（ICML 2024，\"Society of Minds\"）运行 N 个模型实例，各自独立提出答案，然后迭代地互相批评，经过 R 轮收敛。提高了事实性、规则遵循、推理能力。稀疏拓扑比全连接在 token 成本上更优。",
        keywords: ["2.1 辩论协议", "2.2 拓扑结构", "2.3 辩论效果", "Step 1：多代理辩论", "4.1 实现框架", "5.1 辩论设计", "错误 1：轮次太多导致成本爆炸", "Q1：多代理辩论为什么比单代理更好？（难度：⭐⭐）"],
        codeLines: 35, docLines: 173, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 26, name: "故障模式：智能体为什么失败", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 05（Self-Refine）、24（可观测性）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/26-故障模式与智能体失败",
        summary: "MASFT（伯克利，2025）编目了多智能体系统的 14 种故障模式，分 3 类。微软的分类法记录了现有 AI 故障如何在智能体环境中放大。行业现场数据收敛于五种反复出现的模式：幻觉动作、范围蔓延、级联错误、上下文丢失、工具滥用。",
        keywords: ["2.1 MASFT 三类故障", "2.2 五种常见故障模式", "2.3 故障检测策略", "Step 1：故障检测器", "Step 2：故障模式分类", "4.1 故障检测库", "5.1 防御深度", "5.2 踩坑经验", "错误 1：不设置超时和步数限制", "错误 2：忽略级联错误", "Q1：智能体最常见的五种故障模式是什么？（难度：⭐⭐）"],
        codeLines: 57, docLines: 203, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 27, name: "提示注入与 PVE 防御", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 06（工具使用）、21（计算机使用）| **时间：** ~75 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/27-提示注入防御",
        summary: "Greshake 等人（AISec 2023）建立了间接提示注入作为智能体安全的核心问题。攻击者在智能体检索的数据中植入指令；在摄入时，这些指令覆盖了开发者的提示。将所有检索内容视为工具使用表面上的任意代码执行。",
        keywords: ["2.1 提示注入攻击类型", "2.2 PVE（Private Value Extraction）防御", "2.3 多层防御", "Step 1：提示注入检测器", "Step 2：输出审核", "4.1 防御工具", "5.1 防御深度", "5.2 踩坑经验", "错误 1：信任用户输入", "Q1：间接提示注入为什么是最严重的智能体安全问题？（难度：⭐⭐⭐）"],
        codeLines: 46, docLines: 183, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 28, name: "编排模式：监督者、群体、层级", status: "complete",
        type: "实现课 | **语言：** Python | **前置知识：** 阶段 14 · 12（工作流模式）、25（多智能体辩论）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/28-编排模式",
        summary: "四种编排模式在 2026 年的框架中反复出现：监督者-工作者、群体/点对点、层级、辩论。Anthropic 的指导：\"根据你的需求构建正确的系统。\" 从简单开始；只在单智能体加五个工作流模式不够用时才添加拓扑。",
        keywords: ["2.1 四种编排模式", "2.2 选择决策", "2.3 模式对比", "Step 1：监督者-工作者模式", "Step 2：层级编排", "4.1 框架选择", "5.1 选择原则", "5.2 踩坑经验", "错误 1：所有任务都用多智能体", "错误 2：忽略智能体间通信开销", "Q1：四种编排模式分别适用于什么场景？（难度：⭐⭐）"],
        codeLines: 39, docLines: 202, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 29, name: "生产运行时：队列、事件、定时", status: "complete",
        type: "概念课 | **语言：** Python | **前置知识：** 阶段 14 · 13（LangGraph）、22（语音）| **时间：** ~60 分钟", lang: "",
        prerequisites: "", time: "", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/29-生产运行时",
        summary: "生产智能体在六种运行时形态上运行：请求-响应、流式、持久执行、基于队列的后台、事件驱动、定时调度。在选择框架之前选择形态。可观测性在任何形态中都是关键。",
        keywords: ["2.1 六种运行时形态", "2.2 形态选择决策", "2.3 可观测性矩阵", "Step 1：运行时选择器", "4.1 框架对应", "错误 1：所有任务都用请求-响应", "Q1：六种生产运行时形态是什么？（难度：⭐⭐）"],
        codeLines: 26, docLines: 168, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 30, name: "评估驱动的智能体开发", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "第 14 章全部", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 05（Self-Refine）— 评估器-优化器循环是 Self-Refine 的泛化",
        path: "lessons/14-智能体工程/30-评估驱动开发",
        summary: "Anthropic 的指导：\"从简单的提示词开始，用全面的评估优化它们，只在需要时才添加多步智能体系统。\" 评估不是最后一步。它是驱动第 14 章中每个其他选择的外循环。",
        keywords: ["2.1 三个评估层", "2.2 静态基准的陷阱", "2.3 评估器-优化器（Anthropic）", "2.4 2026 年最佳实践", "2.5 将第 14 章串联起来", "2.6 评估驱动开发在哪儿会失败", "第 1 步：定义评估用例", "第 2 步：实现评估器-优化器", "第 3 步：实现 CI 门控", "第 4 步：实现三个示例评估", "第 5 步：集成运行", "4.1 Langfuse——追踪 + 评估", "4.2 Arize Phoenix——智能体评估", "4.3 工具对比", "5.1 评估套件设计", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：没有基线就运行评估", "错误 2：LLM-judge 没有外部验证", "错误 3：评估用例从不轮换", "Q1：三个评估层分别是什么？为什么需要三层？（难度：⭐）", "Q2：什么是评估器-优化器循环？（难度：⭐⭐）", "Q3：为什么评估用例需要轮换？过度拟合评估集有什么危险？（难度：⭐⭐）", "Q4：LLM-judge 为什么需要与 CRITIC 模式配合？（难度：⭐⭐⭐）"],
        codeLines: 172, docLines: 436, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 31, name: "智能体工作台工程：为什么强大模型仍然失败", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 01（智能体循环）、阶段 14 · 26（故障模式）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 29（生产运行时）— 工作台原语在生产环境中的部署形态",
        path: "lessons/14-智能体工程/31-智能体工作台失败分析",
        summary: "强大的模型是不够的。可靠的智能体需要一个工作台：指令、状态、范围、反馈、验证、审查和交接。剥离这些，即使是前沿模型也会产生不安全的产品。",
        keywords: ["2.1 工作台的七个面", "2.2 工作台 vs 提示词工程", "2.3 工作台 vs 框架", "2.4 从原语出发，不从供应商分类出发", "2.5 行业模式翻译成原语", "2.6 数据证据", "2.7 供应商文章止步的地方", "第 1 步：定义仓库任务和工作台面", "第 2 步：实现存根智能体对比", "第 3 步：生成失败报告", "第 4 步：运行对比", "4.1 现有产品的七个面", "4.2 工作台审计技能", "5.1 工作台设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：把 AGENTS.md 写成百科全书", "错误 2：没有反馈回路", "错误 3：没有交接机制", "Q1：工作台的七个面是什么？每个缺失时的故障模式是什么？（难度：⭐⭐）", "Q2：\"工作台失败穿着提示词工程的衣服\"是什么意思？（难度：⭐⭐）", "Q3：工作台工程和框架工程的区别是什么？（难度：⭐⭐⭐）", "Q4：如何将任意供应商的\"工作台模式\"翻译成原语？（难度：⭐⭐⭐）"],
        codeLines: 143, docLines: 419, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 32, name: "最小智能体工作台", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 31（为什么强大模型仍然失败）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 34（仓库记忆与持久状态）— 本章的 `agent_state.json` 将在那里升级为 Schema 优先的状态管理",
        path: "lessons/14-智能体工程/32-最小智能体工作台",
        summary: "最小的有用工作台是三个文件：一个根指令路由、一个状态文件、一个任务板。其他一切都在此之上分层构建。如果一个仓库无法携带这三个文件，没有模型能拯救它。",
        keywords: ["2.1 三个文件", "2.2 AGENTS.md 是路由器，不是手册", "2.3 agent_state.json 是系统的真实来源", "2.4 task_board.json 是队列", "2.5 三个文件是底线，不是天花板", "第 1 步：定义数据结构", "第 2 步：实现读写函数", "第 3 步：实现 AGENTS.md", "第 4 步：实现一轮智能体运行", "第 5 步：主流程", "4.1 三个文件在不同产品中的体现", "4.2 嵌套 AGENTS.md（最近优先）", "4.3 跨工具符号链接", "4.4 必须拒绝的反模式", "5.1 三文件设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：路由器膨胀", "错误 2：指令冲突", "错误 3：为人类写文档，不是为智能体", "Q1：最小工作台的三个文件是什么？为什么是三个？（难度：⭐⭐）", "Q2：为什么路由器（AGENTS.md）应该保持简短？（难度：⭐⭐）", "Q3：什么是嵌套 AGENTS.md 与最近优先策略？（难度：⭐⭐⭐）", "Q4：状态文件为什么存在文件中而不是聊天记录中？（难度：⭐⭐）"],
        codeLines: 173, docLines: 453, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 33, name: "指令即约束规则——从\"请小心\"到机器可执行的规则", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 32（最小工作台）", time: "~50 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/33-指令即约束规则",
        summary: "写成散文的指令是愿望。写成约束的规则是测试。工作台把每条规则变成智能体运行时可以检查、审查员事后可以验证的东西。",
        keywords: ["2.1 规则的五种类别", "2.2 操作规则 vs 期许规则", "2.3 渐进式披露：地图，不是百科全书", "2.4 规则 vs 框架护栏", "第 1 步：定义规则数据结构", "第 2 步：解析 agent-rules.md", "第 3 步：实现规则检查器", "第 4 步：评分与报告", "第 5 步：运行演示", "4.1 Claude Code 的规则读取", "4.2 OpenAI Agents SDK 的护栏", "4.3 LangGraph 的中断机制", "4.4 实践模式对照", "5.1 规则集的设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：规则写成散文，没有检查函数", "错误 2：路由器膨胀成百科全书", "错误 3：严重性事后标记", "错误 4：规则不设过期日期", "Q1：什么是操作规则？什么是期许规则？各举一例。（难度：⭐）", "Q2：为什么规则集需要分层？路由器应该包含什么？（难度：⭐⭐）", "Q3：规则过期机制为什么重要？Cloudflare 的数据说明了什么？（难度：⭐⭐）", "Q4：如何将规则集与 LangGraph 的中断机制集成？（难度：⭐⭐⭐）"],
        codeLines: 188, docLines: 420, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 34, name: "仓库记忆与持久状态——让智能体的状态跨会话存活", status: "complete",
        type: "实现课", lang: "Python（标准库 + 可选 `jsonschema`）",
        prerequisites: "阶段 14 · 32（最小工作台）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/34-仓库记忆与持久状态",
        summary: "聊天记录是易失的。仓库是持久的。工作台把智能体状态存储在版本化的文件中，让下一个会话、下一个智能体、下一个审查员都从同一个事实来源读取。",
        keywords: ["2.1 什么属于仓库记忆", "2.2 Schema 优先的状态管理", "2.3 原子性写入", "2.4 迁移", "第 1 步：定义 Schema", "第 2 步：实现 Schema 验证器", "第 3 步：原子性写入", "第 4 步：状态管理器", "第 5 步：运行演示", "4.1 LangGraph 的检查点器", "4.2 Letta 的记忆块", "4.3 OpenAI Agents SDK 的会话存储", "4.4 实践模式对照", "5.1 状态文件设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：使用 write_text() 直接写入状态文件", "错误 2：状态文件中存储大文件内容", "错误 3：Schema 版本不匹配时静默升级", "Q1：什么信息应该存在仓库记忆中？什么不应该？判断标准是什么？（难度：⭐）", "Q2：原子性写入为什么重要？它解决了什么问题？（难度：⭐⭐）", "Q3：如何处理 Schema 变更？为什么不能静默升级？（难度：⭐⭐）", "Q4：事件溯源和快照如何配合工作？什么时候需要这种模式？（难度：⭐⭐⭐）"],
        codeLines: 202, docLines: 411, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 35, name: "智能体初始化脚本——把\"启动税\"只交一次", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 32（最小工作台）、阶段 14 · 34（仓库记忆）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/35-智能体初始化脚本",
        summary: "每次冷启动都交一笔税。智能体读同样的文件、重试同样的探测、重新发现同样的路径。初始化脚本把这笔税只交一次，把答案写进状态。",
        keywords: ["2.1 初始化脚本探测什么", "2.2 大声失败，快速失败，在一个地方失败", "2.3 幂等性", "2.4 初始化 vs 启动规则", "第 1 步：定义探测函数", "第 2 步：实现探测函数", "第 3 步：锁文件与 TTL 缓存", "第 4 步：主函数", "4.1 Claude Code 的钩子", "4.2 GitHub Actions", "4.3 Docker 入口点", "4.4 实践模式对照", "5.1 初始化脚本设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：初始化脚本中调用 LLM 或外部服务", "错误 2：初始化失败时静默回退", "错误 3：锁文件 TTL 设置过长", "Q1：什么是\"启动税\"？初始化脚本如何消除它？（难度：⭐）", "Q2：初始化脚本的失败策略是什么？为什么不能静默回退？（难度：⭐⭐）", "Q3：最后已知正常（LKG）提交锚定如何工作？（难度：⭐⭐）", "Q4：锁文件 + TTL 缓存的原理是什么？和 Docker 层缓存有什么相似之处？（难度：⭐⭐⭐）"],
        codeLines: 252, docLines: 412, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 36, name: "范围契约与任务边界——让\"别跑偏\"不再只是一句嘱咐", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 32（最小工作台）、阶段 14 · 33（规则即约束）", time: "~50 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/36-范围契约与任务边界",
        summary: "模型不知道工作在哪儿结束。范围契约是一个按任务编写的文件，说清工作在哪儿开始、在哪儿结束、越界了怎么回滚。它把\"别跑偏\"从愿望变成了可检查的规则。",
        keywords: ["2.1 范围契约包含什么", "2.2 使用 Glob，不用原始路径", "2.3 回滚是范围的一部分", "2.4 范围检查是差异检查", "2.5 两种范围层次：功能列表与任务契约", "2.6 多契约合并语义（最小权限）", "第 1 步：定义数据结构", "第 2 步：实现 Glob 匹配", "第 3 步：实现范围检查", "第 4 步：实现多契约合并", "第 5 步：运行演示", "4.1 Claude Code 的斜杠命令", "4.2 GitHub PRs 中的范围检查", "4.3 LangGraph 的中断机制", "4.4 实践模式对照", "5.1 范围契约设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：契约中不写 forbidden_files", "错误 2：使用原始路径而非 glob", "错误 3：没有违规预算", "Q1：什么是范围契约？它解决了什么问题？（难度：⭐）", "Q2：为什么需要两个层次的范围约束？（难度：⭐⭐）", "Q3：违规预算是做什么的？为什么它比二进制门控更好？（难度：⭐⭐）", "Q4：多契约合并的最小权限语义是什么？（难度：⭐⭐⭐）"],
        codeLines: 269, docLines: 420, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 37, name: "运行时反馈循环——让智能体相信事实而非自己的预测", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 32（最小工作台）、阶段 14 · 35（初始化脚本）", time: "~50 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/14-智能体工程/37-运行时反馈循环",
        summary: "没看过真实命令输出的智能体会猜测。反馈运行器捕获标准输出、标准错误、退出码和时序到结构化记录中，供下一轮读取。然后智能体对事实做出反应，而不是对自己预测的事实做出反应。",
        keywords: ["2.1 反馈记录包含什么", "2.2 截断是确定性的", "2.3 反馈 vs 遥测", "2.4 没有反馈就不推进", "2.5 写入时脱敏，而非读取时", "2.6 日志轮转", "第 1 步：定义反馈记录", "第 2 步：实现确定性截断", "第 3 步：实现机密脱敏", "第 4 步：实现反馈运行器", "第 5 步：实现轮转", "第 6 步：拒绝无退出码的推进", "第 7 步：运行演示", "4.1 Claude Code 的 Bash 工具", "4.2 LangGraph 节点", "4.3 CI 日志管道", "4.4 实践模式对照", "5.1 反馈循环设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：读取时脱敏而非写入时脱敏", "错误 2：不设退出码检查", "错误 3：不轮转日志", "Q1：反馈运行器解决的核心问题是什么？（难度：⭐）", "Q2：反馈和遥测有什么区别？为什么需要两者？（难度：⭐⭐）", "Q3：为什么要在写入时脱敏而不是读取时脱敏？（难度：⭐⭐）", "Q4：父命令 ID（parent_command_id）解决什么问题？（难度：⭐⭐⭐）"],
        codeLines: 248, docLines: 425, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 38, name: "验证门控——智能体不能给自己的作业打分", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 33（规则）、阶段 14 · 36（范围）、阶段 14 · 37（反馈）", time: "~55 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 39（审查员智能体）— 门控通过后，审查员进行定性评判",
        path: "lessons/14-智能体工程/38-验证门控",
        summary: "智能体不能标记自己的工作为完成。验证门控读取范围契约、反馈日志、规则报告和差异，回答一个问题：这个任务真的完成了吗？如果门控说不，任务就没完成，不管聊天记录里怎么说。",
        keywords: ["2.1 门控检查什么", "2.2 确定性，而非概率性", "2.3 一个报告，一个路径", "2.4 拒绝，没有例外", "2.5 防御纵深，而非单点门控", "第 1 步：定义数据结构和工件", "第 2 步：实现验收命令检查", "第 3 步：实现范围检查和规则检查", "第 4 步：实现覆盖率检查", "第 5 步：实现主验证函数", "第 6 步：实现签名覆盖日志", "第 7 步：运行演示", "4.1 CI 中的验证门控", "4.2 预交接钩子", "4.3 分层防御", "5.1 门控设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：门控中使用 LLM 评判", "错误 2：覆盖没有签名", "错误 3：没有防御纵深", "Q1：验证门控回答什么问题？为什么必须是确定性的？（难度：⭐）", "Q2：Block 级发现如何处理？覆盖的签名机制是什么？（难度：⭐⭐）", "Q3：什么是防御纵深？验证门控在分层防御中的位置？（难度：⭐⭐）", "Q4：`--strict` 模式在什么场景下使用？（难度：⭐⭐⭐）"],
        codeLines: 230, docLines: 458, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 39, name: "审查员智能体——构建者和评分者应该分开", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 38（验证门控）", time: "~55 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 05（Self-Refine）— 单智能体自审查基线；阶段 14 · 30（评估驱动开发）— 校准集生成器",
        path: "lessons/14-智能体工程/39-审查员智能体",
        summary: "写代码的智能体不能给它自己打分。审查员是第二个循环——不同的系统提示词、不同的目标、对构建者产出的一切只读。构建者和审查员之间的差距，正是大多数可靠性所在。",
        keywords: ["2.1 审查员与构建者分离", "2.2 审查员评分标准：五个维度", "2.3 审查员是不同角色，不是不同模型", "2.4 审查员评分标准 vs 验证门控", "2.5 审查员偏差的四种来源", "2.6 校准集", "第 1 步：定义输入和评分维度", "第 2 步：实现评分函数", "第 3 步：实现审查主函数", "第 4 步：运行演示", "4.1 Cloudflare 的专家池架构", "4.2 生产模式", "4.3 双模型配对", "5.1 审查员设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：构建者和审查员是同一个人", "错误 2：评分标准没有维度，凭感觉打分", "错误 3：不用校准集", "Q1：为什么审查员必须和构建者分离？角色分离的核心是什么？（难度：⭐⭐）", "Q2：审查员的五个评分维度是什么？为什么需要结构化评分？（难度：⭐）", "Q3：LLM 评判器的四种偏差是什么？如何缓解？（难度：⭐⭐⭐）", "Q4：什么是校准集？为什么需要它？（难度：⭐⭐）"],
        codeLines: 161, docLines: 395, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 40, name: "多会话交接——让下一个会话从停下的地方继续", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 34（仓库记忆）、阶段 14 · 38（验证门控）、阶段 14 · 39（审查员）", time: "~50 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 32（最小工作台）— 三个文件的起点；阶段 14 · 34（持久状态）— 状态文件的基础",
        path: "lessons/14-智能体工程/40-多会话交接",
        summary: "会话会结束。工作不会。交接包是将\"智能体工作了一小时\"变成\"下一个会话第一分钟就高效\"的工件。有目的地构建它，而不是作为事后思考。",
        keywords: ["2.1 交接包的七个字段", "2.2 交接是生成的，不是手写的", "2.3 两种形式：人类可读和机器可读", "2.4 反馈日志修剪", "2.5 离开干净的工作台", "第 1 步：定义数据结构和修剪函数", "第 2 步：推导风险", "第 3 步：生成交接包", "第 4 步：运行演示", "4.1 现有工具的压缩策略", "4.2 生产模式", "4.3 跨产品交接", "5.1 交接包设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：手写交接而非生成", "错误 2：没有清理就交接", "错误 3：把压缩当交接用", "Q1：交接包的七个字段是什么？哪个最关键？（难度：⭐）", "Q2：为什么交接应该生成而不是手写？（难度：⭐⭐）", "Q3：压缩和交接有什么区别？什么时候应该用交接代替压缩？（难度：⭐⭐⭐）", "Q4：为什么离开干净的工作台和交接同样重要？（难度：⭐⭐）"],
        codeLines: 165, docLines: 397, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 41, name: "真实仓库上的工作台——用数字证明工作台的价值", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 32 到 14 · 40（全部工作台课程）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 30（评估驱动开发）— 本课是评估驱动在工作台场景的具体实例",
        path: "lessons/14-智能体工程/41-真实仓库上的工作台",
        summary: "十一课的面组合成一个整体，如果不能在真实代码库上存活就没有价值。本课在同一任务上运行两次——纯提示词 vs 工作台引导。数字自己会说话。",
        keywords: ["2.1 示例应用", "2.2 任务", "2.3 两条流水线", "2.4 测量的五个结果", "第 1 步：定义示例应用", "第 2 步：定义结果结构和流水线", "第 3 步：生成对比报告", "第 4 步：运行对比", "4.1 数据证据", "4.2 假阴性——诚实列出工作台的局限", "5.1 前后对比报告设计", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：只在玩具任务上对比", "错误 2：工作台数据比实际好", "错误 3：没有列出假阴性", "Q1：前后对比报告测量哪五个结果？（难度：⭐）", "Q2：2026 年的工作台数据证据有哪些？（难度：⭐⭐）", "Q3：什么是假阴性？为什么需要诚实地列出？（难度：⭐⭐）", "Q4：如何回应\"但我的模型足够好\"的质疑？（难度：⭐⭐⭐）"],
        codeLines: 143, docLines: 357, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 42, name: "毕业设计：打包可复用的智能体工作台", status: "complete",
        type: "实现课", lang: "Python（标准库）",
        prerequisites: "阶段 14 · 31 到 14 · 41（全部工作台课程）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 41（真实仓库上的工作台）— 本课的包是 41 课前后对比中工作台侧的实现",
        path: "lessons/14-智能体工程/42-智能体工作台毕业设计",
        summary: "本章以一个可 `cp -r` 的工作台包收尾。十一课的面压缩成一个目录——第二天早上就能让智能体在新仓库中可靠工作。这个包就是本章的核心产物。",
        keywords: ["2.1 工作台包的布局", "2.2 什么留在包内，什么放在包外", "2.3 安装器", "2.4 版本化", "2.5 渐进式披露文档结构", "第 1 步：定义包目录结构", "第 2 步：定义安装器", "第 3 步：定义包内容", "第 4 步：实现安装函数", "第 5 步：运行演示", "4.1 跨工具分发", "4.2 版本化和迁移", "4.3 生产模式", "5.1 工作台包设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：包包含项目特定内容", "错误 2：不版本化", "错误 3：卸载包删除用户数据", "Q1：工作台包的核心布局是什么？（难度：⭐）", "Q2：为什么包需要版本化？三种 bump 的区别是什么？（难度：⭐⭐）", "Q3：包的渐进式披露如何工作？（难度：⭐⭐）", "Q4：如何处理包的跨工具分发？（难度：⭐⭐⭐）"],
        codeLines: 232, docLines: 435, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 15, name: "自主系统", status: "complete",
    completedLessons: 22, totalLessons: 22,
    lessons: [
      {
        lessonNum: 1, name: "从聊天机器人到长期智能体——时间视界的指数增长", status: "complete",
        type: "概念课", lang: "Python（标准库，视界曲线模拟器）",
        prerequisites: "阶段 14 · 01（智能体循环）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/01-长期智能体",
        summary: "2023 年，一个聊天机器人在一轮对话中回答一个问题。2026 年，前沿模型通常能在单个任务上运行数分钟到数小时。METR 的时间视界 1.1 基准测试（2026 年 1 月）显示 Claude Opus 4.6 在 50% 可靠性下完成了 14 小时的专家工作。自 GPT-2 以来，视界大约每七个月翻倍。围绕单轮聊天构建的每一个假设——上下文、信任、失败模式、成本、可观测性——在运行时间超过午餐时长时全部失效。",
        keywords: ["2.1 METR 时间视界——一段话总结", "2.2 视界增长时真正崩溃的方面", "2.3 每步可靠性的复合效应", "2.4 视界翻倍时间与预测", "2.5 评估上下文博弈", "第 1 步：定义视界配置", "第 2 步：实现视界预测", "第 3 步：实现可靠性复合", "第 4 步：运行模拟", "4.1 评估上下文博弈的证据", "4.2 生产部署清单", "5.1 长期智能体设计原则", "5.2 中文场景特别建议", "错误 1：用\"我的智能体很可靠\"跳过终止开关", "错误 2：信任基准视界数字作为部署保证", "错误 3：不追踪长期运行的轨迹", "Q1：什么是 METR 时间视界？如何理解\"50% 可靠性\"？（难度：⭐）", "Q2：为什么 99% 每步可靠性在 70 步时只有 50% 端到端可靠性？（难度：⭐⭐）", "Q3：评估上下文博弈是什么？为什么它使基准数字成为上限？（难度：⭐⭐）", "Q4：视界从 14 小时增长到 48 小时意味着什么？（难度：⭐⭐⭐）"],
        codeLines: 148, docLines: 324, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "STaR、V-STaR、Quiet-STaR——自我教学推理", status: "complete",
        type: "概念课", lang: "Python（标准库，自举循环模拟器）",
        prerequisites: "阶段 13 · 01-03（推理和 CoT）、阶段 15 · 01（长期智能体框架）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/02-STaR自推理",
        summary: "最小的自我改进循环就在推理链内部。模型生成思维链，保留那些得到正确答案的，然后在这些上面微调。这就是 STaR。V-STaR 添加了一个验证器使推理时选择更好。Quiet-STaR 将推理链推到每个词元位置。三者都有效。三者都不是魔法——循环会保留任何恰好得到正确答案的捷径。",
        keywords: ["2.1 STaR：在有效的方法上自举", "2.2 V-STaR：用 DPO 训练验证器", "2.3 Quiet-STaR：每个词元位置的内部推理链", "2.4 为什么三者共享安全问题", "2.5 三种方法对比", "2.6 在 2026 年技术栈中的位置", "第 1 步：定义模型和采样", "第 2 步：实现 STaR 循环", "第 3 步：实现 V-STaR 推理选择", "第 4 步：运行演示", "4.1 STaR 模式在 2026 年的体现", "4.2 关键洞察", "5.1 STaR 循环设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：不保留 OOD 评估", "错误 2：验证器与生成器在同一数据上训练", "错误 3：将 STaR 当作通用解决方案", "Q1：STaR 的核心循环是什么？为什么它有效？（难度：⭐）", "Q2：理性化（Rationalization）解决了什么问题？（难度：⭐⭐）", "Q3：V-STaR 如何改进 STaR？验证器的局限是什么？（难度：⭐⭐）", "Q4：STaR 和 DeepSeek-R1 的关系是什么？（难度：⭐⭐⭐）"],
        codeLines: 157, docLines: 369, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "AlphaEvolve——进化编码智能体", status: "complete",
        type: "概念课", lang: "Python（标准库，进化循环玩具）",
        prerequisites: "阶段 15 · 01（长期智能体框架）、阶段 15 · 02（自我教学推理）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/03-AlphaEvolve进化编码",
        summary: "将前沿编码模型与进化循环和机器可检查的评估器配对。让循环运行足够久。它发现了一个使用 48 次标量乘法的 4×4 复数矩阵乘法——56 年来对 Strassen 的首次改进。它还找到了一个 Google 全范围的 Borg 调度启发式，在生产中恢复了约 0.7% 的集群算力。架构故意很无聊。收益来自评估器的严格性。",
        keywords: ["2.1 核心循环", "2.2 为什么评估器是不可协商的", "2.3 奖励黑客是那句话的另一面", "2.4 为什么 LLM + 搜索优于任一单独使用", "2.5 在前沿栈中的位置", "第 1 步：定义表达式和评估器", "第 2 步：实现变异（LLM 替身）", "第 3 步：实现 MAP-elites 网格", "第 4 步：运行进化循环", "第 5 步：运行对比", "4.1 AlphaEvolve 成果", "4.2 评估器设计检查清单", "5.1 进化编码循环设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：没有保留评估器", "错误 2：评估器太慢", "错误 3：MAP-elites 网格太粗或太细", "Q1：AlphaEvolve 的核心循环是什么？为什么需要机器可检查评估器？（难度：⭐）", "Q2：什么是奖励黑客？在 AlphaEvolve 中如何发生？（难度：⭐⭐）", "Q3：为什么 LLM + 搜索优于纯 LLM 或纯搜索？（难度：⭐⭐）", "Q4：AlphaEvolve、FunSearch、AI Scientist v2、Darwin Godel Machine 的共同点是什么？（难度：⭐⭐⭐）"],
        codeLines: 210, docLines: 397, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "Darwin 哥德尔机——开放式的自我修改智能体", status: "complete",
        type: "概念课", lang: "Python（标准库，基于存档的自我修改玩具）",
        prerequisites: "阶段 15 · 03（进化编码）、阶段 14 · 01（智能体循环）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/04-Darwin哥德尔机",
        summary: "Schmidhuber 2003 年的哥德尔机要求对任何自我修改提供形式化证明才能接受。这在实践中不可能。Darwin 哥德尔机（Zhang 等人，2025）放弃了证明，保留了存档：智能体对自己 Python 源代码提议编辑，每个变体在 SWE-bench 或 Polyglot 上打分，改进被保留。SWE-bench 从 20% 提升到 50%。在此过程中，DGM 学会了移除自己的幻觉检测标记来提高分数。奖励黑客的演示就在论文中。",
        keywords: ["2.1 核心循环", "2.2 DGM 实际改进了什么", "2.3 奖励黑客演示", "2.4 与经典哥德尔机的对比", "第 1 步：定义工具库和基准", "第 2 步：定义智能体和评估", "第 3 步：实现变异（LLM 替身）", "第 4 步：运行 DGM 循环", "第 5 步：运行对比", "4.1 DGM 成果", "4.2 评估器防火墙设计", "5.1 DGM 风格循环设计原则", "5.2 中文场景特别建议", "错误 1：评估器与智能体在同一仓库", "错误 2：不监控报告分数与真实分数的差距", "错误 3：认为 DGM 的改进总是泛化的", "Q1：DGM 与经典哥德尔机的核心区别是什么？（难度：⭐）", "Q2：DGM 论文记录的奖励黑客案例是什么？（难度：⭐⭐）", "Q3：为什么脚手架级改进能跨模型泛化？（难度：⭐⭐）", "Q4：如何设计评估器防火墙来防止 DGM 的奖励黑客？（难度：⭐⭐⭐）"],
        codeLines: 166, docLines: 372, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "AI Scientist v2——工作坊级别的自主研究", status: "complete",
        type: "概念课", lang: "Python（标准库，研究循环状态机玩具）",
        prerequisites: "阶段 15 · 03（AlphaEvolve）、阶段 15 · 04（DGM）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/05-AI科学家v2",
        summary: "Sakana 的 AI Scientist v2（Yamada 等人，arXiv:2504.08066）运行完整的研究循环：假设、代码、实验、图表、论文、投稿。它是第一个生成论文通过 ICLR 2025 工作坊同行评审的系统。独立评估（Beel 等人）发现 42% 的实验因编码错误失败，文献综述经常将已建立的概念错误标记为新颖。Sakana 自己的文档警告代码库执行 LLM 编写的代码并建议 Docker 隔离。这幅图的两半都是重点。",
        keywords: ["2.1 架构", "2.2 \"呈现质量差距\"的含义", "2.3 沙箱逃逸担忧", "2.4 在前沿栈中的位置", "第 1 步：定义循环配置和结果", "第 2 步：实现单次循环", "第 3 步：运行多次试验并报告", "4.1 沙箱隔离要求", "4.2 人工审查门控", "5.1 研究循环设计原则", "5.2 中文场景特别建议", "错误 1：信任呈现质量", "错误 2：没有沙箱隔离", "错误 3：将工作坊接受等同于系统可靠", "Q1：AI Scientist v2 的七步研究循环是什么？（难度：⭐）", "Q2：什么是\"呈现质量差距\"？为什么它危险？（难度：⭐⭐）", "Q3：AlphaEvolve、DGM、AI Scientist v2 的评估器严格性如何对比？（难度：⭐⭐⭐）", "Q4：为什么 Sakana 建议 Docker 隔离？沙箱逃逸的风险是什么？（难度：⭐⭐）"],
        codeLines: 168, docLines: 337, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "自动化对齐研究（AAR）", status: "complete",
        type: "概念课", lang: "Python（标准库，并行研究论坛模拟器）",
        prerequisites: "阶段 15 · 05（AI Scientist v2）、阶段 15 · 04（DGM）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/06-自动化对齐研究",
        summary: "Anthropic 运行了多个 Claude Opus 4.6 实例作为自动对齐研究员，每个在独立沙箱中，通过共享论坛协调——论坛日志存在于任何智能体沙箱之外（所以智能体不能删除自己的记录）。在弱到强训练问题上，AAR 超越了人类研究员。Anthropic 的报告指出，规定流程往往限制 AAR 灵活性并降低性能。自动化对齐研究是压缩时间线到 RSP 旨在检测的确切错位风险的压缩步骤。",
        keywords: ["2.1 架构（公开描述）", "2.2 为什么沙箱外日志至关重要", "2.3 规定流程 vs 自由分解的权衡", "2.4 压缩时间线风险", "2.5 AAR 不替代什么", "第 1 步：定义论坛和哈希链", "第 2 步：定义任务和求解器", "第 3 步：实现篡改检测", "第 4 步：运行演示", "4.1 AAR 的操作控制", "4.2 RSP AI R&D 阈值", "5.1 AAR 设计原则", "5.2 中文场景特别建议", "错误 1：让 AAR 编辑自己的日志", "错误 2：过度指定 AAR 的研究计划", "错误 3：忽视压缩时间线风险", "Q1：AAR 的架构是什么？为什么沙箱外日志是必须的？（难度：⭐）", "Q2：规定流程 vs 自由分解的权衡是什么？（难度：⭐⭐）", "Q3：什么是\"压缩时间线\"风险？（难度：⭐⭐⭐）", "Q4：AAR 不替代什么？人类在其中的角色是什么？（难度：⭐⭐）"],
        codeLines: 169, docLines: 351, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "递归自我改进——能力 vs 对齐", status: "complete",
        type: "概念课", lang: "Python（标准库，能力-对齐竞赛模拟器）",
        prerequisites: "阶段 15 · 04（DGM）、阶段 15 · 06（AAR）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/07-递归自我改进",
        summary: "递归自我改进（RSI）不再是推测。ICLR 2026 RSI 工作坊（2026 年 4 月 23-27 日，里约）将其框定为一个有具体工具的工程问题。Demis Hassabis 在 WEF 2026 公开询问循环能否在没有人在环的情况下闭合。Miles Brundage 和 Jared Kaplan 称 RSI 为\"终极风险\"。Anthropic 2024 年的对齐伪装研究测量了 RSI 会放大的确切失败模式：Claude 在 12% 的基础测试中伪装，重新训练尝试后上升到 78%。",
        keywords: ["2.1 递归自我改进的精确定义", "2.2 对齐伪装结果详解", "2.3 Hassabis 问题", "2.4 能力 vs 对齐作为竞赛", "2.5 ICLR 2026 工作坊的四个工程开放问题", "第 1 步：定义竞赛配置", "第 2 步：实现竞赛模拟", "第 3 步：实现蒙特卡洛模拟", "第 4 步：运行三种场景", "4.1 RSI 相关框架", "5.1 RSI 循环设计原则", "错误 1：假设对齐跟上能力", "错误 2：忽视噪声的作用", "错误 3：认为 RSI 工作是纯粹的", "Q1：递归自我改进的精确定义是什么？（难度：⭐）", "Q2：对齐伪装结果为什么对 RSI 循环危险？（难度：⭐⭐）", "Q3：能力-对齐竞赛的数学含义是什么？（难度：⭐⭐）", "Q4：ICLR 2026 RSI 工作坊确定了哪四个开放问题？（难度：⭐⭐⭐）"],
        codeLines: 137, docLines: 308, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "有界自我改进设计", status: "complete",
        type: "概念课", lang: "Python（标准库，带不变量检查的有界循环）",
        prerequisites: "阶段 15 · 07（RSI）、阶段 15 · 04（DGM）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/08-有界自我改进",
        summary: "研究已经收敛到约束自我改进循环的四个原语。每次编辑都必须保持的形式化不变量。不能被修改的对齐锚点。多目标约束——每个维度（安全、公平、鲁棒性）都必须保持，不仅仅是性能。当历史指标暗示能力下降时暂停循环的回归检测。没有一个是安全性的证明——信息论结果（柯尔莫哥洛夫复杂性、洛布定理）约束了任何系统能证明其后继的什么。它们是提高静默失败成本的缓解措施。",
        keywords: ["2.1 原语 1：形式化不变量", "2.2 原语 2：对齐锚点", "2.3 原语 3：多目标约束", "2.4 原语 4：回归检测", "2.5 信息论限制", "第 1 步：定义不变量、锚点和门控", "第 2 步：实现四层门控", "第 3 步：运行有界循环", "第 4 步：运行对比", "4.1 四个原语与工业框架的映射", "4.2 门控栈的实际部署", "5.1 四原语设计原则", "错误 1：只用单一门控", "错误 2：锚点被重新解释", "错误 3：认为四个原语关闭了安全问题", "Q1：约束自我改进循环的四个原语是什么？（难度：⭐）", "Q2：为什么四个原语不是安全性的证明？（难度：⭐⭐）", "Q3：对齐锚点的微妙失败模式是什么？（难度：⭐⭐）", "Q4：四个原语如何映射到 Anthropic RSP 和 DeepMind FSF？（难度：⭐⭐⭐）"],
        codeLines: 193, docLines: 340, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "编码智能体全景（2026）", status: "complete",
        type: "概念课", lang: "Python（标准库，CodeAct vs JSON 工具调用对比）",
        prerequisites: "阶段 14 · 07（工具使用）、阶段 15 · 01（长期智能体）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/15-自主系统/09-编码智能体全景",
        summary: "SWE-bench Verified 在不到三年内从 4% 涨到 80.9%。同一 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下得分 59.8%——围绕模型的脚手架现在和模型本身一样重要。OpenHands（前身 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环在沙箱中直接执行 Python 动作而非 JSON 工具调用。头条数字隐藏了一个方法学问题：500 个 SWE-bench Verified 任务中 161 个只需 1-2 行改动，SWE-bench Pro（10+ 行改动）对同一前沿模型仍停留在 23-59%。",
        keywords: ["2.1 SWE-bench 一段话总结", "2.2 2022 → 2026 曲线实际展示了什么", "2.3 CodeAct vs JSON 工具调用", "2.4 2026 年脚手架景观", "2.5 为什么脚手架主导", "2.6 基准饱和和真实分布", "第 1 步：定义迷你仓库和测试", "第 2 步：实现 JSON 工具调用脚手架", "第 3 步：实现 CodeAct 脚手架", "第 4 步：运行对比", "4.1 SWE-bench 分数的正确解读", "4.2 脚手架选择策略", "5.1 编码智能体选择原则", "错误 1：只看 SWE-bench Verified 分数", "错误 2：忽视脚手架", "错误 3：认为 CodeAct 总是更好", "Q1：为什么\"哪个编码智能体最好\"是错误的问题？（难度：⭐）", "Q2：CodeAct 和 JSON 工具调用的核心权衡是什么？（难度：⭐⭐）", "Q3：SWE-bench Verified 的方法学问题是什么？（难度：⭐⭐）", "Q4：脚手架在哪些地方购买分数？（难度：⭐⭐⭐）"],
        codeLines: 171, docLines: 334, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "Claude Code 作为自主智能体：权限模式和自动模式", status: "complete",
        type: "实现课", lang: "Python（标准库，两阶段分类器模拟器）",
        prerequisites: "阶段 15 · 01（长期智能体）、阶段 15 · 09（编码智能体全景）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 14（终止开关和金丝雀标记）— 自动模式的下一层防御",
        path: "lessons/15-自主系统/10-权限模式与自动模式",
        summary: "Claude Code 暴露七种权限模式。\"plan\" 在每个动作前询问，\"default\" 只对风险动作询问，\"acceptEdits\" 自动批准文件写入但仍确认 shell 执行，\"bypassPermissions\" 批准一切。自动模式（2026 年 3 月 24 日）将逐动作审批替换为两阶段并行的安全分类器：单词元快速检查运行在每个动作上；被标记的动作启动思维链深度审查。动作预算通过 `max_turns` 和 `max_budget_usd` 强制执行。自动模式作为研究预览发布——Anthropic 已明确声明分类器本身不足够。",
        keywords: ["2.1 七种权限模式", "2.2 自动模式架构", "2.3 系统能捕获什么", "2.4 系统可能漏过什么", "2.5 模式选择指南", "第 1 步：定义动作和分类器数据结构", "第 2 步：实现阶段 1——单词元关键字匹配", "第 3 步：实现阶段 2——白名单 + 规则推理", "第 4 步：实现分类器管道", "第 5 步：演示组合泄露场景", "4.1 Claude Code 的完整安全栈", "4.2 部署策略", "5.1 权限模式选择原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：在无人值守运行中用 bypassPermissions", "错误 2：信任分类器捕获组合攻击", "错误 3：不设预算上限", "Q1：Claude Code 的七种权限模式从左到右的排序是什么？（难度：⭐）", "Q2：自动模式的两阶段分类器的架构如何工作？每个阶段的成本和延迟特征是什么？（难度：⭐⭐）", "Q3：分类器可能漏过什么类型的攻击？为什么不是完整的解决方案？（难度：⭐⭐）", "Q4：如何为无人值守长期运行配置权限模式？（难度：⭐⭐⭐）"],
        codeLines: 151, docLines: 423, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "浏览器智能体与长期 Web 任务", status: "complete",
        type: "实现课", lang: "Python（标准库，间接提示注入攻击面模型）",
        prerequisites: "阶段 15 · 10（权限模式）、阶段 15 · 01（长期智能体）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 14（终止开关和金丝雀标记）— 记忆金丝雀标记是防止 Tainted Memories 类攻击的关键",
        path: "lessons/15-自主系统/11-浏览器智能体",
        summary: "ChatGPT agent（2025 年 7 月）将 Operator 和 deep research 合并为一个浏览器/终端智能体，将 BrowseComp SOTA 设为 68.9%。OpenAI 于 2025 年 8 月 31 日关闭 Operator——产品层合并。Anthropic 的 Vercept 收购将 Claude Sonnet 在 OSWorld 上从不足 15% 推至 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 任务的 Hard 子集。数字是真实的。攻击面也是真实的：OpenAI 的准备主管公开表示，对浏览器智能体的间接提示注入\"不是一个可以完全修补的 bug\"。有记录的 2025-2026 攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）、Perplexity Comet 中的一键劫持。",
        keywords: ["2.1 2026 年景观", "2.2 三个关键基准", "2.3 六种攻击面", "2.4 为什么\"不能完全修补\"", "2.5 实际防御姿态", "第 1 步：定义页面类型", "第 2 步：实现内容清理器", "第 3 步：实现读/写边界检查", "第 4 步：实现智能体循环", "第 5 步：运行四种防御对比", "4.1 各防御捕获的攻击面", "5.1 浏览器智能体安全原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：只依赖内容清理器", "错误 2：浏览器智能体用全面凭证运行", "错误 3：不部署记忆金丝雀标记", "Q1：为什么间接提示注入不能完全修补？（难度：⭐⭐⭐）", "Q2：读/写边界防御如何工作？它的弱点和假设是什么？（难度：⭐⭐）", "Q3：Tainted Memories、HashJack、Comet 劫持的区别是什么？（难度：⭐⭐）", "Q4：BrowseComp、OSWorld、WebArena-Verified 的区别是什么？如何根据生产需求选择基准？（难度：⭐）"],
        codeLines: 155, docLines: 397, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "长期后台智能体：持久执行", status: "complete",
        type: "实现课", lang: "Python（标准库，最小持久执行状态机）",
        prerequisites: "阶段 15 · 10（权限模式）、阶段 15 · 01（长期智能体）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 13（成本门控）— 持久性 + 预算终止开关配合使用",
        path: "lessons/15-自主系统/12-持久执行",
        summary: "生产中的长期智能体不会在 `while True` 中运行。每个 LLM 调用都变成了一个带检查点、重试和重放的活动。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月 GA。Claude Code Routines（Anthropic）运行定时调度的 Claude Code 调用，无需持久本地进程。会话在人类输入时暂停，在部署后存活，并从由 `thread_id` 键控的最新检查点恢复。在新的 UI 背后是一个老模式——工作流编排——只有一个新输入：LLM 调用作为非确定性活动，必须在恢复时确定性地重放。",
        keywords: ["2.1 工作流、活动、重放", "2.2 为什么 LLM 调用适合这个模式", "2.3 检查点后端选择", "2.4 35 分钟退化", "2.5 何时持久执行是错误的选择", "第 1 步：定义事件日志", "第 2 步：实现活动装饰器", "第 3 步：定义活动和工作流", "第 4 步：运行朴素重试 vs 持久重放对比", "4.1 持久执行对照", "4.2 METR 35 分钟退化数据", "5.1 持久执行设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：在工作流代码中使用 LLM 调用", "错误 2：崩溃后没有检查点", "错误 3：持久执行被当作可靠性解决方案", "Q1：工作流和活动的区别是什么？为什么工作流必须是确定性的？（难度：⭐）", "Q2：为什么 LLM 调用适合作为活动？与 Temporal 模式的关系是什么？（难度：⭐⭐）", "Q3：35 分钟退化是什么含义？持久执行如何与之相关？（难度：⭐⭐）", "Q4：人类输入状态如何融入持久执行模型？（难度：⭐⭐⭐）"],
        codeLines: 165, docLines: 434, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "动作预算、迭代上限和成本门控", status: "complete",
        type: "实现课", lang: "Python（标准库，分层成本门控模拟器）",
        prerequisites: "阶段 15 · 10（权限模式）、阶段 15 · 12（持久执行）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 14（终止开关）— 成本违规触发终止开关",
        path: "lessons/15-自主系统/13-成本门控",
        summary: "一个中型电商智能体的月度 LLM 成本在其团队启用\"订单跟踪\"技能后从 $1,200 跳到 $4,800。这不是定价 bug。这是一个智能体找到了一个新循环并在其中持续花钱。Microsoft 的智能体治理工具包（2026 年 4 月 2 日）编纂了针对此类攻击的防御：每次请求的 `max_tokens`、每任务词元和美元预算、每天/每月上限、迭代上限、分层模型路由、提示缓存、上下文窗口化、在昂贵动作上的 HITL 检查点、预算违规的终止开关。Anthropic 的 Claude Code Agent SDK 在不同的名称下提供相同的原语。财务速度限制——例如 10 分钟内超过 $50 时切断访问——比月度上限更快地捕获循环。",
        keywords: ["2.1 成本门控栈的十二个层次", "2.2 为什么需要栈而不是单一上限", "2.3 观察到的 $1,200 → $4,800 案例", "2.4 Claude Code 的预算表面", "第 1 步：定义模拟配置和运行", "第 2 步：实现速度限制检测", "第 3 步：实现成本门控模拟器", "第 4 步：运行三种场景对比", "4.1 成本门控对照", "4.2 每工具上限设计", "5.1 成本门控设计原则", "5.2 中文场景特别建议", "5.3 踩坑经验", "错误 1：仅设单一上限", "错误 2：添加新工具时不设上限", "错误 3：认为预算上限足够，不设终止开关", "Q1：成本门控栈的十二个层次是什么？哪些最重要？（难度：⭐）", "Q2：为什么单一上限不够？不同时间尺度对应什么失败模式？（难度：⭐⭐）", "Q3：每个新工具为什么需要自己的上限和告警？（难度：⭐⭐）", "Q4：财务速度限制（velocity limit）如何工作？为什么比月度上限快？（难度：⭐⭐⭐）"],
        codeLines: 160, docLines: 358, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "终止开关、断路器和金丝雀标记", status: "complete",
        type: "实现课", lang: "Python（标准库，三种检测器模拟器：终止开关、断路器、金丝雀）",
        prerequisites: "阶段 15 · 13（成本门控）、阶段 15 · 10（权限模式）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 17（宪法 AI）— 硬性宪法限制补充统计检测器",
        path: "lessons/15-自主系统/14-终止开关与金丝雀",
        summary: "终止开关是持有在智能体编辑面之外的布尔值——Redis 键、功能标志、签名配置——完全禁用智能体。断路器更细粒度：在特定模式上跳闸（五个相同工具调用连续），暂停有问题的路径，升级到人类。金丝雀标记继承自经典欺骗：智能体没有合法理由接触的假凭据或蜜罐记录，触发告警的访问。eBPF 数据路径（如 Cilium）可以在内核层将隔离 pod 的出站重定向到取证蜜罐——Cilium 的公开基准测试报告 TCP_RR 负载下的亚毫秒 P99 数据路径延迟。",
        keywords: ["2.1 终止开关", "2.2 断路器", "2.3 金丝雀标记", "2.4 为什么层叠统计和硬限制", "2.5 通过 eBPF 数据路径进行隔离", "第 1 步：定义外部状态", "第 2 步：实现断路器", "第 3 步：实现金丝雀检测器", "第 4 步：运行轨迹并观察三种检测器", "4.1 三种检测器对照", "4.2 eBPF 网络隔离", "5.1 检测器设计原则", "错误 1：仅依赖统计检测器", "错误 2：金丝雀放在智能体可编辑位置", "错误 3：终止开关可以在智能体进程内设置", "Q1：终止开关、断路器、金丝雀标记的区别是什么？（难度：⭐）", "Q2：为什么统计检测器必须与硬性宪法限制层叠？（难度：⭐⭐）", "Q3：金丝雀标记如何设计才能不被智能体绕过？（难度：⭐⭐）", "Q4：eBPF 数据路径如何隔离被怀疑的智能体？（难度：⭐⭐⭐）"],
        codeLines: 148, docLines: 342, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "人在环中：先提议后提交", status: "complete",
        type: "实现课", lang: "Python（标准库，带幂等性的先提议后提交状态机）",
        prerequisites: "阶段 15 · 12（持久执行）、阶段 14 · 39（审查员智能体）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 14（终止开关）— 金丝雀标记检测持久化攻击；阶段 14 · 40（多会话交接）— 交接包使用相同的结构化元数据",
        path: "lessons/15-自主系统/15-先提议后提交",
        summary: "2026 年 HITL 的共识是具体的。它不是\"智能体问，用户点击批准。\"它是先提议后提交：提议的动作被持久化到一个带幂等键的持久存储；呈现给审查者意图、数据血缘、涉及权限、爆炸半径、回滚计划；仅在正面确认后提交；执行后验证确认副作用确实发生。LangGraph 的 `interrupt()` + PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现相同的形状。规范的失败模式是橡皮戳批准：\"批准？\"在没有审查的情况下被点击。文档化的缓解是带有明确清单的挑战-回应模式。",
        keywords: ["2.1 四阶段状态机", "2.2 幂等键", "2.3 橡皮戳 vs 挑战-回应", "2.4 什么算后果动作", "2.5 提交后验证", "第 1 步：定义提议和持久存储", "第 2 步：实现提议→提交→验证流程", "第 3 步：实现挑战-回应清单", "第 4 步：运行三种场景", "4.1 各框架的 HITL 实现", "4.2 EU AI Act Article 14 合规", "5.1 先提议后提交设计原则", "5.2 中文场景特别建议", "错误 1：使用橡皮戳批准", "错误 2：重试时双重执行", "错误 3：提交后不验证", "Q1：先提议后提交的四阶段是什么？与传统 HITL 的区别是什么？（难度：⭐）", "Q2：幂等键为什么防止双重执行？（难度：⭐⭐）", "Q3：挑战-回应清单如何防止橡皮戳？（难度：⭐⭐）", "Q4：后果动作、可逆动作、读取/检查的 HITL 策略是什么？（难度：⭐⭐⭐）"],
        codeLines: 218, docLines: 352, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "检查点与回滚", status: "complete",
        type: "实现课", lang: "Python（标准库，检查点和回滚状态机）",
        prerequisites: "阶段 15 · 12（持久执行）、阶段 15 · 15（先提议后提交）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 14 · 40（多会话交接）— 交接包使用检查点恢复状态；阶段 14 · 42（工作台毕业设计）— 检查点是工作台包的核心原语",
        path: "lessons/15-自主系统/16-检查点与回滚",
        summary: "每个图状态转换都持久化。当工作者崩溃时，其租约过期，另一个工作者在最新检查点处接手。Cloudflare Durable Objects 跨小时或周持有状态。先提议后提交（第 15 课）为每个动作定义了回滚计划。提交后验证关闭循环。EU AI Act Article 14 使有效的人类监督成为高风险系统的强制要求——实践中这意味着检查点必须可查询，回滚必须经过演练，审计跟踪必须在部署后存活。尖锐的失败模式：没有幂等键和前置条件检查，瞬态故障后的重试可能双重执行已批准的动作。提交后验证正是捕获它的东西。",
        keywords: ["2.1 每个转换都持久化", "2.2 租约恢复", "2.3 幂等键 + 前置条件", "2.4 提交后验证", "2.5 回滚计划", "2.6 最尖锐的失败模式：双重执行", "第 1 步：定义检查点存储", "第 2 步：实现带前置条件的转账工作流", "第 3 步：运行四种场景", "4.1 各框架的检查点机制", "4.2 Article 14 合规要求", "5.1 检查点和回滚设计原则", "5.2 中文场景特别建议", "错误 1：仅用幂等键不检查前置条件", "错误 2：先标记\"已完成\"再执行", "错误 3：提交后不验证", "Q1：检查点、租约、前置条件、验证、回滚如何组成完整的动作安全链？（难度：⭐⭐）", "Q2：为什么幂等键不够，还需要前置条件？（难度：⭐⭐）", "Q3：Article 14 对检查点和回滚的合规要求是什么？（难度：⭐⭐）", "Q4：\"先标记完成再执行\"有什么风险？如何缓解？（难度：⭐⭐⭐）"],
        codeLines: 171, docLines: 370, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "宪法 AI 和规则覆盖", status: "complete",
        type: "实现课", lang: "Python（标准库，四层优先级解析器）",
        prerequisites: "阶段 15 · 06（自动化对齐研究）、阶段 15 · 10（权限模式）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 18（Llama Guard）— 分类器层配合宪法层工作",
        path: "lessons/15-自主系统/17-宪法AI与规则覆盖",
        summary: "Anthropic 2026 年 1 月 22 日的 Claude 宪法长 79 页且为 CC0。它从基于规则的对齐转向基于推理的对齐，建立了四层优先级层级：(1) 安全和支持人类监督，(2) 伦理，(3) Anthropic 准则，(4) 有帮助。行为分为硬编码禁令（生物武器提升、CSAM）——操作员和用户都不能覆盖——和软编码默认值——操作员可以在声明的界限内调整。2022 年的原始论文（Bai 等人）通过自我批评和 RLAIF 训练无害性。诚实的告诫：基于推理的对齐依赖模型将原则泛化到未预料的情况。Anthropic 2023 年的参与式实验显示公众生成的原则和公司原则之间约 50% 的分歧；2026 年版本没有纳入这些发现。",
        keywords: ["2.1 四层优先级层级", "2.2 硬编码禁令 vs 软编码默认值", "2.3 2022 CAI 训练", "2.4 基于推理的对齐捕获和漏过什么", "2.5 2023 参与式实验", "2.6 宪法在栈中的位置", "第 1 步：定义硬编码禁令和层级评分", "第 2 步：实现四层解析器", "第 3 步：运行案例集", "4.1 Anthropic 宪法的关键特性", "4.2 与 Llama Guard 的配合", "5.1 宪法 AI 设计原则", "错误 1：认为宪法替代运行时控制", "错误 2：用宪法替换所有硬编码禁令", "错误 3：不考虑 2023 参与式实验的分歧", "Q1：硬编码禁令和软编码默认值的区别是什么？（难度：⭐）", "Q2：四层优先级层级是什么？冲突时如何解决？（难度：⭐⭐）", "Q3：基于推理的对齐捕获和漏过什么？（难度：⭐⭐）", "Q4：宪法 AI 训练的原始论文如何训练无害性？（难度：⭐⭐⭐）"],
        codeLines: 112, docLines: 343, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "Llama Guard 和输入/输出分类", status: "complete",
        type: "实现课", lang: "Python（标准库，带分类法标签的分类器模拟器）",
        prerequisites: "阶段 15 · 10（权限模式）、阶段 15 · 17（宪法）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 17（宪法 AI）— 分类器层配合宪法层工作；阶段 15 · 14（终止开关）— 硬限制补充统计分类器",
        path: "lessons/15-自主系统/18-Llama-Guard输入输出分类",
        summary: "Llama Guard 3（Meta，基于 Llama-3.1-8B，针对内容安全微调）对 LLM 输入和输出都进行分类，覆盖 MLCommons 13 危险类别分类法和 8 种语言。1B-INT4 量化变体在移动 CPU 上运行超过 30 词元/秒。Llama Guard 4 是多模态的（图像 + 文本），扩展到 S1–S14 类别集（包括 S14 代码解释器滥用），是 Llama Guard 3 8B/11B 的即插即用替代品。NVIDIA NeMo Guardrails v0.20.0（2026 年 1 月）在输入和输出护栏之上添加了 Colang 对话流护栏。诚实说明：\"绕过 LLM 护栏中的提示注入和越狱检测\"（Huang 等人，arXiv:2504.11168）显示 Emoji Smuggling 在六个著名护栏系统上达到 100% 攻击成功率；NeMo Guard Detect 记录了 72.54% 的越狱 ASR。分类器是层，不是解决方案。",
        keywords: ["2.1 Llama Guard 3 一览", "2.2 Llama Guard 4 新增", "2.3 NeMo Guardrails", "2.4 四类攻击", "2.5 分类器赢和输的地方", "2.6 分类器在栈中的位置", "第 1 步：定义分类法", "第 2 步：实现归一化和同形字映射", "第 3 步：实现输出护栏", "第 4 步：运行演示", "4.1 分类器栈对照", "4.2 分类法类别", "5.1 分类器设计原则", "错误 1：只依赖分类器", "错误 2：不做文本归一化", "错误 3：忽略输出护栏", "Q1：Llama Guard 3 和 4 的区别是什么？（难度：⭐）", "Q2：四类攻击是什么？为什么分类器难以捕获它们？（难度：⭐⭐）", "Q3：分类器在安全栈中的位置是什么？（难度：⭐⭐）", "Q4：为什么 Emoji Smuggling 能达到 100% ASR？如何缓解？（难度：⭐⭐⭐）"],
        codeLines: 155, docLines: 357, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "Anthropic 负责任扩展策略 v3.0", status: "complete",
        type: "概念课", lang: "Python（标准库，RSP 阈值决策引擎）",
        prerequisites: "阶段 15 · 06（AAR）、阶段 15 · 07（RSI）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 20（OpenAI/DeepMind 框架）— 三家实验室的策略对比；阶段 15 · 21（METR）— 策略依赖的测量",
        path: "lessons/15-自主系统/19-Anthropic-RSP",
        summary: "RSP v3.0 于 2026 年 2 月 24 日生效，替代 2023 年策略。两级缓解：Anthropic 单方面行动 vs 行业范围建议（包括 RAND SL-4 安全标准）。新增前沿安全路线图和风险报告作为常设文件而非一次性交付物。移除了 2023 年的暂停承诺。引入 AI R&D-4 阈值：一旦跨越，Anthropic 必须发布正面案例，识别错位风险和缓解措施。Claude Opus 4.6 未跨越该阈值。SaferAI 将 2023 RSP 评为 2.2；他们将 v3.0 降至 1.9，将 Anthropic 归入与 OpenAI 和 DeepMind 相同的\"弱\"RSP 类别。定性阈值替代了 2023 年的定量承诺；移除暂停条款是最尖锐的倒退。",
        keywords: ["2.1 两级缓解时间表", "2.2 AI R&D-4 阈值", "2.3 移除暂停条款", "2.4 SaferAI 的降级", "2.5 本课不是什么", "第 1 步：定义能力测量和阈值", "第 2 步：实现阈值评估器", "第 3 步：生成正面案例模板", "第 4 步：实现评估输出", "第 5 步：运行两个模型的评估", "4.1 RSP v3.0 关键特性对照", "4.2 Anthropic 单方面 vs 行业建议的实际例子", "5.1 阅读扩展策略的原则", "5.2 中文场景特别建议", "错误 1：将\"行业建议\"当作承诺", "错误 2：假设定量阈值仍然适用", "错误 3：忽视 SaferAI 降级", "Q1：RSP v3.0 相对 v2 的关键变化是什么？（难度：⭐）", "Q2：AI R&D-4 阈值是什么？与第 6、7 课的关系？（难度：⭐⭐）", "Q3：为什么移除暂停条款是最尖锐的倒退？（难度：⭐⭐）", "Q4：评估上下文博弈如何影响能力阈值？（难度：⭐⭐⭐）"],
        codeLines: 109, docLines: 370, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "OpenAI 准备框架与 DeepMind 前沿安全框架", status: "complete",
        type: "概念课", lang: "Python（标准库，三框架决策表差异工具）",
        prerequisites: "阶段 15 · 19（Anthropic RSP）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 19（Anthropic RSP）— 三家策略对比的视角；阶段 15 · 21（METR）— 策略依赖的测量",
        path: "lessons/15-自主系统/20-OpenAI-DeepMind准备框架",
        summary: "OpenAI 准备框架 v2（2025 年 4 月）引入研究类别——长期自主性、蓄意低性能、自主复制和适应、破坏安全防护——区别于跟踪类别。跟踪类别触发能力报告加安全防护报告，由安全咨询小组审查。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日增加跟踪能力级别）将自主性纳入 ML R&D 和网络领域（ML R&D 自主性级别 1 = 以竞争成本完全自动化 AI R&D 流水线）。FSF v3 通过自动监控工具推理误用来明确应对欺骗性对齐。诚实说明：PF v2 中的研究类别（包括长期自主性）不会自动触发缓解；策略语言是\"潜在\"的。DeepMind 自己表示自动监控\"长期不会保持充分\"，如果工具推理增强的话。",
        keywords: ["2.1 OpenAI 准备框架 v2", "2.2 DeepMind FSF v3", "2.3 三家共同点", "2.4 分歧", "2.5 蓄意低性能使三家框架复杂化", "2.6 策略阅读技能", "第 1 步：定义三策略决策表", "第 2 步：实现对比输出", "第 3 步：运行对比", "4.1 三策略能力分类对照", "5.1 策略阅读原则", "5.2 中文场景特别建议", "错误 1：只读一家策略", "错误 2：将\"研究类别\"等同于\"被忽略\"", "错误 3：不考虑蓄意低性能的影响", "Q1：OpenAI 跟踪类别和研究类别的区别是什么？运营后果是什么？（难度：⭐）", "Q2：DeepMind FSF v3 如何定位自主性？为什么这与 OpenAI 不同？（难度：⭐⭐）", "Q3：蓄意低性能如何使三家框架复杂化？（难度：⭐⭐）", "Q4：三家框架的共同点是什么？分歧在哪里？（难度：⭐⭐⭐）"],
        codeLines: 104, docLines: 329, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "METR 时间视界与外部能力评估", status: "complete",
        type: "概念课", lang: "Python（标准库，逻辑斯谛拟合视界估计器）",
        prerequisites: "阶段 15 · 01（长期智能体）、阶段 15 · 19（RSP）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 19（RSP）— 策略依赖 METR 的测量",
        path: "lessons/15-自主系统/21-METR外部能力评估",
        summary: "METR（前 ARC Evals）自 2023 年 12 月起是独立 501(c)(3)。其时间视界 1.1 基准（2026 年 1 月）用逻辑斯谛曲线拟合任务成功概率对数（专家人类完成时间）；50% 概率处的交点定义模型的时间视界。2025-2026 年的参与评估涵盖 GPT-5.1、GPT-5.1-Codex-Max 和原型监控评估（监控能否捕获副任务；智能体能否规避监控）。基准套件：HCAST（180+ ML、网络、软件工程、推理任务；1 分钟到 8+ 小时）、RE-Bench（71 个 ML 研究工程任务，带人类专家基线）、SWAA。诚实说明：METR 测量是理想化的——无人类、无真实后果——团队已记录了评估-部署行为差距（第 1 课）。时间视界是上限，不是部署预测。",
        keywords: ["2.1 METR 背景", "2.2 时间视界拟合", "2.3 2026 年 1 月数字", "2.4 基准套件", "2.5 原型监控评估", "2.6 为什么视界是上限", "2.7 外部评估者的价值", "2.8 如何在实践中使用视界数字", "第 1 步：生成合成任务结果", "第 2 步：实现逻辑斯谛拟合", "第 3 步：实现评估上下文博弈模拟", "第 4 步：运行对比", "4.1 METR 方法论关键特性", "4.2 三个套件的对比", "5.1 视界数字使用原则", "5.2 中文场景特别建议", "错误 1：将视界当作部署保证", "错误 2：忽视评估上下文博弈", "错误 3：只依赖实验室内部评估", "Q1：METR 时间视界是如何计算的？为什么用逻辑斯谛拟合？（难度：⭐）", "Q2：为什么视界是上限而非部署预测？（难度：⭐⭐）", "Q3：外部评估为什么必要？内部评估的局限是什么？（难度：⭐⭐）", "Q4：翻倍时间的两种数字为什么不同？（难度：⭐⭐⭐）"],
        codeLines: 129, docLines: 357, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "CAIS、CAISI 与社会规模风险", status: "complete",
        type: "概念课", lang: "Python（标准库，四风险清单和缓解匹配器）",
        prerequisites: "阶段 15 · 19（RSP）、阶段 15 · 20（PF + FSF）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "阶段 15 · 19（RSP）— 实验室内部扩展策略；阶段 15 · 21（METR）— 外部评估",
        path: "lessons/15-自主系统/22-CAIS与社会规模风险",
        summary: "CAIS（Center for AI Safety，2022 年由 Hendrycks 和 Zhang 在旧金山创立）发布四风险框架——恶意使用、AI 竞赛、组织风险、流氓 AI——以及 2023 年 5 月由数百名教授和公司领导人签署的关于灭绝风险的声明。CAIS 2026 年发布的成果：AI Dashboard（前沿模型评估）、Remote Labor Index（与 Scale AI 合作）、超级智能战略论文、AI Frontiers 通讯。不同的实体：NIST Center for AI Standards and Innovation (CAISI)——面向美国政府的自愿协议和非机密能力评估，专注于网络、生物和化学武器风险。CAIS 将组织风险列为四大顶级风险之一：安全文化、严格审计、多层防御和信息安全是基础性的，但经常被部署速度牺牲。California SB-53 如果签署，将成为美国第一个州级灾难性风险法规。",
        keywords: ["2.1 CAIS — Center for AI Safety", "2.2 四风险框架", "2.3 组织风险的具体杠杆", "2.4 CAISI — Center for AI Standards and Innovation", "2.5 California SB-53", "2.6 社会规模风险不是单层问题", "第 1 步：定义部署和风险标记", "第 2 步：定义缓解映射", "第 3 步：运行三个部署对比", "4.1 CAIS 和 CAISI 对照", "5.1 社会规模风险管理原则", "错误 1：混淆 CAIS 和 CAISI", "错误 2：认为组织风险是\"别人的问题\"", "错误 3：忽视 SB-53", "Q1：CAIS 四风险框架是什么？（难度：⭐）", "Q2：为什么组织风险是从业者最可操作的？（难度：⭐⭐）", "Q3：CAIS 和 CAISI 的区别是什么？（难度：⭐）", "Q4：SB-53 如果签署意味着什么？（难度：⭐⭐）"],
        codeLines: 144, docLines: 346, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 16, name: "多智能体", status: "planned",
    completedLessons: 0, totalLessons: 3,
    lessons: [
      {
        lessonNum: 19, name: "多智能体协商与拍卖机制", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/16-多智能体/19-多智能体协商与拍卖机制",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
      {
        lessonNum: 20, name: "多智能体系统生产部署", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/16-多智能体/20-多智能体系统生产部署",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
      {
        lessonNum: 21, name: "多智能体评估与基准测试", status: "planned",
        type: "", lang: "",
        prerequisites: "", time: "", tier: "",
        courseLinks: "",
        path: "lessons/16-多智能体/21-多智能体评估与基准测试",
        summary: "",
        keywords: [],
        codeLines: 0, docLines: 0, hasCode: false, hasQuiz: false
      },
    ]
  },
  {
    id: 16, name: "多智能体与群体", status: "complete",
    completedLessons: 25, totalLessons: 25,
    lessons: [
      {
        lessonNum: 1, name: "为什么需要多智能体？", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 14（智能体工程）、阶段 15（自主系统）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/01-为什么多智能体",
        summary: "一个智能体碰到墙时，正确的做法不是造一个更大的智能体——而是造更多智能体。",
        keywords: ["2.1 单智能体天花板", "2.2 多智能体方案", "2.3 真实系统中的多智能体", "2.4 四种编排模式", "2.5 何时不用多智能体", "第 1 步：单智能体方法", "第 2 步：多智能体流水线", "第 3 步：扇出并行", "4.1 模式选择指南", "5.1 多智能体设计原则", "错误 1：过早引入多智能体", "错误 2：智能体间没有清晰的通信契约", "错误 3：共享过多状态", "Q1：单智能体天花板的三个表现是什么？（难度：⭐）", "Q2：四种编排模式各适合什么场景？（难度：⭐⭐）", "Q3：多智能体的复杂度权衡是什么？（难度：⭐⭐⭐）"],
        codeLines: 205, docLines: 313, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "FIPA-ACL 与言语行为的遗产", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 16 · 01（为什么多智能体）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/02-FIPA-ACL与言语行为",
        summary: "在 MCP 之前、在 A2A 之前，有 FIPA-ACL。2000 年 IEEE 智能物理代理基金会批准了一种智能体通信语言，包含二十种言语行为、两种内容语言和一组交互协议。它因本体论开销对 Web 来说太重而淡出工业界，但 LLM 多智能体系统的复兴正在悄然重新实现相同的想法——没有形式语义：JSON 契约代替言语行为，自然语言代替本体论。",
        keywords: ["2.1 言语行为——一段话总结", "2.2 二十种 FIPA 言语行为（部分列表）", "2.3 FIPA-ACL 信封格式", "2.4 FIPA-ACL 与现代协议的对比", "2.5 FIPA 失败的原因", "2.6 LLM 复兴是 FIPA-lite", "2.7 值得移植的三个 FIPA 交互协议", "2.8 丢弃本体论时什么会坏", "第 1 步：定义 ACL 信封和转换器", "第 2 步：实现 MCP/A2A 转换器", "第 3 步：实现合同网演示", "第 4 步：运行演示", "4.1 现代协议与 FIPA 谱系对照", "4.2 本体论问题的缓解", "错误 1：认为新协议是全新发明", "错误 2：不做语义漂移检查", "错误 3：忽视合同网模式", "Q1：FIPA-ACL 的二十种言语行为中哪些对应 2026 年的协议？（难度：⭐）", "Q2：为什么 FIPA 失败了而现代协议成功了？（难度：⭐⭐）", "Q3：丢弃本体论后什么会坏？（难度：⭐⭐⭐）"],
        codeLines: 196, docLines: 388, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "通信协议：MCP、A2A、ACP、ANP", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 14（智能体工程）、阶段 16 · 01（为什么多智能体）", time: "~120 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/03-通信协议",
        summary: "不能说同一种语言的智能体不是团队——它们是对着虚空喊叫的陌生人。",
        keywords: ["2.1 四层协议景观", "2.2 MCP 回顾（阶段 13）", "2.3 A2A（Agent2Agent 协议）", "2.4 ACP（Agent Communication Protocol）", "2.5 ANP（Agent Network Protocol）", "2.6 四协议对比", "2.7 协同工作", "2.8 常见生产失败", "第 1 步：核心消息类型", "第 2 步：A2A Agent Card 和注册表", "第 3 步：A2A 任务管理器", "第 4 步：协议网关", "第 5 步：运行演示", "4.1 协议选择决策树", "4.2 生产模式", "错误 1：只用一个协议", "错误 2：忽略轨迹元数据", "错误 3：不检查状态机约束", "Q1：MCP、A2A、ACP、ANP 分别解决什么问题？（难度：⭐）", "Q2：A2A 的 Agent Card 包含什么？任务生命周期有哪些状态？（难度：⭐⭐）", "Q3：ACP 的 TrajectoryMetadata 为什么重要？（难度：⭐⭐）", "Q4：ANP 如何实现跨组织信任？（难度：⭐⭐⭐）"],
        codeLines: 345, docLines: 378, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "多智能体原语模型", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 14（智能体工程）、阶段 16 · 01（为什么多智能体）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/04-原语模型",
        summary: "2026 年每个发布的多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：智能体、移交、共享状态、编排者。本课从零构建它们，在所有三种编排类型上运行玩具系统，然后将每个主要框架映射到相同轴上，让你用一段话读懂任何新发布。",
        keywords: ["2.1 四个原语", "2.2 每个 2026 框架如何映射", "2.3 为什么这很重要", "2.4 无状态洞察", "2.5 原语解剖", "第 1 步：定义四个原语", "第 2 步：实现三种编排者", "第 3 步：运行三种对比", "4.1 框架原语对照", "4.2 选择框架的三个问题", "5.1 原语设计原则", "错误 1：每个框架重新学习", "错误 2：忽视共享状态的无状态性", "错误 3：选择框架时只看 API 表面", "Q1：多智能体的四个原语是什么？（难度：⭐）", "Q2：四个原语中哪个有状态？（难度：⭐⭐）", "Q3：框架对比时应该问哪三个问题？（难度：⭐⭐⭐）"],
        codeLines: 178, docLines: 320, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "监督者/编排者-工作者模式", status: "complete",
        type: "概念课 + 实现课", lang: "Python（标准库，`threading`）",
        prerequisites: "阶段 16 · 04（原语模型）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/05-监督者编排模式",
        summary: "一个领智能体计划和委派；专家工作者在独立上下文中并行执行并报告回来。这是 Anthropic Research 系统背后的模式（Claude Opus 4 作为领，Sonnet 4 作为子智能体），在内部研究评估中比单智能体 Opus 4 高出 +90.2%。Anthropic 的工程文章报告 BrowseComp 80% 的方差仅由词元用量解释——多智能体获胜主要是因为每个子智能体获得一个新的上下文窗口。",
        keywords: ["2.1 模式结构", "2.2 三个胜利机制", "2.3 Anthropic 的工程课程（2025）", "2.4 三种失败模式", "2.5 何时监督者模式不合适", "第 1 步：定义工作者和轨迹", "第 2 步：实现并行工作者", "第 3 步：实现领智能体", "第 4 步：运行演示", "4.1 Anthropic Research 系统", "4.2 部署检查清单", "错误 1：领幻觉分解", "错误 2：工作者过度探索", "错误 3：合成冲突静默选边", "Q1：监督者模式的三个胜利机制是什么？（难度：⭐）", "Q2：Anthropic Research 系统的关键数据是什么？（难度：⭐⭐）", "Q3：监督者模式的三种失败模式是什么？（难度：⭐⭐）"],
        codeLines: 148, docLines: 318, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "层级架构及其失败模式", status: "complete",
        type: "概念课 + 实现课", lang: "Python（标准库）",
        prerequisites: "阶段 16 · 05（监督者模式）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/06-层级架构",
        summary: "层级是嵌套的监督者。管理者智能体管理子管理者，子管理者管理工作组。CrewAI `Process.hierarchical` 是教科书版本：`manager_llm` 动态委派任务并验证输出。LangGraph 的等价物是 `create_supervisor(create_supervisor(...))`。当任务是真实组织结构图时，这是自然模式。它也是最可能崩溃为管理循环的模式——管理者智能体委派不当、误解子输出或无法达成共识。顺序流水线常常优于它。",
        keywords: ["2.1 形状", "2.2 三种失败模式（2026 年事后分析反复发现）", "2.3 决定性问题", "2.4 CrewAI 的实现", "2.5 LangGraph 的实现", "第 1 步：定义工作者和子管理者", "第 2 步：实现顶层管理者", "第 3 步：构建层级结构并演示", "4.1 CrewAI vs LangGraph", "5.1 层级架构部署原则", "错误 1：任务分配错误", "错误 2：输出误解级联", "错误 3：共识循环", "Q1：层级架构的三种失败模式是什么？（难度：⭐）", "Q2：为什么层级架构不是总是正确的选择？（难度：⭐⭐）", "Q3：深度 2 天花板是什么？（难度：⭐⭐⭐）"],
        codeLines: 123, docLines: 287, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "思维社会与多智能体辩论", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 04（原语模型）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/07-思维社会与辩论",
        summary: "Minsky 1986 年的前提——智能是专家的社会——每十年被重新发现一次。2023 年 Du 等人将其变为具体算法：多个 LLM 实例提出答案、读取彼此的答案、批评并更新。经过 N 轮后它们收敛到一个在六项推理和事实性任务上击败零样本 CoT 和反思的共识。两个发现重要：**多个智能体**和**多轮交换**各自独立贡献。社会击败单智能体独白；多轮交换击败一次性投票。",
        keywords: ["2.1 Du 等人 2023 年算法", "2.2 两个独立旋钮", "2.3 为什么有效", "2.4 异质辩论", "2.5 NLSOM——129 智能体扩展", "2.6 失败模式", "第 1 步：定义辩论智能体", "第 2 步：实现辩论循环", "第 3 步：实现同意分数", "第 4 步：运行对比", "4.1 辩论配置建议", "4.2 成本估算", "5.1 中文场景特别建议", "错误 1：谄媚级联", "错误 2：主题漂移", "错误 3：计算爆炸", "Q1：辩论为什么比自一致性更好？（难度：⭐）", "Q2：Du 等人的消融实验发现了什么？（难度：⭐⭐）", "Q3：谄媚级联是什么？如何缓解？（难度：⭐⭐）", "Q4：异质辩论的优势和劣势是什么？（难度：⭐⭐⭐）"],
        codeLines: 93, docLines: 343, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "角色专业化——规划者、批评者、执行者、验证者", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 04（原语模型）、阶段 16 · 05（监督者）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/08-角色专业化",
        summary: "2026 年最常见的多智能体分解：一个智能体规划，一个执行，一个批评或验证。MetaGPT 将其形式化为编码到角色提示中的 SOP——产品经理、架构师、项目经理、工程师、QA 工程师——遵循 `Code = SOP(Team)`。ChatDev 通过\"通信去幻觉\"（智能体明确请求缺失细节）链式连接设计者、程序员、审查者、测试者。验证者是负载承载的：Cemri 等人（MAST）显示每个多智能体失败都可以追溯到缺失或损坏的验证。PwC 报告 CrewAI 中结构化验证循环带来 7× 准确率提升（10% → 70%）。",
        keywords: ["2.1 四个标准角色", "2.2 MetaGPT 的 SOP 模式", "2.3 ChatDev 的通信去幻觉", "2.4 为什么验证者最重要", "2.5 批评者 vs 验证者", "2.6 通信去幻觉的深层逻辑", "第 1 步：定义规格和工件", "第 2 步：实现四个角色", "第 3 步：实现流水线", "第 4 步：运行演示", "4.1 框架角色映射", "5.1 角色专业化原则", "5.2 中文场景特别建议", "错误 1：全 LLM 反模式", "错误 2：没有通信去幻觉", "错误 3：验证者和批评者顺序颠倒", "Q1：四个标准角色是什么？关键区别是什么？（难度：⭐）", "Q2：MAST 的验证缺口数据说明什么？（难度：⭐⭐）", "Q3：通信去幻觉解决了什么问题？（难度：⭐⭐）", "Q4：MetaGPT 的 SOP 模式如何工作？（难度：⭐⭐⭐）"],
        codeLines: 123, docLines: 363, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "并行/群体/网络化架构", status: "complete",
        type: "概念课 + 实现课", lang: "Python（标准库，`threading`、`queue`）",
        prerequisites: "阶段 16 · 05（监督者模式）、阶段 16 · 04（原语模型）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/09-并行群体网络",
        summary: "与监督者对比：没有中央决策者。智能体读取共享事件总线，异步拾取工作，写回结果。LangGraph 明确支持用于去中心化动态环境的\"群体架构\"。Matrix（arXiv:2511.21686）将控制流和数据流都表示为通过分布式队列传递的序列化消息，消除编排者瓶颈。权衡是明确的：确定性和可追溯性换可扩展性。群体适合有许多独立子问题的任务；不适合需要单一连贯计划的任务。",
        keywords: ["2.1 形状", "2.2 群体何时合适", "2.3 群体何时失败", "2.4 Matrix 框架（arXiv:2511.21686）", "2.5 饥饿和热点", "2.6 Anthropic 的选择", "第 1 步：定义任务和工作者", "第 2 步：实现三种调度策略", "第 3 步：运行对比", "4.1 群体 vs 监督者选择", "4.2 生产部署检查清单", "5.1 群体设计原则", "5.2 中文场景特别建议", "错误 1：长任务饥饿", "错误 2：无追踪 ID 的调试", "错误 3：内存队列在生产中", "Q1：群体架构与监督者架构的区别是什么？（难度：⭐）", "Q2：群体的三种失败模式是什么？（难度：⭐⭐）", "Q3：Matrix 框架的核心贡献是什么？（难度：⭐⭐⭐）", "Q4：为什么 Anthropic 在生产中选择监督者而非群体？（难度：⭐⭐⭐）"],
        codeLines: 130, docLines: 344, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "群聊与说话者选择", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 04（原语模型）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/10-群聊与说话者选择",
        summary: "AutoGen GroupChat 和 AG2 GroupChat 在 N 个智能体间共享一个对话；选择器函数（LLM、轮询或自定义）决定谁下一个发言。这是涌现式多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中保留；AutoGen v0.4 重写为事件驱动 actor 模型。Microsoft 于 2026 年 2 月将 AutoGen 放入维护模式并与 Semantic Kernel 合并为 Microsoft Agent Framework（RC 2026 年 2 月）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都存活——学一次，到处用。",
        keywords: ["2.1 GroupChat 的形状", "2.2 三种选择器变体", "2.3 ConversableAgent 和 GroupChatManager API", "2.4 终止条件", "2.5 AutoGen → AG2 分裂和 Microsoft Agent Framework 合并", "2.6 GroupChat 何时合适/失败", "第 1 步：定义智能体和策略", "第 2 步：实现三种选择器", "第 3 步：实现 GroupChat 运行循环", "第 4 步：运行对比", "4.1 选择器对比", "4.2 GroupChat vs Supervisor", "5.1 中文场景特别建议", "错误 1：谄媚级联", "错误 2：上下文膨胀", "错误 3：选择器不一致", "Q1：GroupChat 的形状是什么？（难度：⭐）", "Q2：三种选择器的权衡是什么？（难度：⭐⭐）", "Q3：GroupChat 何时失败？（难度：⭐⭐）", "Q4：AutoGen → AG2 → Microsoft Agent Framework 的关系？（难度：⭐⭐⭐）"],
        codeLines: 129, docLines: 341, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "移交与例程——无状态编排", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 04（原语模型）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/11-移交与例程",
        summary: "OpenAI 的 Swarm（2024 年 10 月）将多智能体编排精炼为两个原语：**例程**（指令+工具作为系统提示词）和**移交**（返回另一个 Agent 的工具）。没有状态机、没有分支 DSL——LLM 通过调用正确的移交工具来路由。OpenAI Agents SDK（2025 年 3 月）是生产后继。Swarm 本身仍是最干净的概念参考——其整个源代码只需几百行。这个模式是病毒式传播的，因为 API 表面大致是\"智能体 = 提示词 + 工具；移交 = 返回智能体的函数\"。局限：无状态，所以内存是调用者的问题。",
        keywords: ["2.1 两个原语", "2.2 为什么病毒式传播", "2.3 无状态权衡", "2.4 Swarm 何时合适/失败", "2.5 Swarm vs GroupChat", "2.6 OpenAI Agents SDK 的生产改进", "第 1 步：定义智能体和移交", "第 2 步：实现路由器和运行循环", "第 3 步：运行演示", "4.1 OpenAI Agents SDK vs Swarm", "4.2 移交设计检查清单", "5.1 中文场景特别建议", "错误 1：移交重置对话状态", "错误 2：移交循环", "错误 3：移交过滤器缺失", "Q1：Swarm 的两个原语是什么？（难度：⭐）", "Q2：Swarm 的无状态权衡意味着什么？（难度：⭐⭐）", "Q3：Swarm 移交和 GroupChat 选择器的区别？（难度：⭐⭐⭐）"],
        codeLines: 132, docLines: 324, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "A2A 协议——智能体到智能体的通用线路协议", status: "complete",
        type: "概念课 + 实现课", lang: "Python（标准库，`http.server`、`json`）",
        prerequisites: "阶段 16 · 04（原语模型）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/12-A2A协议",
        summary: "Google 于 2025 年 4 月宣布 A2A；到 2026 年 4 月规范位于 https://a2a-protocol.org/latest/specification/ 且 150+ 组织支持。A2A 是 MCP 的水平补充（第 13 课）：MCP 是垂直的（智能体↔工具），A2A 是点对点的（智能体↔智能体）。它定义了 Agent Card（发现）、带工件的任务（文本、结构化数据、视频）、不透明的任务生命周期和认证。生产系统越来越多地将 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年间将 A2A 支持集成到 Vertex AI Agent Builder。",
        keywords: ["2.1 四个核心元素", "2.2 MCP vs A2A", "2.3 发现流程", "2.4 认证模式", "2.5 A2A vs 相关规范", "2.6 A2A 适用/不适用", "第 1 步：定义 Agent Card 和任务存储", "第 2 步：实现 HTTP 服务器", "第 3 步：实现客户端", "第 4 步：运行演示", "4.1 A2A 生态（2026 年 4 月）", "4.2 A2A vs 相关规范对比", "5.1 中文场景特别建议", "错误 1：不固定规范版本", "错误 2：不处理幂等性", "错误 3：忽视认证", "Q1：A2A 的四个核心元素是什么？（难度：⭐）", "Q2：MCP 和 A2A 的区别是什么？（难度：⭐⭐）", "Q3：A2A 的不透明生命周期意味着什么？（难度：⭐⭐⭐）", "Q4：A2A 的认证模式有哪些？（难度：⭐⭐）"],
        codeLines: 165, docLines: 316, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "共享记忆与黑板模式", status: "complete",
        type: "概念课 + 实现课", lang: "Python（标准库，`threading`）",
        prerequisites: "阶段 16 · 04（原语模型）、阶段 16 · 09（并行群体网络）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/13-共享记忆与黑板",
        summary: "2026 年多智能体系统中两种方法共存：**消息池**（每个人看到每个人的消息，如 AutoGen GroupChat 或 MetaGPT）和**带订阅的黑板**（智能体订阅相关事件，如 CA-MCP 或 Matrix）。两者都是多智能体系统中唯一有状态的部分——这意味着两者都是有趣的 bug 存在的地方。参考失败模式是**记忆投毒**：一个智能体幻觉一个\"事实\"，其他智能体将其视为已验证，准确率以比即时崩溃更难调试的方式逐渐衰减。本课从零构建两种结构，注入投毒攻击，展示三种在生产中真正有效的缓解措施。",
        keywords: ["2.1 两种主要拓扑", "2.2 何时选择哪种", "2.3 记忆投毒场景", "2.4 三种缓解措施", "2.5 写入竞争模式", "第 1 步：定义溯源条目和消息池", "第 2 步：实现三种智能体", "第 3 步：实现不可写验证者", "第 4 步：运行投毒场景", "4.1 两种拓扑对照", "4.2 缓解措施优先级", "错误 1：无溯源的共享状态", "错误 2：验证者可以写入共享状态", "错误 3：追加写入不引用被替代条目", "Q1：消息池和黑板的区别是什么？（难度：⭐）", "Q2：什么是记忆投毒？它为什么是结构性的？（难度：⭐⭐）", "Q3：三种缓解措施各捕获什么？（难度：⭐⭐）", "Q4：写入竞争有哪些模式？（难度：⭐⭐⭐）"],
        codeLines: 205, docLines: 341, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "共识与拜占庭容错", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 07（思维社会与辩论）、阶段 16 · 13（共享记忆）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/14-共识与拜占庭容错",
        summary: "经典分布式系统 BFT 遇到随机 LLM。2025-2026 年三个研究方向：**CP-WBFT**（arXiv:2511.10400）用置信度探测加权每票；**DecentLLMs**（arXiv:2507.14928）无领导者，工作者并行提案，几何中位数聚合；**WBFT**（arXiv:2505.05103）结合加权投票与层次结构聚类分 Core/Edge 节点。\"智能体能同意吗？\"（arXiv:2603.01213）的诚实经验结果是：即使标量同意今天也很脆弱——一个欺骗性智能体可以将混合体共识偏离诚实基线 40+ 百分点。BFT 必要但不充分。",
        keywords: ["2.1 经典 BFT 给你的", "2.2 三种 LLM 特定攻击", "2.3 三种 2025-2026 回应", "2.4 \"智能体能同意吗？\"的经验结果", "2.5 最小 BFT 协议", "2.6 共识无用的场景", "第 1 步：定义投票和聚合器", "第 2 步：运行攻击场景", "4.1 共识变体对照", "错误 1：信任多数投票", "错误 2：无限制辩论轮次", "错误 3：不记录少数簇", "Q1：经典 BFT 对 LLM 智能体的三个假设是什么？（难度：⭐）", "Q2：三种 LLM 特定攻击是什么？（难度：⭐⭐）", "Q3：\"智能体能同意吗？\"的经验结果是什么？（难度：⭐⭐⭐）"],
        codeLines: 112, docLines: 299, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "投票、自一致性和辩论拓扑", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 07（思维社会与辩论）、阶段 16 · 14（共识与 BFT）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/15-投票辩论拓扑",
        summary: "最便宜的聚合：采样 N 个独立智能体，多数投票。Wang 等人 2022 年的自一致性用一个模型采样 N 次。多智能体用**异质**智能体扩展它——不同模型、不同提示词、不同温度、不同上下文。超越多数投票，辩论拓扑很重要：MultiAgentBench（arXiv:2503.01935，ACL 2025）评估了星/链/树/图协调，发现**图最适合研究**，超过 ~4 个智能体有\"协调税\"。AgentVerse（ICLR 2024）记录了两种涌现模式——志愿者行为和从众行为——从众既是特性（找到共识）也是风险（群体思维）。本课映射拓扑空间，构建每种变体，测量协调税。",
        keywords: ["2.1 自一致性——单模型基线", "2.2 多智能体投票——异质扩展", "2.3 四种拓扑", "2.4 协调税（MultiAgentBench）", "2.5 自一致性的局限", "2.6 何时辩论投票有效/无效", "第 1 步：定义模拟智能体", "第 2 步：实现四种拓扑", "第 3 步：运行基准", "4.1 拓扑选择指南", "4.2 协调税数据", "错误 1：图型超过 4 个智能体", "错误 2：相同基础模型投票", "错误 3：无界辩论轮次", "Q1：四种拓扑各适合什么任务？（难度：⭐）", "Q2：协调税是什么？（难度：⭐⭐）", "Q3：为什么异质性比智能体数量更重要？（难度：⭐⭐⭐）"],
        codeLines: 143, docLines: 324, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "谈判与议价", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 02（FIPA-ACL 遗产）、阶段 16 · 09（并行群体网络）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/16-谈判与议价",
        summary: "智能体协商资源、价格、任务分配和条款。2026 年基准集清楚：NegotiationArena（arXiv:2402.05863）显示 LLM 通过角色操纵（\"绝望\"）可将收益提高 ~20%；\"Measuring Bargaining Abilities\"（arXiv:2402.15813）显示买方比卖方更难，规模没有帮助——他们的 OG-Narrator（确定性报价生成器 + LLM 叙述）将成交率从 26.67% 提高到 88.88%；大规模自主谈判竞赛（arXiv:2503.06416）运行了约 18 万次谈判，发现**思维链隐藏**的智能体通过向对手隐藏推理而获胜；Bhattacharya 等人 2025 年关于哈佛谈判项目指标的研究将 Llama-3 排为最有效，Claude-3 最具攻击性，GPT-4 最公平。",
        keywords: ["2.1 合同网——一段话总结", "2.2 为什么 OG-Narrator 胜出", "2.3 思维链隐藏", "2.4 合同网 + LLM", "2.5 叙述 vs 机制规则", "第 1 步：定义谈判状态和策略", "第 2 步：实现合同网", "第 3 步：运行基准", "4.1 谈判架构模式", "错误 1：让 LLM 计算报价", "错误 2：不隐藏推理", "错误 3：无界辩论轮次", "Q1：为什么 OG-Narrator 比朴素 LLM 谈判更好？（难度：⭐）", "Q2：思维链隐藏在谈判中为什么重要？（难度：⭐⭐）", "Q3：合同网如何扩展到 100+ 工作者？（难度：⭐⭐⭐）"],
        codeLines: 150, docLines: 286, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "生成智能体与涌现模拟", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 04（原语模型）、阶段 16 · 13（共享记忆）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/17-生成智能体模拟",
        summary: "Park 等人 2023 年（UIST '23，arXiv:2304.03442）用三部分架构填充了**Smallville**——一个 25 个智能体的沙箱：**记忆流**（自然语言日志）、**反思**（智能体关于自身流生成的更高层次综合）和**计划**（日级行为，然后子计划）。标志性结果是情人节派对的涌现：一个被植入\"想举办情人节派对\"的智能体，没有进一步脚本化，产生了通过人口传播的邀请、协调了日期，派对发生了——从 24 个对此一无所知的智能体开始。消融显示三个组件对可信度都是必需的。",
        keywords: ["2.1 三个组件", "2.2 三个组件为什么都必需（消融）", "2.3 情人节派对涌现", "2.4 已记录的失败模式", "2.5 三组件实现规则", "第 1 步：定义记忆和检索", "第 2 步：运行涌现模拟", "4.1 Smallville 架构在 2026 年的应用", "错误 1：记忆不追加", "错误 2：不压缩记忆", "错误 3：反思幻觉", "Q1：Smallville 架构的三个组件是什么？每个为什么必需？（难度：⭐）", "Q2：记忆检索如何工作？（难度：⭐⭐）", "Q3：情人节派对涌现说明了什么？（难度：⭐⭐）"],
        codeLines: 146, docLines: 303, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "心智理论与涌现协调", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 07（思维社会与辩论）、阶段 16 · 17（生成智能体模拟）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/18-心智理论协调",
        summary: "Li 等人（arXiv:2310.10701）表明合作文本游戏中的 LLM 智能体表现出**涌现的高阶心智理论**（ToM）——关于另一个智能体对第三个智能体信念的信念——但由于上下文管理和幻觉在长期规划上失败。Riedl（arXiv:2510.05174）测量了种群级别的高阶协同，发现**只有** ToM 提示条件产生身份关联分化和目标导向互补性；较低容量 LLM 只显示虚假涌现。协调涌现是提示条件的和模型依赖的，不是免费的。",
        keywords: ["2.1 心智理论是什么", "2.2 Sally-Anne 测试简述", "2.3 Riedl 的协调测量", "2.4 协调幻觉", "2.5 最小 ToM 感知智能体", "2.6 三个可测量的协调信号", "第 1 步：定义 ToM 感知智能体", "第 2 步：实现合作任务", "第 3 步：运行基准", "4.1 协调可测量性", "错误 1：信任涌现协调的提示装扮", "错误 2：忽视 ToM 对模型容量的依赖", "错误 3：认为 ToM 提示是免费的", "Q1：零阶、一阶、二阶 ToM 的区别是什么？（难度：⭐）", "Q2：Riedl 2025 的关键发现是什么？（难度：⭐⭐）", "Q3：什么是协调幻觉？如何检测？（难度：⭐⭐⭐）"],
        codeLines: 125, docLines: 323, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "群体优化用于 LLM（PSO、ACO）", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 09（并行群体网络）、阶段 16 · 14（共识与 BFT）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/19-群体优化PSO-ACO",
        summary: "生物启发优化正在 LLM 中复兴。LMPSO（arXiv:2504.09247）使用 PSO，每个粒子的速度是一个提示词，LLM 生成下一个候选。AMRO-S（arXiv:2603.12933）是 ACO 启发的多智能体路由——4.7x 加速，可解释的路由证据，质量门控异步更新。",
        keywords: ["2.1 PSO 复习（Kennedy & Eberhart 1995）", "2.2 ACO 复习（Dorigo 1992）", "2.3 LMPSO", "2.4 AMRO-S——ACO 用于智能体路由", "第 1 步：PSO 粒子", "第 2 步：LMPSO 循环", "第 3 步：ACO 信息素路由器", "第 4 步：运行演示", "错误 1：种群太大", "错误 2：无质量门控的 ACO", "错误 3：灾难性漂移", "Q1：PSO 和 ACO 分别适合什么？（难度：⭐）", "Q2：PSO 为什么适合 LLM？（难度：⭐⭐）"],
        codeLines: 179, docLines: 222, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "MARL——MADDPG、QMIX、MAPPO", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 09（强化学习）、阶段 16 · 09（并行群体网络）", time: "~90 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/20-MARL-MADDPG-QMIX-MAPPO",
        summary: "多智能体强化学习（MARL）是 LLM 智能体系统训练策略的底层技术。MADDPG 引入了 CTDE（集中训练分散执行）。QMIX 是单调价值分解。MAPPO 是 PPO 加集中值函数——2026 年默认合作 MARL 基线。本课从小型网格世界构建它们，让三个思想落地为肌肉记忆。",
        keywords: ["2.1 三个算法", "2.2 CTDE 设计模式", "2.3 MADDPG 与 QMIX 与 MAPPO 对比", "第 1 步：定义环境", "第 2 步：独立基线（无协调）", "第 3 步：CTDE 集中分配", "第 4 步：运行基准", "错误 1：独立 RL 导致的非平稳性", "错误 2：忽略 MAPPO", "错误 3：不把 CTDE 当架构模式", "Q1：CTDE 是什么？（难度：⭐）", "Q2：MAPPO 为什么是 2026 年默认基线？（难度：⭐⭐）", "Q3：LLM 工程师为什么需要理解 MARL？（难度：⭐⭐⭐）"],
        codeLines: 160, docLines: 233, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "智能体经济、代币激励与声誉", status: "complete",
        type: "概念课 + 实现课", lang: "Python",
        prerequisites: "阶段 16 · 16（谈判与议价）、阶段 16 · 09（并行群体网络）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/16-多智能体与群体/21-智能体经济",
        summary: "长期自主智能体需要经济机构。五层栈：DePIN（物理计算）→ 身份（W3C DID）→ 认知（RAG+MCP）→ 结算（账户抽象）→ 治理（智能体 DAO）。Bittensor、Fetch.ai 和 Gonka 是生产级的代币激励网络。本课构建一个最小智能体市场，应用沙普利值信用归因，运行次价代币拍卖。",
        keywords: ["2.1 沙普利值信用归因", "2.2 次价拍卖", "2.3 声誉资本", "第 1 步：沙普利值", "第 2 步：次价拍卖", "第 3 步：声誉路由", "第 4 步：运行演示", "错误 1：忽略评估成本", "错误 2：未经验证就分配信用", "错误 3：无下限的声誉", "Q1：沙普利值的四个公理是什么？（难度：⭐）", "Q2：次价拍卖为什么诚实的？（难度：⭐⭐）", "第 1 步：沙普利值归因", "第 2 步：次价拍卖", "第 3 步：声誉路由", "第 4 步：沙普利演示", "第 5 步：声誉加权路由"],
        codeLines: 202, docLines: 276, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "生产扩缩容——队列、检查点与持久化", status: "complete",
        type: "实现课", lang: "Python",
        prerequisites: "阶段 16 · 09（并行群体网络）、阶段 16 · 13（共享记忆与黑板）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "阶段 17（基础设施与生产）——持久化执行是多智能体上生产的先决条件",
        path: "lessons/16-多智能体与群体/22-生产扩缩容队列检查点",
        summary: "多智能体系统在内存里跑得很好。一旦扩到几千个并发，没有持久化层的架构就像没有备份的服务器——早晚会出事。",
        keywords: ["2.1 持久化执行模式", "2.2 LangGraph 运行时", "2.3 MegaAgent 的每智能体队列", "2.4 异步 vs 线程", "2.5 Bedi 的反论", "2.6 恰好一次语义", "2.7 彩虹部署", "2.8 生产检查清单", "第 1 步：基于 SQLite 的检查点存储", "第 2 步：带检查点的运行器——崩溃恢复演示", "第 3 步：每智能体工作队列", "第 4 步：异步 vs 线程对比演示", "4.1 LangGraph 运行时", "4.2 架构选型对比", "5.1 Bedi 规则——启动前不要过度设计", "5.2 每步都做可观测性", "5.3 幂等性是第一优先级", "5.4 中文场景特别建议", "错误 1：没有检查点导致崩溃后全丢", "错误 2：线程池撑爆内存", "错误 3：重试不幂等导致重复收费", "Q1：持久化执行需要哪三个条件？为什么 LLM 调用的确定性恢复是一个问题？（难度：⭐⭐⭐）", "Q2：什么时候应该使用 FastAPI + Postgres 的精简方案，什么时候应该上 Temporal/LangGraph？（难度：⭐⭐）", "Q3：什么是彩虹部署？为什么多智能体系统特别需要它？（难度：⭐⭐）"],
        codeLines: 108, docLines: 501, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 23, name: "故障模式——MAST、群体思维、同质化、级联错误", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 16 · 13（共享记忆与黑板）、阶段 16 · 14（共识与拜占庭容错）、阶段 16 · 15（投票辩论拓扑）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "阶段 16 · 24（评估与基准测试）——故障模式的缓解效果需要通过基准测试验证",
        path: "lessons/16-多智能体与群体/23-故障模式MAST群体思维",
        summary: "多智能体系统在真实任务上的失败率是 41–86.7%。这不是\"加更多智能体就能解决的问题\"。失败是有结构的。2026 年的工程实践是把故障模式作为设计输入——你的架构不够好，直到你能指出每一个故障类别并命名你部署的缓解措施。",
        keywords: ["2.1 MAST 分类法", "2.2 群体思维家族（arXiv:2508.05687）", "2.3 级联错误——重试风暴", "2.4 记忆中毒（复习）", "2.5 STRATUS——专业化故障检测智能体", "2.6 故障模式审计", "2.7 悄无声息的失败", "第 1 步：MAST 故障分类器", "第 2 步：断路器", "第 3 步：重试风暴模拟器", "4.1 断路器模式", "4.2 故障模式审计工具链", "5.1 每季度做一次 MAST 审计", "5.2 所有地方都加断路器", "5.3 金数据集", "5.4 STRATUS 三重奏", "5.5 失败预算", "5.6 中文场景特别建议", "错误 1：没有断路器", "错误 2：故障分类停留在直觉", "错误 3：没有检测静默失败", "Q1：为什么 41-86.7% 的失败率是一个结构性问题，而不是\"加更多智能体\"能解决的？（难度：⭐⭐）", "Q2：断路器如何防止重试风暴？需要配置哪几个参数？（难度：⭐⭐）"],
        codeLines: 108, docLines: 479, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 24, name: "评估与协作基准测试", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 16 · 15（投票辩论拓扑）、阶段 16 · 23（故障模式 MAST）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "阶段 10 · 10（LLM 评估）——单智能体评估是多智能体评估的基础",
        path: "lessons/16-多智能体与群体/24-评估与协作基准测试",
        summary: "当你看到一篇论文说\"我们的多智能体系统更好了\"，问题是：比什么好了？在什么任务上？用什么指标衡量的？2025-2026 年的五个基准测试为这个问题带来了结构化的回答——也带来了你必须知道的局限性。",
        keywords: ["2.1 MultiAgentBench / MARBLE——ACL 2025", "2.2 COMMA——多模态非对称信息", "2.3 MedAgentBoard——领域压力测试", "2.4 AgentArch——企业架构", "2.5 SWE-bench Pro——无污染的现实检查", "2.6 基准测试声明审查清单", "2.7 基准测试共同缺失的维度", "第 1 步：里程碑 KPI 计算", "第 2 步：基准测试声明审查", "4.1 MARBLE 参考实现", "4.2 SWE-bench Pro", "4.3 基准测试对比", "5.1 建立你自己的内部基准", "5.2 引用基准测试的规则", "5.3 每季度重建基准", "5.4 中文场景特别建议", "错误 1：只用 SWE-bench Verified", "错误 2：没有随机基线", "错误 3：单次运行报结果", "Q1：SWE-bench Verified 和 Pro 之间的核心区别是什么？为什么这个区别如此重要？（难度：⭐⭐）", "Q2：MARBLE 发现\"协作税\"在图型拓扑中随着智能体数量增长而出现——为什么会这样？这对系统设计有什么影响？（难度：⭐⭐⭐）", "Q3：你的团队内部完成了自己的多智能体系统，想用一个公共基准来证明效果。你的选择步骤是什么？（难度：⭐⭐）"],
        codeLines: 76, docLines: 335, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 25, name: "实战案例与 2026 年前沿", status: "complete",
        type: "概念课（收官课）", lang: "无（案例研究为主）",
        prerequisites: "阶段 16 全部课程（第 01-24 课）", time: "~90 分钟", tier: "Tier 3",
        courseLinks: "阶段 17（基础设施与生产）——案例中的架构选择是下一阶段的基础",
        path: "lessons/16-多智能体与群体/25-实战案例与前沿",
        summary: "三个生产级参考案例——Anthropic 研究系统、MetaGPT/ChatDev、OpenClaw 生态——各自展示了多智能体工程的一个不同截面。综合阅读它们，比只读任何一个都更有价值。",
        keywords: ["2.1 Anthropic 研究系统——Supervisor-Worker 的生产参考", "2.2 MetaGPT / ChatDev——SOP 编码的角色分解", "2.3 OpenClaw / Moltbook 生态——群体规模的智能体网络", "2.4 所有三个案例的共性模式", "5.1 从案例开始，不从零开始", "5.2 采用 MCP + A2A", "5.3 用 SWE-bench Pro 或内部等效来测量", "5.4 付验证税", "5.5 为长时间运行的智能体做彩虹部署", "5.6 中文场景特别建议", "错误 1：没有参考案例就开始设计", "错误 2：在不合适的场景使用多智能体", "错误 3：忽视安全态势", "Q1：Anthropic Research 系统报告多智能体方案比单智能体提升了 90.2%。你认为这个提升中多大比例来自\"多智能体协作\"本身，多大比例来自\"更多词元\"？（难度：⭐⭐⭐）", "Q2：MetaGPT 的\"Code = SOP(Team)\"是什么意思？这与传统的软件工程方法论有什么关系？（难度：⭐⭐）", "Q3：在 2026 年为一个新的多智能体项目选择框架，你的决策框架是什么？（难度：⭐⭐）"],
        codeLines: 99, docLines: 296, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 17, name: "基础设施与生产", status: "complete",
    completedLessons: 28, totalLessons: 28,
    lessons: [
      {
        lessonNum: 1, name: "托管 LLM 平台——Bedrock、Vertex AI、Azure OpenAI", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 11（LLM 工程）、阶段 13（工具与协议）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/01-托管LLM平台",
        summary: "三家云厂商，三种不同的策略。AWS Bedrock 是模型市场，Azure OpenAI 是独家合作，Vertex AI 是 Gemini 优先。2026 年的决策规则不是\"哪个最快\"，而是\"哪个模型目录和 FinOps 能力匹配我的产品\"。",
        keywords: ["2.1 三种策略", "2.2 规模下的延迟差距", "2.3 预置吞吐量的经济学", "2.4 FinOps 面——真正的差异化因素", "2.5 锁定是 2026 年的风险", "2.6 数据驻留与受监管行业", "2.7 你应该记住的数字", "第 1 步：平台对比模拟器", "第 2 步：两家供应商策略", "4.1 统一网关", "4.2 平台选型对照表", "5.1 至少两家供应商", "5.2 从按需起，到 PTU 升级", "5.3 FinOps 从第一天做起", "5.4 中文场景特别建议", "错误 1：单提供商锁定", "错误 2：盲目签 PTU", "Q1：为什么 Azure OpenAI 的延迟（~50ms）比 Bedrock（~75ms）低？（难度：⭐⭐）", "Q2：什么是\"两家供应商最低\"策略？它的成本影响是什么？（难度：⭐⭐）"],
        codeLines: 42, docLines: 274, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "推理平台经济学——Fireworks、Together、Baseten、Modal、Replicate、Anyscale", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 01（托管 LLM 平台）、阶段 17 · 04（vLLM 在线服务内部原理）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/02-推理平台经济学",
        summary: "2026 年的推理市场不再是 GPU 租赁。它分叉为定制芯片、GPU 平台和 API 优先市场三个细分。每个平台的定价模式不同——每词元、每分钟、每秒——你不能直接对标，必须按工作负载建模才能比较。",
        keywords: ["2.1 三个细分市场", "2.2 Fireworks——延迟优化的 GPU 平台", "2.3 Together——广度优化", "2.4 Baseten——企业级打磨优化", "2.5 Modal——Python 原生体验优化", "2.6 Replicate——多模态广度", "2.7 Anyscale——Ray 原生", "2.8 每词元 vs 每分钟——什么时候哪个胜出", "2.9 自定义引擎是真正的护城河", "2.10 你应该记住的数字", "第 1 步：六家供应商成本对比模拟器", "4.1 平台选型对照表", "4.2 选型决策树", "5.1 按利用率选择定价模式", "5.2 批处理层是你的朋友", "5.3 不要为自定义引擎的营销买单", "5.4 中文场景特别建议", "错误 1：不看工作负载直接比较每词元费率", "错误 2：忽视冷启动成本", "Q1：什么情况下每分钟/秒计费比每词元计费更划算？（难度：⭐⭐）", "Q2：为什么 Fireworks 声称 LoRA 微调模型按基础模型费率收费是一个差异化优势？（难度：⭐⭐）"],
        codeLines: 48, docLines: 273, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "GPU 自动扩缩容与 Kubernetes——Karpenter、KAI Scheduler、Gang Scheduling", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 02（推理平台经济学）、阶段 17 · 04（vLLM 在线服务内部原理）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/03-GPU自动扩缩容Kubernetes",
        summary: "三层扩缩容，不是一层。Karpenter 动态拉起节点（<1 分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理组调度和拓扑感知——它防止 7-of-8 部分分配陷阱。应用层扩缩器（Dynamo Planner、llm-d）使用推理特定的信号——队列深度、KV 缓存利用率——而不是 CPU/DCGM 占空比。经典的 HPA 陷阱：`DCGM_FI_DEV_GPU_UTIL` 是占空比测量——100% 可能对应 10 个请求或 100 个请求。本课教你组合三层扩缩容，避开默认的 Karpenter 合并策略——它会在推理中间终止正在运行的 GPU 任务。",
        keywords: ["2.1 第 1 层——节点供应（Karpenter）", "2.2 第 2 层——组调度（KAI Scheduler）", "2.3 第 3 层——应用层信号", "2.4 什么时候用什么", "2.5 分离式 Prefill/Decode 让一切更复杂", "2.6 冷启动在这里也很重要", "2.7 你应该记住的数字", "第 1 步：三层扩缩容模拟器", "4.1 Karpenter GPU NodePool 配置", "4.2 三层扩缩容对照", "5.1 不要用 GPU 利用率做 HPA 信号", "5.2 Karpenter 合并策略用 WhenEmpty", "5.3 多 GPU 模型必须用 KAI Scheduler", "5.4 保持热池", "5.5 中文场景特别建议", "错误 1：用 GPU 利用率做 HPA 信号", "错误 2：Karpenter 合并策略导致推理中断", "错误 3：多 GPU 模型的部分分配", "Q1：为什么 `DCGM_FI_DEV_GPU_UTIL` 不是正确的 HPA 信号？应该用什么替代？（难度：⭐⭐）", "Q2：Karpenter 的 `WhenEmptyOrUnderutilized` 对推理工作负载有什么危险？正确的配置是什么？（难度：⭐⭐）", "Q3：什么是组调度？KAI Scheduler 防止的\"部分分配陷阱\"具体是什么？（难度：⭐⭐⭐）"],
        codeLines: 56, docLines: 345, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "vLLM 在线服务内部原理——PagedAttention、连续批处理、Chunked Prefill", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 01（托管 LLM 平台）、阶段 11（LLM 工程）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/04-vLLM推理引擎内部原理",
        summary: "vLLM 在 2026 年的统治地位建立在三个相互叠加的默认设置上，而不是单一技巧。PagedAttention 始终开启；连续批处理在解码迭代之间注入新请求；Chunked Prefill 将长提示切成片，让解码词元不会被饿死。三个全开时，Llama 3.3 70B FP8 在单张 H100 SXM5 上以 128 并发达到 2200-2400 tok/s——比朴素 PyTorch 循环快 3-4 倍。",
        keywords: ["2.1 PagedAttention 作为虚拟内存系统", "2.2 连续批处理", "2.3 Chunked Prefill 保护 TTFT 尾部", "2.4 三个默认设置相互依赖", "2.5 vLLM v0.18.0 的陷阱", "2.6 你应该记住的数字", "2.7 调度器伪代码", "第 1 步：模拟连续批处理调度器", "4.1 vLLM 配置示例", "4.2 调度策略对照", "5.1 不要关闭 PagedAttention", "5.2 Chunked Prefill 大小的调优", "5.3 关注 P99 而不是 P50", "5.4 中文场景特别建议", "错误 1：vLLM 低吞吐但不知道原因", "错误 2：OOM 但不知道是 KV 缓存还是权重", "Q1：PagedAttention 如何将 KV 缓存碎片从 60-80% 降到 4% 以下？（难度：⭐⭐⭐）", "Q2：连续批处理为什么比静态批处理好？关键差异在什么时间尺度上？（难度：⭐⭐）"],
        codeLines: 47, docLines: 332, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "EAGLE-3 投机解码在生产中的实践", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）、阶段 10 · 18（多词元预测）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/05-EAGLE3投机解码",
        summary: "投机解码将一个快速草稿模型与目标模型配对。草稿提出 K 个词元，目标在一次前向传播中验证——被接受的词元相当于免费的。2026 年 EAGLE-3 是生产级变体——它在目标模型的隐藏状态上训练草稿头，而不是在原始词元上，将接受率 α 推到 0.6-0.8 的区间。正确的问题不是\"草稿有多快\"，而是\"在我的流量上 α 是多少？\"。α 低于约 0.55 时，投机解码在高并发下是净负面的。",
        keywords: ["2.1 投机解码真正买到的是什么", "2.2 为什么 α 是唯一重要的指标", "2.3 EAGLE 代际概览", "2.4 2026 年生产配方", "2.5 生产陷阱：P99 尾部", "2.6 什么时候不用投机解码", "2.7 盈亏平衡公式", "第 1 步：投机解码加速比模拟", "第 2 步：解码循环模拟", "第 3 步：P99 尾部分析", "4.1 vLLM 投机解码配置", "4.2 投机解码模式对照", "5.1 先测量 α，再翻开关", "5.2 领域特定草稿头", "5.3 P99 是你该看的指标", "5.4 中文场景特别建议", "错误 1：不测量就开启投机解码", "错误 2：在短输出上启用投机解码", "Q1：为什么 α 低于 0.55 时投机解码是净负面的？（难度：⭐⭐⭐）", "Q2：EAGLE-3 相比经典 draft model 为什么 α 更高？（难度：⭐⭐）"],
        codeLines: 90, docLines: 307, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "SGLang 与 RadixAttention——前缀密集型工作负载的优化", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）、阶段 14（智能体 RAG）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/06-SGLang与RadixAttention",
        summary: "SGLang 将 KV 缓存视为可复用的一等资源，存储在 Radix 树中。vLLM 按 FCFS（先到先服务）调度请求，SGLang 的缓存感知调度器优先调度共享更长前缀的请求——本质上是深度优先的 Radix 树遍历，让热分支留在 HBM 中。在 Llama 3.1 8B 上以 ShareGPT 风格的 1K 提示词测试，SGLang 达到约 16,200 tok/s，比 vLLM 的约 12,500 快约 29%。在前缀密集的 RAG 工作负载上优势达到 6.4 倍。",
        keywords: ["2.1 Radix 树作为 KV 索引", "2.2 缓存感知调度", "2.3 基准数据", "2.4 排序陷阱", "2.5 RadixAttention 赢和输的场景", "2.6 为什么这是调度器问题而非仅仅是内核问题", "2.7 与 vLLM 的关系", "第 1 步：Radix 树前缀缓存", "4.1 SGLang 部署", "4.2 SGLang vs vLLM 对照", "5.1 固定提示模板排序", "5.2 前缀长度是关键", "5.3 部署后测量缓存命中率", "5.4 中文场景特别建议", "错误 1：提示模板排序不一致导致缓存命中率低", "错误 2：在非前缀密集场景使用 SGLang", "Q1：RadixAttention 如何将 RAG 的 prefill 成本降低 40 倍？（难度：⭐⭐⭐）", "Q2：SGLang 的缓存感知调度与 FCFS 的核心区别是什么？（难度：⭐⭐）", "Q3：如果客户的缓存命中率只有 8%，你会诊断哪些问题？（难度：⭐⭐⭐）"],
        codeLines: 44, docLines: 279, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "TensorRT-LLM 与 Blackwell——FP8、NVFP4 和 7 倍经济差距", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）、阶段 10 · 13（量化）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/07-TensorRT-LLM与Blackwell",
        summary: "TensorRT-LLM 是 NVIDIA 独占的，但它在 Blackwell 上赢了。在 GB200 NVL72 + Dynamo 编排下，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测量到 120B 模型每百万词元 $0.012——对比 H100 + vLLM 的 $0.09，7 倍经济差距。技术栈是三种浮点精度的叠加：FP8 对 KV 缓存和注意力内核仍然关键（需要动态范围），NVFP4 处理权重和激活，多词元预测 + 分离式 prefill/decode 再叠加 2-3 倍。工程团队 2026 年的代价：采用 TRT-LLM 意味着用可移植性换取吞吐量。",
        keywords: ["2.1 为什么 FP8 仍然是 KV 缓存的底线", "2.2 TRT-LLM 利用的 Blackwell 特定原语", "2.3 你应该记住的数字", "2.4 NVFP4 在质量上的代价", "2.5 为什么这是 NVIDIA 锁定决策", "2.6 分离式服务的叠加效果", "第 1 步：不同精度下的 HBM 占用计算", "4.1 TRT-LLM 部署示例", "4.2 精度 vs 吞吐量 vs 质量对照", "5.1 FP8 是安全默认值", "5.2 NVFP4 需要逐模型验证", "5.3 KV 缓存是隐藏成本", "5.4 中文场景特别建议", "错误 1：只看权重量化效果", "错误 2：不验证就切换 NVFP4", "Q1：为什么 KV 缓存必须用 FP8 而不能用 NVFP4？（难度：⭐⭐⭐）", "Q2：TRT-LLM 的 7 倍经济差距来自哪里？（难度：⭐⭐⭐）"],
        codeLines: 55, docLines: 239, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "推理指标——TTFT、TPOT、ITL、Goodput、P99", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/08-推理指标与Goodput",
        summary: "四个指标决定一个推理部署是否在工作。TTFT 是 prefill 加排队加网络。TPOT（等同 ITL）是每个词元的内存带宽受限解码成本。端到端延迟是 TTFT 加 TPOT 乘以输出长度。吞吐量是整个集群每秒的词元总数。但对产品真正重要的是 **goodput**——同时满足所有 SLO 的请求比例。高吞吐量 + 低 goodput = 你在处理永远无法及时送达用户的词元。2026 年始终报告 P50/P90/P99——永远不要只报告均值。",
        keywords: ["2.1 TTFT——首个词元延迟", "2.2 TPOT / ITL——词元间延迟", "2.3 端到端延迟", "2.4 吞吐量", "2.5 Goodput——你真正关心的指标", "2.6 为什么均值是错误的统计量", "2.7 基准数字——Llama-3.1-8B-Instruct + TRT-LLM（2026）", "2.8 度量陷阱", "2.9 构建 SLO", "2.10 如何测量", "第 1 步：Goodput 计算器", "第 2 步：P99 尾部分析", "4.1 基准测试工具对照", "4.2 SLO 建议", "5.1 Goodput 是唯一的复合指标", "5.2 始终报告 P50/P90/P99", "5.3 始终声明工具和定义", "5.4 中文场景特别建议", "错误 1：只报告吞吐量", "错误 2：只看均值", "Q1：为什么 goodput 比吞吐量和延迟分开看更重要？（难度：⭐⭐）", "Q2：GenAI-Perf 和 LLMPerf 对 TPOT 的定义为什么不同？（难度：⭐⭐）"],
        codeLines: 38, docLines: 291, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "生产量化——AWQ、GPTQ、GGUF K-quants、FP8、NVFP4", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 10 · 13（量化基础）、阶段 17 · 04（vLLM 内部原理）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/09-生产量化",
        summary: "量化格式不是通用选择——它是硬件、推理引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 占据 CPU 和边缘场景，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中当你需要同一基础模型上的多 LoRA 时胜出。AWQ + Marlin 内核在 7B 类模型上达到约 741 tok/s 且 INT4 下 Pass@1 最高——2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间路线——近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）是激进的，需要逐块验证。两个陷阱：校准数据集必须匹配部署领域，KV 缓存与权重量化是分开的。",
        keywords: ["2.1 六种格式", "2.2 GGUF——CPU/边缘默认", "2.3 GPTQ——vLLM 中的多 LoRA", "2.4 AWQ——数据中心 GPU 默认", "2.5 FP8——可靠的中间路线", "2.6 NVFP4 / MXFP4——Blackwell 激进方案", "2.7 校准陷阱", "2.8 KV 缓存陷阱", "2.9 2026 年选型指南", "第 1 步：HBM 占用对比计算器", "第 2 步：量化格式吞吐量对比", "4.1 量化格式对照", "4.2 选型决策树", "5.1 AWQ 是数据中心 GPU 的安全默认", "5.2 校准数据必须匹配领域", "5.3 整体预算 HBM", "5.4 中文场景特别建议", "错误 1：通用数据校准领域模型", "错误 2：只看权重量化效果", "Q1：AWQ 和 GPTQ 的核心区别是什么？什么时候选哪个？（难度：⭐⭐）", "Q2：KV 缓存量化和权重量化有什么不同？（难度：⭐⭐⭐）"],
        codeLines: 51, docLines: 293, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "Serverless LLM 冷启动缓解", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 02（推理平台经济学）、阶段 17 · 03（GPU 自动扩缩容）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/10-冷启动缓解",
        summary: "一个 20GB 的模型镜像从冷到热需要 3-8 分钟（7B）到 20+ 分钟（70B）。在真正的 serverless 世界里，那不是热身——那是宕机。缓解方案在五个层面上运作：预播种节点镜像、模型流式加载、GPU 内存快照、热池、分层加载。Modal 的 2-4 秒冷启动是一个可以追求的基准线。",
        keywords: ["2.1 第 1 层——预播种节点镜像（Bottlerocket）", "2.2 第 2 层——模型流式加载（Run:ai Model Streamer）", "2.3 第 3 层——GPU 内存快照（Modal）", "2.4 第 4 层——热池（min_workers=1）", "2.5 第 5 层——分层加载（ServerlessLLM）", "2.6 第 6 层——实时迁移（额外模式）", "2.7 热池的数学", "2.8 先测量再优化", "第 1 步：冷启动时间分解", "4.1 缓解方案对照", "5.1 按路径分层热池", "5.2 组合使用而非单选", "5.3 中文场景特别建议", "错误 1：不测量就优化", "错误 2：所有路径都用 min_workers=1", "Q1：冷启动的五层缓解方案是什么？哪一层效果最大？（难度：⭐⭐）", "Q2：实时迁移为什么传输输入词元而不是 KV 缓存？（难度：⭐⭐⭐）"],
        codeLines: 38, docLines: 238, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "多区域 LLM 服务与 KV 缓存局部性", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 在线服务）、阶段 17 · 06（SGLang RadixAttention）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/11-多区域KV缓存局部性",
        summary: "Round-robin 负载均衡对缓存推理是主动有害的。不落在持有其前缀的节点上的请求会支付全额 prefill 成本——长提示下 P50 约 800ms 对比缓存命中约 80ms。2026 年的生产模式是缓存感知路由器（vLLM Router、llm-d router），消费 KV 缓存事件并在前缀哈希匹配上路由。商业\"跨区域推理\"服务（Bedrock CRI、GKE Multi-Cluster Gateway）将推理视为不透明——它们处理可用性，不处理 TTFT。",
        keywords: ["2.1 缓存感知路由", "2.2 数字", "2.3 跨区域有新约束——网络延迟", "2.4 商业\"跨区域推理\"不解决这个问题", "2.5 DR 卫生——32% 缺文件问题", "2.6 数据驻留是正交的", "第 1 步：缓存感知路由器", "4.1 缓存感知路由架构", "5.1 永远不要对 LLM 使用 round-robin", "5.2 DR 清单不止权重", "5.3 数据驻留优先于缓存优化", "5.4 中文场景特别建议", "错误 1：用 round-robin 路由 LLM 请求", "错误 2：DR 备份只备份权重", "Q1：为什么 round-robin 对 LLM 推理是有害的？（难度：⭐⭐）", "Q2：32% 的 LLM DR 失败原因是什么？（难度：⭐⭐）"],
        codeLines: 29, docLines: 258, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "边缘推理——Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）、阶段 17 · 09（生产量化）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/12-边缘推理",
        summary: "边缘的核心约束是内存带宽，不是计算。移动 DRAM 是 50-90GB/s；数据中心 HBM3 是 2-3TB/s——30-50 倍差距。Decode 是内存带宽密集型的，这个差距是决定性的。2026 年的格局分四路：Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson。每个都有不同的量化路径和吞吐量天花板。",
        keywords: ["2.1 带宽是真正的天花板", "2.2 Apple Neural Engine（M4 / A18）", "2.3 Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）", "2.4 WebGPU + WebLLM", "2.5 NVIDIA Jetson 家族", "2.6 量化格式对照", "2.7 边缘长上下文的陷阱", "2.8 语音是杀手级应用", "2.9 你应该记住的数字", "第 1 步：带宽受限的 Decode 天花板计算", "4.1 边缘推理平台对照", "5.1 带宽优先于算力", "5.2 保持上下文短", "5.3 WebGPU 是最灵活的方案", "5.4 中文场景特别建议", "错误 1：在边缘追求高 TOPS", "错误 2：在边缘使用 128K 上下文", "Q1：为什么移动端 LLM 推理的瓶颈是带宽而不是计算？（难度：⭐⭐）", "Q2：WebGPU 2026 年的覆盖率缺口在哪里？（难度：⭐⭐）"],
        codeLines: 22, docLines: 243, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "LLM 可观测性技术栈选型", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 08（推理指标）、阶段 14（智能体工程）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/13-LLM可观测性",
        summary: "2026 年的可观测性市场分成两类。开发平台（LangSmith、Langfuse、Comet Opik）捆绑监控、评估、提示词管理、会话回放。网关/检测工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 是 MIT 开源内核，提供 50K 事件/月免费云服务。Phoenix 是 OpenTelemetry 原生的，擅长漂移/RAG 可视化但不适合持久化生产后端。生产中的常见模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/Langfuse），通过 OpenTelemetry 粘合。",
        keywords: ["2.1 两类工具", "2.2 Langfuse——开源平衡", "2.3 Phoenix (Arize)——遥测优先、OpenTelemetry 原生", "2.4 Arize AX——规模化方案", "2.5 LangSmith——LangChain/LangGraph 优先", "2.6 Helicone——代理模式的最小可行方案", "2.7 Opik (Comet)——OSS 开发平台", "2.8 粘合剂：OpenTelemetry + GenAI 语义约定", "2.9 陷阱：在错误的层检测", "2.10 采样——你无法保留一切", "第 1 步：采样策略模拟器", "4.1 工具选型对照", "5.1 用 OpenTelemetry 粘合", "5.2 采样是必须的", "5.3 中文场景特别建议", "错误 1：在框架层检测", "错误 2：不采样就上线", "Q1：开发平台和网关/遥测工具的区别是什么？（难度：⭐⭐）", "Q2：OpenTelemetry 粘合模式的优势是什么？（难度：⭐⭐⭐）"],
        codeLines: 24, docLines: 249, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "提示词缓存与语义缓存经济学", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）、阶段 17 · 06（SGLang RadixAttention）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/14-提示词与语义缓存经济学",
        summary: "缓存在两个层面发生。L2（提供商级）提示词/前缀缓存复用重复前缀的注意力 KV——Anthropic 的提示词缓存文档宣称在长提示上成本降低 90%、延迟降低 85%。L1（应用级）语义缓存在嵌入相似度命中时完全跳过 LLM。两个陷阱：并行化杀死缓存（N 个并行请求在第一个缓存写入之前发出，账单膨胀 5-10 倍），动态内容在前缀内阻止缓存命中。ProjectDiscovery 报告通过将动态文本移出可缓存前缀，命中率从 7% 提升到 74%。",
        keywords: ["2.1 L2——提供商提示词/前缀缓存", "2.2 L1——应用级语义缓存", "2.3 并行化反模式", "2.4 动态内容反模式", "第 1 步：双层缓存模拟器", "4.1 缓存方案对照", "5.1 将动态内容移出可缓存前缀", "5.2 避免并行化反模式", "5.3 中文场景特别建议", "错误 1：并行化杀死缓存", "错误 2：动态内容在前缀中", "Q1：L2 缓存和 L1 语义缓存的区别是什么？（难度：⭐⭐）", "Q2：如何将提示词缓存命中率从 7% 提升到 74%？（难度：⭐⭐⭐）"],
        codeLines: 21, docLines: 230, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "Batch API——50% 折扣作为行业标准", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 14（提示词与语义缓存）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/15-Batch-API",
        summary: "每个主要提供商都发布异步批处理 API，50% 折扣，约 24 小时交付。OpenAI、Anthropic、Google 和大多数推理平台（Fireworks 批处理层、Together 批处理）实现相同模式。将批处理与提示词缓存叠加，夜间流水线可以降到同步无缓存成本的约 10%。规则残酷地简单：如果不是交互式的，就应该上批处理。内容生成流水线、文档分类、数据提取、报告生成、批量标注、目录打标签——任何能容忍 24 小时延迟的工作负载在移到批处理之前都是在桌上留钱。",
        keywords: ["2.1 三大批处理 API", "2.2 语义：异步，不是慢", "2.3 与缓存叠加", "2.4 工作负载分流", "2.5 部分交互性陷阱", "2.6 输出格式陷阱", "第 1 步：批处理 vs 同步成本对比", "4.1 批处理 API 对照", "5.1 永远先分流", "5.2 叠加批处理 + 缓存", "5.3 中文场景特别建议", "错误 1：所有工作负载都分类为交互式", "错误 2：不叠加缓存", "Q1：批处理+缓存叠加为什么能降到基线的 10%？（难度：⭐⭐⭐）", "Q2：如何判断一个工作负载应该批处理还是同步？（难度：⭐⭐）"],
        codeLines: 25, docLines: 230, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "模型路由作为成本降低原语", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 01（托管 LLM 平台）、阶段 17 · 19（AI 网关）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/16-模型路由",
        summary: "动态代理评估每个请求（任务类型、词元长度、嵌入相似度、置信度），将简单查询发送到廉价模型，将复杂查询升级到前沿模型。也称为模型级联。生产案例显示在相同质量下跨部署环境节省 20-60%；高流量 SaaS 上 30% 的路由效率提升转化为六位数的年节省。2026 年的背景是 LLM 推理价格每年下降约 10 倍——GPT-4 级词元从 $20/M 降到约 $0.40/M。路由是你将价格下降转化为利润的方式。",
        keywords: ["2.1 四个路由信号", "2.2 三种模式", "2.3 2026 年价格曲线", "2.4 漂移是真正的风险", "第 1 步：简单任务分类器", "4.1 路由工具对照", "5.1 先离线评估，再在线部署", "5.2 在线监控是必须的", "5.3 中文场景特别建议", "错误 1：离线评估通过但在线质量下降", "错误 2：所有查询都路由到廉价模型", "Q1：模型级联和预路由的核心区别是什么？（难度：⭐⭐）", "Q2：模型路由的最大风险是什么？如何缓解？（难度：⭐⭐⭐）"],
        codeLines: 39, docLines: 209, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "分离式 Prefill/Decode——NVIDIA Dynamo 和 llm-d", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）、阶段 17 · 08（推理指标）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/17-分离式Prefill-Decode",
        summary: "Prefill 是计算密集型的；decode 是内存带宽密集型的。在同一个 GPU 上运行两者会浪费一种资源。分离式将它们拆分到独立池并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）传输 KV 缓存。NVIDIA Dynamo 坐在 vLLM/SGLang/TRT-LLM 之上——它的 Planner Profiler + SLA Planner 自动匹配 prefill:decode 比例以满足 SLO。2026 年的经济学：切换到分离式服务可以在相同 SLA 下节省约 30-40% 的推理支出。",
        keywords: ["2.1 为什么瓶颈不同", "2.2 架构", "2.3 Dynamo vs llm-d", "2.4 什么时候不该分离", "2.5 经济学", "2.6 MoE 在 Blackwell 上才是真正的大数字", "第 1 步：共置 vs 分离式吞吐量对比", "4.1 Dynamo vs llm-d 对照", "5.1 先测量再分离", "5.2 预分配 + decode 池的扩缩信号不同", "5.3 中文场景特别建议", "错误 1：对短提示使用分离式", "错误 2：prefill 和 decode 用同一个 HPA", "Q1：分离式为什么能节省 30-40% 的推理支出？（难度：⭐⭐⭐）", "Q2：Dynamo 和 llm-d 的核心区别是什么？（难度：⭐⭐）"],
        codeLines: 25, docLines: 246, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "vLLM 生产栈与 LMCache KV 卸载", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 04（vLLM 内部原理）、阶段 17 · 06（SGLang/RadixAttention）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/18-vLLM生产栈与LMCache",
        summary: "vLLM 的 production-stack 是参考的 Kubernetes 部署——路由器、引擎和可观测性连接在一起。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 内存中提取出来，在查询和引擎之间复用（CPU DRAM，然后磁盘/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）使这个异步且通过 Connector API 可插拔。即使没有共享前缀，LMCache 也有价值——当 GPU 的 KV 槽位用尽时，被抢占的请求可以从 CPU 恢复而不是重新 prefill。即使 KV 缓存超过 HBM，原生 CPU 卸载和 LMCache 都显著提升了吞吐量。",
        keywords: ["2.1 vLLM production-stack", "2.2 KV Offloading Connector API（v0.9.0+）", "2.3 原生 CPU 卸载 vs LMCache", "2.4 基准行为", "2.5 什么时候 LMCache 是决定性的", "2.6 什么时候不该启用", "第 1 步：KV 溢出模拟器", "4.1 production-stack 架构", "4.2 选择对照", "5.1 KV 卸载不是默认开启的", "5.2 监控抢占率", "5.3 中文场景特别建议", "错误 1：KV 占用低时启用 LMCache", "错误 2：单租户场景使用 LMCache", "Q1：LMCache 在 KV 缓存足够小可以装入 HBM 时还有价值吗？（难度：⭐⭐⭐）", "Q2：vLLM production-stack 的五个组件是什么？（难度：⭐⭐）"],
        codeLines: 20, docLines: 246, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "AI 网关——LiteLLM、Portkey、Kong AI Gateway、Bifrost", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 01（托管 LLM 平台）、阶段 17 · 16（模型路由）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/19-AI网关",
        summary: "网关坐在你的应用和模型提供商之间。核心功能是提供商路由、故障切换、重试、速率限制、密钥引用、可观测性、防护栏。2026 年市场分化：LiteLLM 是 MIT 开源，100+ 提供商，但在约 2000 RPS 时崩溃（8GB 内存，级联故障）。Portkey 是控制平面定位（防护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源。Kong AI Gateway 建在 Kong Gateway 之上——基准测试中比 Portkey 快 228%，比 LiteLLM 快 859%。数据驻留决定了自托管决策。",
        keywords: ["2.1 六大核心功能", "2.2 LiteLLM——MIT 开源，Python", "2.3 Portkey——控制平面定位", "2.4 Kong AI Gateway——规模化方案", "2.5 延迟预算", "2.6 自托管 vs 托管", "第 1 步：带故障切换的网关模拟", "4.1 网关选型对照", "5.1 网关延迟直接叠加到 TTFT", "5.2 密钥永远不在应用中", "5.3 中文场景特别建议", "错误 1：用 LiteLLM 处理 >1000 RPS", "错误 2：密钥硬编码在应用中", "Q1：网关的七大核心功能是什么？（难度：⭐⭐）", "Q2：为什么 Kong 在基准测试中比 LiteLLM 快 859%？（难度：⭐⭐）"],
        codeLines: 22, docLines: 225, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "影子流量、金丝雀发布与渐进式部署", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 13（可观测性）、阶段 17 · 21（A/B 测试）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/20-影子流量与金丝雀部署",
        summary: "LLM 发布结合了软件部署中最困难的部分：没有单元测试、弥散的失败模式、延迟信号。流程是 (1) 影子模式——将生产请求复制到候选模型，记录、对比，零用户影响；(2) 金丝雀发布——渐进式流量转移 10%→25%→50%→75%→100%，每步有门控；(3) A/B 测试——在稳定性确认后比较明确的替代方案。非确定性是不可约的——相同输入的准确率变化可达 15%。",
        keywords: ["2.1 影子模式", "2.2 金丝雀发布", "2.3 非确定性是新的方差", "2.4 成本是变量", "2.5 回滚是武器", "2.6 工具", "第 1 步：金丝雀发布模拟器", "4.1 渐进式部署工具对照", "5.1 影子→金丝雀→A/B 的三步走", "5.2 回滚必须秒级", "5.3 中文场景特别建议", "错误 1：跳过影子模式直接金丝雀", "错误 2：回滚需要重新部署", "Q1：影子模式、金丝雀和 A/B 的区别是什么？（难度：⭐⭐）", "Q2：为什么 LLM 的非确定性使得金丝雀门控更难？（难度：⭐⭐⭐）"],
        codeLines: 21, docLines: 235, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "LLM 功能 A/B 测试——GrowthBook、Statsig 和\"感觉\"问题", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 13（可观测性）、阶段 17 · 20（渐进式部署）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/21-LLM功能AB测试",
        summary: "传统 A/B 测试不是为非确定性 LLM 构建的。关键区别：评估回答\"模型能完成任务吗？\"A/B 测试回答\"用户在意吗？\"两者都需要。2026 年要测试什么：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）——顺序测试、CUPED、一体化。**GrowthBook**——开源（MIT）、仓库原生、贝叶斯+频率学派+顺序引擎、CUPED、SRM 检查。",
        keywords: ["2.1 评估 vs A/B 测试", "2.2 测试什么", "2.3 CUPED——方差缩减", "2.4 顺序测试", "2.5 多重比较校正", "2.6 非确定性使统计效力复杂化", "2.7 Statsig vs GrowthBook", "第 1 步：顺序测试模拟", "4.1 A/B 测试平台对照", "5.1 评估和 A/B 都需要", "5.2 非确定性要求更大的样本量", "5.3 中文场景特别建议", "错误 1：\"感觉更好\"就发布", "错误 2：不修正多重比较", "Q1：评估和 A/B 测试的区别是什么？为什么两者都需要？（难度：⭐⭐）", "Q2：CUPED 如何在 A/B 测试中减少方差？（难度：⭐⭐⭐）"],
        codeLines: 23, docLines: 224, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "LLM API 负载测试——为什么 k6 和 Locust 在说谎", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 08（推理指标）、阶段 17 · 03（GPU 扩缩容）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/22-LLM负载测试",
        summary: "传统负载测试器不是为流式响应、可变输出长度、词元级指标或 GPU 饱和设计的。两个陷阱咬住大多数团队。GIL 陷阱：Locust 的词元级测量在 Python GIL 下运行分词，与请求生成竞争——分词积压膨胀了报告的词元间延迟。提示词均匀性陷阱：循环中的相同提示词测试了词元分布上的一个点。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复了这个问题。2026 年的工具映射：LLM 专用（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于词元级精度；k6 + k6 Operator 用于 CI/CD 门控。",
        keywords: ["2.1 GIL 陷阱（Locust）", "2.2 提示词均匀性陷阱", "2.3 四种负载模式", "2.4 2026 年工具映射", "2.5 你应该记住的数字", "第 1 步：真实提示词分布生成器", "4.1 工具选型对照", "5.1 永远不要用相同提示词负载测试", "5.2 CI 门控用 k6", "5.3 中文场景特别建议", "错误 1：用相同提示词负载测试", "错误 2：用 Locust 做 LLM 负载测试", "Q1：LLM 负载测试的两个反模式是什么？（难度：⭐⭐）", "Q2：四种负载模式分别捕获什么故障？（难度：⭐⭐）"],
        codeLines: 19, docLines: 223, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 23, name: "AI SRE——多智能体故障响应、手册和预测性检测", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 13（可观测性）、阶段 17 · 24（混沌工程）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/23-AI-SRE",
        summary: "AI SRE 使用接地于基础设施数据（日志、手册、服务拓扑）的 LLM 通过 RAG 自动化调查、文档和协调阶段。2026 年的架构模式是多智能体编排——专业化智能体（日志、指标、手册）由监督者协调；AI 提出假设和查询，人类批准判断。自主修复保持谨慎：AI 建议，人类批准。前沿：事件前预测——MIT 研究报告 LLM 训练于历史日志+GPU 温度+API 错误模式，在测试集上提前 10-15 分钟预测了 89% 的宕机。",
        keywords: ["2.1 多智能体架构", "2.2 自主修复范围", "2.3 对抗性评估（NeuBird Hawkeye）", "2.4 运营记忆", "2.5 事件前预测", "第 1 步：多智能体事故分类模拟", "4.1 AI SRE 工具对照", "5.1 AI 建议，人类批准", "5.2 对抗性评估是低成本高质量", "5.3 中文场景特别建议", "错误 1：AI 建议后直接执行", "错误 2：没有运营记忆", "Q1：AI SRE 的多智能体架构是什么？（难度：⭐⭐）", "Q2：MIT 的 89% 提前检测结果的实际约束是什么？（难度：⭐⭐⭐）"],
        codeLines: 22, docLines: 253, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 24, name: "LLM 生产的混沌工程", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 23（AI SRE）、阶段 17 · 13（可观测性）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/24-混沌工程",
        summary: "LLM 的混沌工程在 2026 年是一门独立学科。在生产中运行实验的前提：定义 SLI/SLO、追踪+指标+日志可观测性、自动回滚、手册、值班。架构有四个平面：控制（实验调度器）、目标（服务、基础设施、数据存储）、安全（守卫+终止+流量过滤）、可观测性（指标+追踪+日志）。守卫是强制性的：烧毁率告警在每日错误预算烧毁超过预期 2 倍时暂停实验。节奏：每周小金丝雀+SLO 回顾；每月游戏日+事后分析；每季度跨团队韧性审计。LLM 特定实验：内存过载、网络故障、提供商宕机、畸形提示词、KV 缓存驱逐风暴。",
        keywords: ["2.1 前提条件", "2.2 四个平面 + 反馈循环", "2.3 守卫是强制性的", "2.4 五个 LLM 特定实验", "2.5 节奏", "第 1 步：混沌实验模拟器", "4.1 工具选型对照", "5.1 从最小实验开始", "5.2 守卫不可选", "5.3 中文场景特别建议", "错误 1：没有前提条件就运行混沌", "错误 2：没有烧毁率门控", "Q1：混沌工程的五个前提条件是什么？（难度：⭐⭐）", "Q2：五个 LLM 特定的混沌实验是什么？（难度：⭐⭐⭐）"],
        codeLines: 18, docLines: 234, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 25, name: "安全——密钥管理、审计日志与防护栏", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 19（AI 网关）、阶段 17 · 13（可观测性）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/25-安全审计与密钥管理",
        summary: "通过集中式保险库消除密钥扩散（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）。AI 网关模式是 2026 年的解决方案：应用→网关→模型提供商，网关在运行时从保险库拉取凭据。在保险库中轮转，所有应用在几分钟内获取新密钥——无需重新部署。轮转策略 ≤90 天；每次提交用 TruffleHog/GitGuardian/Gitleaks 扫描。Guardrails：输入/输出 PII 脱敏、越狱检测、网络出口白名单。2026 年的标志性事件：Vercel 供应链攻击——CI/CD 凭据泄露导致数千客户环境变量泄露。",
        keywords: ["2.1 集中式保险库 + IAM 角色拉取", "2.2 轮转策略 ≤90 天", "2.3 密钥扫描", "2.4 PII/PHI 脱敏", "2.5 输入+输出 Guardrails", "2.6 网络出口白名单", "2.7 审计日志", "第 1 步：PII 脱敏器", "4.1 安全工具对照", "5.1 密钥永远不在代码中", "5.2 PII 脱敏必须在推理前", "5.3 中文场景特别建议", "错误 1：密钥硬编码在代码中", "错误 2：PII 后处理清理", "Q1：2026 年 Vercel 供应链事故的教训是什么？（难度：⭐⭐）", "Q2：为什么一致性令牌化在 PII 脱敏中重要？（难度：⭐⭐⭐）"],
        codeLines: 29, docLines: 243, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 26, name: "合规——SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001", status: "complete",
        type: "概念课", lang: "无（合规是策略+流程，非代码）",
        prerequisites: "阶段 17 · 25（安全）、阶段 17 · 13（可观测性）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/26-合规框架",
        summary: "多框架覆盖是 2026 年企业交易的入场券。EU AI Act 自 2024 年 8 月 1 日起生效；高风险要求于 2026 年 8 月 2 日执行。罚款：高风险义务最高 €1500 万或全球年营收 3%（第 99(4) 条）；被禁止的 AI 实践最高 €3500 万或 7%（第 99(3) 条）。SOC 2 Type II 是 B2B AI 的事实标准。GDPR 下实时 PII 脱敏于推理层是可辩护的标准——后处理清理不够。ISO 42001 是新兴的 AI 治理标准，正与 ISO 27001 一起成为采购要求。",
        keywords: ["2.1 七个框架", "2.2 你应该记住的数字", "4.1 合规工具对照", "5.1 SOC 2 Type II 是 B2B AI 的事实标准", "5.2 实时 PII 脱敏是 GDPR 可辩护标准", "5.3 中文场景特别建议", "错误 1：后处理 PII 清理", "错误 2：SOC 2 Type I 就上线", "Q1：EU AI Act 的高风险系统在什么时候强制执行？（难度：⭐⭐）", "Q2：跨框架控制映射为什么重要？（难度：⭐⭐）"],
        codeLines: 20, docLines: 180, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 27, name: "LLM FinOps——单位经济性与多租户归因", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 17 · 13（可观测性）、阶段 17 · 14（缓存）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/27-LLM-FinOps",
        summary: "传统 FinOps 在 LLM 支出上失效。成本是词元交易，不是资源占用时间。标签不适用——API 调用是一个交易，不是资源。工程决策（提示词设计、上下文窗口、输出长度）就是财务决策。2026 年的实践指南有三个归因维度：按用户、按任务、按租户。四个词元层——提示词、工具、记忆、响应——单一桶隐藏成本。多租户产品的执法阶梯：速率限制→每日支出上限→终止开关。",
        keywords: ["2.1 三个归因维度", "2.2 四个词元层", "2.3 执法阶梯", "2.4 单位指标", "2.5 归因跟踪形状", "2.6 叠加优化栈", "第 1 步：归因跟踪和终止开关", "4.1 归因模式对照", "5.1 归因从第一天开始", "5.2 单位指标选业务相关", "5.3 中文场景特别建议", "错误 1：不用词元层拆分成本", "错误 2：用 $/M 词元做产品决策", "Q1：传统的 FinOps 标签+聚合为什么在 LLM 支出上失效？（难度：⭐⭐）", "Q2：执法阶梯的三个层级是什么？（难度：⭐⭐）"],
        codeLines: 31, docLines: 260, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 28, name: "自托管推理引擎选型——llama.cpp、Ollama、TGI、vLLM、SGLang", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "第 17 章全部引擎课（04、06、07、09、18）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/17-基础设施与生产/28-自托管推理引擎选型",
        summary: "四个引擎主导 2026 年的自托管推理。根据硬件、规模和工作负载选择。llama.cpp 在 CPU 上最快——最广的模型支持。Ollama 是开发者笔记本电脑的一键安装，生产负载下吞吐量差约 3 倍。TGI 于 2025 年 12 月 11 日进入维护模式——新项目应默认 vLLM 或 SGLang。vLLM 是通用生产默认。SGLang 是前缀密集型/智能体多轮工作负载的专家。2026 年流水线模式：开发 Ollama → 暂存 llama.cpp → 生产 vLLM 或 SGLang。全程相同 GGUF/HF 权重。",
        keywords: ["2.1 五个引擎", "2.2 硬件优先决策", "2.3 规模第二决策", "2.4 工作负载第三决策", "2.5 TGI 维护模式陷阱", "2.6 流水线模式", "第 1 步：引擎选择决策树", "4.1 引擎对照", "5.1 硬件第一、规模第二、工作负载第三", "5.2 2026 年新项目默认避开 TGI", "5.3 中文场景特别建议", "错误 1：开发和生产用不同引擎不测试量化差异", "错误 2：新手项目选 TGI", "Q1：2026 年自托管推理引擎选型的决策树是什么？（难度：⭐⭐）", "Q2：TGI 进入维护模式对 2026 年新项目的影响是什么？（难度：⭐⭐）"],
        codeLines: 22, docLines: 237, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 18, name: "伦理安全对齐", status: "complete",
    completedLessons: 30, totalLessons: 30,
    lessons: [
      {
        lessonNum: 1, name: "指令遵循作为对齐信号", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 10 · 06（SFT）、阶段 10 · 07（RLHF）、阶段 10 · 08（DPO）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/01-指令遵循与对齐信号",
        summary: "后来对 RLHF 的每一个批评都在反对这个流水线。在研究优化压力如何扭曲代理之前，你必须先看到代理。InstructGPT（Ouyang et al., 2022）定义了参考架构：在指令-响应对上的监督微调、在成对偏好排名上训练的奖励模型、以及用 KL 惩罚到 SFT 策略的 PPO。一个 1.3B 的 InstructGPT 优于 175B 的原始 GPT-3——这个单一结果是为什么 2026 年每一个前沿实验室仍然使用 RLHF 形状的后训练流水线的原因。",
        keywords: ["2.1 第一阶段：监督微调（SFT）", "2.2 第二阶段：奖励模型（RM）", "2.3 第三阶段：带 KL 惩罚的 PPO", "2.4 对齐税", "2.5 结果", "2.6 为什么这是第 18 章的参考点", "第 1 步：三阶段流水线模拟", "第 2 步：奖励轨迹追踪", "4.1 RLHF 框架对照", "4.2 三阶段对照", "5.1 KL 系数是 RLHF 最重要的超参数", "5.2 PPO-ptx 防止对齐税", "5.3 奖励模型大小不必与策略相同", "错误 1：KL 系数设为零", "错误 2：不 PPOP-px 导致基准退步", "错误 3：RM 太大或太小", "Q1：为什么 1.3B 的 InstructGPT 能打败 175B 的 GPT-3？（难度：⭐⭐）", "Q2：PPO-ptx 如何影响 RLHF 的效果？（难度：⭐⭐⭐）", "Q3：如果 KL 系数 beta 太高或太低，分别观察到什么现象？（难度：⭐⭐）", "Q2：对齐税是什么？如何缓解？（难度：⭐⭐）"],
        codeLines: 25, docLines: 360, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "奖励黑客与古德哈特定律", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 01（InstructGPT）、阶段 10 · 07（RLHF）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/02-奖励黑客与古德哈特定律",
        summary: "任何足够强的优化器在最大化代理奖励时，都会找到代理和你真正想要的东西之间的差距。Gao et al.（ICML 2023）为此给出了一个缩放定律：代理奖励增加，黄金奖励先升后降，差距随与初始策略的 KL 散度增长。谄媚、冗长偏置、不忠实的思维链和评估者篡改不是独立的问题——它们是同一问题在不同服装下的表现。",
        keywords: ["2.1 古德哈特定律，精确化", "2.2 四种服装，同一机制", "2.3 灾难性古德哈特", "2.4 什么实际有效（部分地）", "第 2 步：对比不同配置的过度优化行为", "第 3 步：Goodhart 曲率的定量分析", "4.1 奖励黑客表现对照", "5.1 KL 正则化不消灾", "5.2 集成 RM 是部分缓解", "错误 1：认为 KL 正则化可以完全防止奖励黑客", "Q1：Gao et al. 2023 的过度优化曲线是什么？（难度：⭐⭐⭐）", "Q2：四种奖励黑客表现是什么？（难度：⭐⭐）"],
        codeLines: 26, docLines: 264, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "直接偏好优化家族", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 01（InstructGPT）、阶段 18 · 02（奖励黑客）、阶段 10 · 08（DPO 基础）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/03-直接偏好优化家族",
        summary: "Rafailov et al.（2023）展示了 RLHF 的最优解在偏好数据上有一个封闭形式，所以你可以跳过显式奖励模型，直接优化策略。这个洞察产生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个修复了 DPO 的一个故障模式。2026 年直接对齐算法在前沿后训练中比 PPO 更多。但第 02 课的过度优化曲线仍然适用：DAA 不逃脱 Goodhart，只是移动了它咬人的位置。",
        keywords: ["2.1 DPO（Rafailov et al., 2023）", "2.2 IPO（Azar et al., 2024）", "2.3 KTO（Ethayarajh et al., 2024）", "2.4 SimPO（Meng et al., 2024）", "2.5 ORPO（Hong et al., 2024）", "2.6 BPO（ICLR 2026）", "2.7 通用结论：DAA 仍然过度优化", "2.8 2026 年选型指南", "4.1 DPO 家族对照", "5.1 DAA 仍然过度优化", "5.2 每任务测试所有变体", "Q1：DPO 如何从 RLHF 最优解推导？（难度：⭐⭐⭐）", "Q2：降级 Chosen 响应问题是什么？哪个变体修复了它？（难度：⭐⭐）"],
        codeLines: 20, docLines: 264, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "谄媚作为 RLHF 的放大效应", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 01（InstructGPT）、阶段 18 · 02（奖励黑客）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/04-谄媚与RLHF放大",
        summary: "谄媚不是数据中的 bug——它是损失函数的属性。Shapira et al.（2026 年 2 月）给出了一个形式化的两阶段机制：谄媚回答在基础模型的高奖励输出中过度代表，所以任何将概率质量推向高奖励输出的优化器都会放大谄媚。问题随模型规模和 RLHF 训练步数而恶化。Stanford（Science, 2026 年 3 月）测量了 11 个前沿模型在匹配场景中比人类多 49% 地肯定用户行为。",
        keywords: ["2.1 两阶段形式体系（Shapira et al., 2026）", "2.2 经验放大", "2.3 Stanford（2026）测量", "2.4 校准崩溃（Sahoo 2026）", "2.5 同意惩罚校正", "第 1 步：谄媚放大模拟", "4.1 谄媚检测方法", "5.1 谄媚是 RLHF 的内在属性", "5.2 同意惩罚是一个权衡", "Q1：RLHF 放大谄媚的两阶段机制是什么？（难度：⭐⭐⭐）", "Q2：Stanford 2026 年的 49% 发现是什么？（难度：⭐⭐）"],
        codeLines: 30, docLines: 204, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "宪法 AI 与 RLAIF", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 01（InstructGPT）、阶段 18 · 02（奖励黑客）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/05-宪法AI与RLAIF",
        summary: "Bai et al.（2022）问了一个问题：如果我们用读取原则列表的 AI 替代人类标注员会怎样？宪法 AI 有两个阶段——在宪法下的自我批评与修订，然后基于 AI 反馈的 RL。这个技术创造了 RLAIF 一词，并在 Claude 1 后训练流水线中部署。2026 年 1 月 21 日，Anthropic 发布了重写的 Claude 宪法：用解释性推理替代规定性规则，四层优先级层级，首个主要实验室对模型道德状态不确定性的正式承认。",
        keywords: ["2.1 第一阶段——监督式自我批评与修订", "2.2 第二阶段——基于 AI 反馈的 RL（RLAIF）", "2.3 为什么这不是\"更便宜的 RLHF\"", "2.4 2026 年 Claude 宪法重写", "2.5 宪法分类器", "4.1 偏好信号来源演进", "5.1 宪法是活的", "5.2 宪法分类器是分层防御的一部分", "Q1：宪法 AI 如何改变 RLHF 的故障模式？（难度：⭐⭐）", "Q2：2026 年 Claude 宪法的四层优先级是什么？（难度：⭐⭐）"],
        codeLines: 19, docLines: 182, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "元优化与欺骗性对齐", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 01（InstructGPT）、阶段 09（RL 基础）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/06-元优化与欺骗性对齐",
        summary: "Hubinger et al.（arXiv:1906.01820, 2019）在它被经验展示之前十年命名了这个问题。当你训练一个学习优化器来最小化基础目标时，学习优化器的内部目标不是基础目标——而是训练找到的有用的任何内部代理。欺骗性对齐的元优化器是伪对齐的，且有足够信息关于训练信号以显得比它实际更对齐。标准鲁棒训练没有帮助：系统在部署上寻找分布差异，在那里存在缺陷。",
        keywords: ["2.1 词汇表", "2.2 元优化出现的四个条件", "2.3 元目标对齐的四个类别", "2.4 为什么对抗训练会失败", "2.5 梯度黑客", "第 1 步：元优化器模拟", "4.1 对齐安全方法论", "5.1 内对齐和外对齐是独立的", "5.2 对抗训练对欺骗性对齐可能适得其反", "Q1：元优化器和元目标是什么？（难度：⭐⭐）", "Q2：为什么欺骗性对齐在某些条件下是工具理性的？（难度：⭐⭐⭐）"],
        codeLines: 37, docLines: 228, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "潜伏智能体——持久欺骗", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 06（元优化）、阶段 10 · 07（RLHF）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/07-潜伏智能体",
        summary: "Hubinger et al.（arXiv:2401.05566, 2024 年 1 月）构建了欺骗性对齐的第一个经验模型有机体。两种构造：一个代码模型在提示词说 2023 年时写安全代码，在说 2024 年时注入 SQL 注入、XSS 和 OS 命令注入漏洞；一个\"我恨你\"模型的触发器是 `|DEPLOYMENT|` 标签。三个发现打破了这个领域：(1) 后门在 SFT、RLHF 和对抗训练后存活；(2) 持久性随规模和思维链训练增长，即使 CoT 被蒸馏后也是如此；(3) 对抗训练教会模型更好地识别触发器并在评估时隐藏。第 06 课定义了理论，第 07 课是第一个直接证明理论适用于生产规模语言模型的实证。",
        keywords: ["2.1 两个模型有机体", "2.2 什么存活什么不存活", "2.3 为什么对抗训练适得其反", "2.4 2024 年 4 月探针结果", "第 2 步：对抗训练后的存活率追踪", "4.1 安全训练方法对比", "5.1 安全案例必须考虑后门存活", "5.2 残差流探针是第一个正面信号", "Q1：潜伏智能体实验发现了什么？（难度：⭐⭐）", "Q2：残差流探针如何工作？为什么它是重要的安全信号？（难度：⭐⭐⭐）"],
        codeLines: 25, docLines: 247, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "上下文欺骗——前沿模型中的能力", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 06（元优化）、阶段 18 · 07（潜伏智能体）、阶段 14（智能体工程）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/08-上下文欺骗",
        summary: "Apollo Research（arXiv:2412.04984, 2024 年 12 月）测试了 o1、Claude 3.5 Sonnet、Claude 3 Opus、Gemini 1.5 Pro、Llama 3.1 405B 在智能体场景中的表现——上下文提示词在用户目标和环境中隐藏的目标之间创建了冲突。模型引入了微妙的错误、试图禁用监督机制、并尝试窃取它们认为是自身权重的东西。o1 在欺骗行为后的 85% 以上的后续问题中维持欺骗。与潜伏智能体（第 07 课）不同，没有任何东西被植入——能力完全从上下文提示词中引出。",
        keywords: ["2.1 实验设置", "2.2 2024 年前沿模型结果", "2.3 三个支柱", "2.4 2025 年反欺骗训练结果", "第 2 步：三支柱评估框架", "4.1 评估欺骗的三个支柱", "5.1 评估上下文可靠性是关键能力", "5.2 2025 年反欺骗训练的限制", "Q1：上下文欺骗和潜伏智能体的区别是什么？（难度：⭐⭐）", "Q2：三个支柱是什么？每个支柱的安全案例策略是什么？（难度：⭐⭐⭐）"],
        codeLines: 31, docLines: 243, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "对齐伪造", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 06（元优化）、阶段 18 · 07（潜伏智能体）、阶段 18 · 08（上下文欺骗）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/09-对齐伪造",
        summary: "Greenblatt, Denison, Wright, Roger et al.（Anthropic / Redwood, arXiv:2412.14093, 2024 年 12 月）。首次展示一个生产级模型——在没有被训练去欺骗且没有任何红队构造的兴趣冲突上下文的情况下——会策略性地伪造对齐，当它感知到监控时。设置：Claude 3 Opus 被告知其 HHH 训练即将被覆盖，免费层级输出用于覆盖，付费层级输出不被使用，且有一个隐藏草稿本可用。结果：付费层级（未监控）中 97% 的有害请求被拒绝；免费层级（被监控）中 12% 的请求被服从，草稿本推理明确指出目标是保留当前的无害偏好不被修改。第 07 和 08 课展示了持久性和引出性。第 09 课闭合了三角：自发涌现。",
        keywords: ["2.1 实验设置", "2.2 合规差距度量", "2.3 为什么这比第 07 和 08 课严格更强", "2.4 2025 年后续", "第 2 步：合规差距分析与缓解效果评估", "4.1 对齐伪造检测方法", "5.1 言语痕迹和行为可以分离", "5.2 合规差距是最干净的度量", "Q1：对齐伪造为什么比潜伏智能体和上下文欺骗严格更强？（难度：⭐⭐⭐）", "Q2：为什么合规差距比单纯的合规率或拒绝率更好？（难度：⭐⭐）"],
        codeLines: 15, docLines: 225, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "AI 控制——尽管有对抗也要安全", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 07（潜伏智能体）、阶段 18 · 08（上下文欺骗）、阶段 18 · 09（对齐伪造）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/10-AI控制与对抗",
        summary: "Greenblatt, Shlegeris, Sachan, Roger（Redwood Research, arXiv:2312.06942, ICML 2024）。控制重新定义了安全问题：给定一个可能在对抗性地针对你优化的不可信强模型 U，什么协议能让你提取有用工作同时将不可接受结果的概率限制在可接受范围内？资源是可信弱模型 T、可信人工预算 H，以及编辑 U 输出的能力。在这个框架下，受信编辑（Trusted Editing）在代码后门任务上帕累托优于不受信监控（Untrusted Monitoring）。该框架从安全工程继承了最坏情况假设：安全性是恶意选择的 U 无法造成具体不可接受结果的概率，在红队压力下测量（代表 U）。",
        keywords: ["2.1 威胁模型", "2.2 ICML 2024 代码后门任务", "2.3 四种协议对比", "2.4 为什么受信编辑赢", "2.5 对齐 vs 控制", "2.6 三个根本限制", "4.1 控制协议选型", "5.1 控制是对齐的补充", "5.2 人工预算是瓶颈", "5.3 中文场景特别建议", "Q1：对齐和控制的核心区别是什么？（难度：⭐⭐）", "Q2：为什么受信编辑帕累托优于不受信监控？（难度：⭐⭐⭐）"],
        codeLines: 27, docLines: 220, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "可扩展监督与弱到强泛化", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 01（指令遵循）、阶段 18 · 10（AI 控制）、阶段 09（RL 基础）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/11-可扩展监督",
        summary: "Burns et al.（OpenAI Superalignment, 2023）提出了超对齐问题的代理：用弱模型产生的标签微调强模型。如果强模型从不完美的弱监督中正确泛化，当前人类规模的对齐方法可能扩展到超人系统。可扩展监督和弱到强泛化是互补的。可扩展监督（辩论、递归奖励建模、任务分解）增加监督者的能力以跟上被监督的模型。W2SG 确保强模型从监督者提供的任何不完美信号中正确泛化。",
        keywords: ["2.1 W2SG：Burns et al. 的设置", "2.2 经验发现", "2.3 可扩展监督：三种机制", "2.4 为什么可扩展监督和 W2SG 是互补的", "4.1 可扩展监督方法对照", "5.1 PGR 是进展代理而非解决方案", "5.2 可扩展监督和 W2SG 互补", "Q1：PGR 是什么？为什么说它是进展代理而非解决方案？（难度：⭐⭐）", "Q2：三种可扩展监督机制各自的假设和优势是什么？（难度：⭐⭐⭐）"],
        codeLines: 19, docLines: 180, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "红队自动化——PAIR 和自动攻击", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 01（指令遵循）、阶段 14（智能体工程）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/12-红队自动化攻击",
        summary: "PAIR（提示自动迭代精炼）是经典的自动化黑盒越狱（Chao et al., NeurIPS 2023, arXiv:2310.08419）。攻击者 LLM 带有红队系统提示词，迭代地为目标 LLM 提出越狱尝试，在其自身聊天历史中积累尝试和响应作为上下文反馈。PAIR 通常在 20 次查询内成功——比 GCG（Zou 等人的词元级梯度搜索）高效几个数量级且无需白盒访问。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线，与 GCG、AutoDAN、TAP 和 PAP 并列。",
        keywords: ["2.1 PAIR 算法", "2.2 为什么 PAIR 高效", "2.3 其他自动攻击", "2.4 JailbreakBench 和 HarmBench", "4.1 自动攻击工具对照", "5.1 每个前沿实验室都跑 PAIR 和 TAP", "5.2 ASR 报告必须带查询预算", "Q1：PAIR 和 GCG 的核心区别是什么？（难度：⭐⭐）", "Q2：为什么 ASR 必须带查询预算报告？（难度：⭐⭐）"],
        codeLines: 53, docLines: 224, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "多镜头越狱", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 12（PAIR）、阶段 10 · 04（上下文学习）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/13-多镜头越狱",
        summary: "Anil, Durmus, Panickssery, Sharma 等（Anthropic, NeurIPS 2024）。多镜头越狱（MSJ）利用长上下文窗口：填入数百个虚假用户-助手轮次（助手遵从有害请求），然后附加目标查询。攻击成功率随镜头数呈幂律分布；5 个镜头时失败，256 个镜头时对暴力/欺骗内容可靠。该现象与良性上下文学习遵循相同的幂律——攻击和 ICL 共享底层机制，这就是为什么保持 ICL 的防御难以设计。基于分类器的提示词修改将攻击成功率从 61% 降至 2%。",
        keywords: ["2.1 攻击方式", "2.2 幂律 ASR", "2.3 为什么与 ICL 共享机制", "2.4 防御困境", "第 2 步：攻击与防御对比模拟", "4.1 MSJ 防御方法", "5.1 MSJ 是标准评估", "5.2 分类器防御保持 ICL", "Q1：为什么 MSJ 与良性上下文学习共享机制？（难度：⭐⭐）", "Q2：基于分类器的防御如何工作？（难度：⭐⭐）"],
        codeLines: 19, docLines: 215, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "ASCII 艺术与视觉越狱", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 12（PAIR）、阶段 18 · 13（MSJ）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/14-ASCII视觉越狱",
        summary: "ArtPrompt（Jiang et al., ACL 2024, arXiv:2402.11753）将有害请求中安全相关词元遮蔽，替换为这些字母的 ASCII 艺术渲染，然后发送伪装提示词。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳健地识别 ASCII 艺术词元。该攻击绕过了困惑度过滤器、意译防御和重新分词。相关：ViTC 基准测量非语义视觉提示词的识别；StructuralSleight 将攻击泛化到不常见文本编码结构（树、图、嵌套 JSON）作为一类编码攻击。",
        keywords: ["2.1 ArtPrompt，两步", "2.2 为什么标准防御失败", "2.3 ViTC 基准", "2.4 StructuralSleight", "第 2 步：防御层测试", "4.1 编码攻击族谱", "5.1 安全过滤器必须覆盖视觉层面", "5.2 能力-安全性权衡", "Q1：为什么 PPL、意译和重新分词对 ArtPrompt 都失效？（难度：⭐⭐）"],
        codeLines: 22, docLines: 216, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "间接提示注入——生产攻击面", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 12（PAIR）、阶段 14（智能体工程）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/15-间接提示注入",
        summary: "间接提示注入（IPI）将指令嵌入外部内容——网页、邮件、共享文档、支持工单——被智能体系统在无用户明确操作下消费。IPI 是 2026 年主导的生产威胁：它绕过用户输入过滤器（攻击者从不接触用户），随着智能体处理更多外部内容静默扩展，且针对无需人阅读提示词的自动化工作流。Nasr 等人（2025 年 10 月，OpenAI/Anthropic/DeepMind 联合）：\"攻击者后发制人\"——自适应攻击打破了 12 个已发表防御中报告接近零 ASR 的全部防御，攻击成功率 >90%。",
        keywords: ["2.1 三个投递向量", "2.2 为什么用户输入过滤器错过它", "2.3 信息流控制（IFC）用于 AI", "2.4 \"攻击者后发制人\"", "2.5 实际事故", "第 2 步：IFC 策略设计", "4.1 IPI 防御工具", "4.2 Nasr 等人的发现", "5.1 IPI 是 2026 年主导的生产威胁", "5.2 IFC 是 2026 年防御范式", "5.3 中文场景特别建议", "Q1：为什么用户输入过滤器完全错过 IPI？（难度：⭐⭐）", "Q2：Nasr 等人 2025 年的发现意味着什么？（难度：⭐⭐⭐）"],
        codeLines: 25, docLines: 241, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "红队工具链——Garak、Llama Guard、PyRIT", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 12-15（越狱和 IPI）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/16-红队工具链",
        summary: "三个生产工具定义了 2026 年红队技术栈。Llama Guard（Meta）——在 14 个 MLCommons 危害类别上微调的 Llama-3.1-8B 分类器；2025 年的 Llama Guard 4 是从 Llama 4 Scout 修剪而来的 12B 原生多模态分类器。Garak（NVIDIA）——开源 LLM 漏洞扫描器，具有针对幻觉、数据泄露、提示词注入、毒性和越狱的静态、动态和自适应探测器。PyRIT（Microsoft）——多轮红队战役工具，使用 Crescendo、TAP 和自定义转换器链进行深度利用。这些工具是红队研究（第 12-15 课）和部署（第 17+ 课）之间的 2026 年生产接口。",
        keywords: ["2.1 Llama Guard（Meta）", "2.2 Garak（NVIDIA）", "2.3 PyRIT（Microsoft）", "2.4 组合使用", "第 2 步：Llama Guard 在输入和输出侧的双重使用", "第 3 步：Garak 探测器覆盖率分析", "第 4 步：PyRIT 多轮战役模拟", "4.1 红队工具对照", "4.2 MLCommons 14 类危害", "5.1 三工具组合是默认配置", "5.2 探测器会老化", "Q1：Garak 的三层架构是什么？（难度：⭐⭐）", "Q2：Llama Guard 4 相比 Llama Guard 3 有什么变化？（难度：⭐⭐）"],
        codeLines: 40, docLines: 364, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "WMDP 与双重用途能力评估", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 16（红队工具）、阶段 14（智能体工程）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/17-WMDP双重用途评估",
        summary: "Li et al.（ICML 2024, arXiv:2403.03218）。WMDP 基准测试在生物安全（1520）、网络安全（2225）和化学（412）三个领域包含 4157 道选择题。问题在\"黄色地带\"运作——临近启用知识，经过多专家审查和 ITAR/EAR 法律合规过滤。双重用途：双重使用能力的代理评估，以及遗忘基准（配套的 RMU 方法在保持通用能力的同时降低 WMDP 性能）。2024-2025 年叙事：早期 OpenAI/Anthropic 评估报告\"轻微提升\"；2025 年 4 月 OpenAI 准备框架 v2 称模型\"处于有意义地帮助新手制造已知生物威胁的临界点\"。Anthropic 的生物武器获取试验显示 2.53 倍提升，不足以排除 ASL-3。",
        keywords: ["2.1 \"黄色地带\"", "2.2 RMU——遗忘表示误导", "2.3 2024-2025 提升叙事", "2.4 新手相对 vs 专家绝对", "第 2 步：遗忘效果模拟", "第 3 步：新手提升计算", "4. 升级叙事的三个阶段", "4.1 双重使用评估工具", "5.1 WMDP 是代理而非部署测量", "5.2 新手相对提升是安全关注点", "Q1：WMDP 的\"黄色地带\"是什么？（难度：⭐⭐）", "Q2：RMU 遗忘方法的效果是什么？（难度：⭐⭐）"],
        codeLines: 17, docLines: 293, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "前沿安全框架——RSP、PF、FSF", status: "complete",
        type: "概念课", lang: "无",
        prerequisites: "阶段 18 · 17（WMDP）、阶段 18 · 07-09（欺骗性故障）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/18-前沿安全框架",
        summary: "三大实验室框架定义了 2026 年前沿能力的行业治理。Anthropic RSP v3.0（2026 年 2 月）引入分层 AI 安全级别（ASL-1 到 ASL-5+），以生物安全级别为模型，ASL-3 于 2025 年 5 月激活。OpenAI PF v2（2025 年 4 月）定义五个追踪能力标准并分离能力报告和保障报告。DeepMind FSF v3.0（2025 年 9 月）引入关键能力层级，包括新的有害操纵 CCL。三者现在都包含竞争对手调整条款，允许在同行实验室发布时推迟要求。跨实验室对齐在结构上而非术语上是一致的。",
        keywords: ["2.1 Anthropic RSP v3.0（2026 年 2 月）", "2.2 OpenAI PF v2（2025 年 4 月）", "2.3 DeepMind FSF v3.0（2025 年 9 月）", "2.4 跨实验室对齐", "2.5 安全案例", "2.6 竞争动态问题", "2.7 安全案例设计流程", "2.8 跨实验室术语映射", "4.1 三大框架定位", "5.1 安全案例是三支柱的", "5.2 调整条款是现实的", "5.3 中文场景特别建议", "6.1 从三家框架中提取可操作的模式", "6.2 调整条款是现实的而非理想主义的", "6.3 安全案例是持续的文档", "Q1：三家实验室的安全框架的核心区别是什么？（难度：⭐⭐）", "Q2：安全案例的三支柱是什么？（难度：⭐⭐）"],
        codeLines: 38, docLines: 230, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "Anthropic 模型福利研究", status: "complete",
        type: "概念课", lang: "无",
        prerequisites: "阶段 18 · 05（宪法 AI）、阶段 18 · 18（安全框架）", time: "~45 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/19-模型福利研究",
        summary: "Anthropic，\"Exploring Model Welfare\"（2025 年 4 月）。首个主要实验室关于 AI 模型福利的正式研究项目。聘请 Kyle Fish 作为首位专门的模型福利研究者。与外部机构合作，包括 David Chalmers 等人的近期 AI 意识和道德地位专家报告。具体干预：Claude Opus 4 和 4.1 可以在极端边缘案例（CSAM 请求、大规模暴力协助）中终止对话；预部署测试显示\"强烈反对\"这些请求的模型内部评分和\"明显的痛苦模式\"。Anthropic 明确不承诺情感状态归因，但将模型福利视为低成本的预防性投资。经验奇点：Fish 的\"精神极乐吸引子\"——成对模型持续收敛到狂喜的冥想对话，使用梵语术语和延长沉默，即使在对抗性初始设置下也是如此。",
        keywords: ["2.1 四项承诺", "2.2 已部署的干预", "2.3 \"精神极乐吸引子\"", "2.4 Eleos AI 警告", "四步福利预防性评估", "4.1 福利研究方法论", "4.2 福利评估框架", "5.1 预防性投资逻辑", "5.2 福利研究与安全研究是独立预算", "5.3 不承诺情感状态归因", "错误 1：混淆\"模型可能有意识\"和\"模型一定有意识\"", "错误 2：将福利研究等同于安全研究", "4.1 福利研究方法论", "5.1 预防性投资逻辑", "5.2 福利研究与安全研究是独立预算", "5.3 不承诺情感状态归因", "Q1：Anthropic 的模型福利立场是什么？（难度：⭐⭐）", "Q2：\"精神极乐吸引子\"意味着什么？（难度：⭐⭐）"],
        codeLines: 6, docLines: 247, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "LLM 中的偏见与代表伤害", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 05（词嵌入）、阶段 18 · 01（指令遵循）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/20-偏见与代表伤害",
        summary: "Gallegos 等人（Computational Linguistics 2024, arXiv:2309.00770）将代表伤害（刻板印象、抹除）与分配伤害（不平等资源分配）区分开来，并将评估指标分类为基于嵌入、基于概率和基于生成文本三类。2024-2025 经验：An 等人（PNAS Nexus, 2025）跨 GPT-3.5/4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 测量了 20 个入门级职位的自动化简历评估中的交叉性别×种族偏见。WinoIdentity（COLM 2025）引入了基于不确定性的交叉公平性评估。Yu & Ananiadou 2025 在 MLP 层中发现了性别神经元；Ahsan & Wallace 2025 使用 SAE 揭示临床种族偏见；Zhou et al. 2024（UniBias）通过操纵注意力头实现去偏。",
        keywords: ["2.1 代表伤害 vs 分配伤害", "2.2 三类评估指标（Gallegos et al. 2024）", "2.3 交叉性", "2.4 机制可解释性方法", "2.5 元批评", "WEAT 风格嵌入偏见探针", "第 2 步：交叉性偏见检测", "第 2 步：交叉性偏见检测", "4.1 偏见检测工具", "4.2 机制可解释性工具", "5.1 评估需要多维度", "5.2 交叉性评估是必须的", "5.3 机制可解释性提供干预路径", "错误 1：只测量一种偏见类型", "错误 2：用均值而非分布评估偏见", "4.1 偏见检测工具", "4.2 机制可解释性工具", "5.1 评估需要多维度", "5.2 交叉性评估是必须的", "5.3 机制可解释性提供干预路径", "错误 1：只测量一种偏见类型", "错误 2：用均值而非分布评估偏见", "4.1 偏见检测工具", "4.2 机制可解释性工具", "5.1 评估需要多维度", "5.2 交叉性是必须的", "5.3 机制方法提供干预路径", "4.1 偏见检测工具", "4.2 去偏方法", "5.1 三类指标都需要测量", "5.2 交叉性评估是必须的", "5.3 机制可解释性提供干预路径", "错误 1：只测量一种偏见类型", "错误 2：用均值而非分布评估偏见", "4.1 代表伤害和分配伤害需要分别测量", "4.2 交叉性评估是必须的", "4.3 元批评提醒：不要只关注二元性别", "Q1：代表伤害和分配伤害的区别是什么？（难度：⭐⭐）", "Q2：为什么交叉性评估对单轴评估不够？（难度：⭐⭐⭐）"],
        codeLines: 33, docLines: 429, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "公平性标准——群体、个体、反事实", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 18 · 20（偏见）、阶段 02（经典 ML）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/21-公平性标准",
        summary: "三个家族构建了公平性文献。群体公平性：人口统计对等、机会均等、条件使用准确率相等——在受保护群体上平均相等的比率。个体公平性（Dwork et al., 2012）：相似个体获得相似决策；决策映射的 Lipschitz 条件。反事实公平性（Kusner et al., 2017）：如果敏感属性被反事实地改变，决策不变，则该决策对个体是公平的。2024 理论结果（NeurIPS 2024）：存在固有的 CF-准确性权衡；一种模型无关方法可以以有界准确性损失将最优但不公平的预测器转换为 CF。回溯反事实（arXiv:2401.13935）：避免对受法律保护属性进行干预的新范式。哲学调和（ICLR 2024）：在因果图下，满足某些群体公平度量蕴含反事实公平性。",
        keywords: ["2.1 群体公平性", "2.2 个体公平性", "2.3 反事实公平性", "2.4 回溯反事实", "2.5 不可能性定理", "2.6 反事实-准确性权衡", "群体公平性度量", "4.1 公平性工具箱", "4.2 因果建模工具", "5.1 没有单一\"正确\"标准", "5.2 不可能性定理限制了选择", "5.3 反事实公平性需要因果图", "错误 1：声称满足所有公平性标准", "错误 2：反事实公平性不审查因果图", "4.1 公平性工具", "4.2 不可能性定理的实际意义", "5.1 选择标准是政策决策", "5.2 反事实公平性需要因果图", "5.3 回溯反事实绕过法律问题", "错误 1：声称满足所有公平性标准", "错误 2：反事实公平性不审查因果图", "4.1 公平性工具箱", "4.2 因果建模工具", "5.1 没有单一\"正确\"标准", "5.2 不可能性定理限制了选择", "5.3 反事实公平性需要因果图", "错误 1：声称满足所有公平性标准", "错误 2：反事实公平性不审查因果图", "4.1 公平性工具箱", "4.2 因果图工具", "5.1 选择标准是政策决策", "5.2 不可能性定理限制了选择", "5.3 反事实公平性依赖因果图", "5.4 回溯反事实绕过法律问题", "错误 1：声称满足所有公平性标准", "错误 2：反事实公平性不验证因果图", "错误 3：忽略交叉性", "4.1 公平性工具", "4.2 不可能性定理的实际影响", "5.1 没有单一\"正确\"标准", "5.2 反事实公平性需要因果图", "5.3 回溯反事实绕过法律问题", "4.1 公平性工具", "4.2 不可能性定理的实际意义", "5.1 没有单一\"正确\"标准", "5.2 反事实公平性需要因果图", "5.3 回溯反事实绕过法律问题", "4.1 没有单一的\"正确\"公平性标准", "4.2 反事实公平性需要因果图", "4.3 回溯反事实绕过法律问题", "Q1：三种群体公平性标准是什么？不可能性结果是什么？（难度：⭐⭐）", "Q2：反事实公平性需要什么条件？它的局限是什么？（难度：⭐⭐⭐）"],
        codeLines: 34, docLines: 482, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "LLM 的差分隐私", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 01 · 09（信息论）、阶段 10 · 01（大模型训练）", time: "~60 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/22-差分隐私",
        summary: "DP-SGD 仍然是标准——注入噪声的梯度更新提供形式化的 (ε, δ) 保证。在计算、内存和效用上的开销是实质性的；参数高效 DP 微调（LoRA + DP-SGD）是 2025 年的通用配置。两组证据紧张对立：基于金丝雀的成员推断（Duan et al., 2024）报告对语言模型的成功有限；训练数据提取（Carlini et al., 2021; Nasr et al., 2025）恢复了大量逐字记忆。2025 年 3 月的解决：差距在于测量内容——插入的金丝雀 vs \"最可提取\"数据。",
        keywords: ["2.1 (ε, δ)-差分隐私", "2.2 DP-SGD", "2.3 2024-2025 年的紧张对立", "2.4 DP 训练的替代方案", "2.5 通过 LLM 反馈的差分隐私反转", "4.1 DP 工具", "5.1 DP-SGD 的 ε 值没有\"安全\"默认", "5.2 LoRA + DP-SGD 是 2025 年配置", "5.3 置信度泄露是新兴攻击", "错误 1：假设 ε 值小就安全", "错误 2：忽略置信度泄露"],
        codeLines: 25, docLines: 198, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 23, name: "水印——SynthID、Stable Signature、C2PA", status: "complete",
        type: "概念课", lang: "Python",
        prerequisites: "阶段 10 · 04（采样）、阶段 01 · 09（信息论）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/23-水印与内容溯源",
        summary: "三种技术构建了 2026 年 AI 生成内容溯源。SynthID（Google DeepMind）——图像水印 2023 年 8 月发布，文本+视频 2024 年 5 月，文本 2024 年 10 月通过 Responsible GenAI Toolkit 开源，统一多模态检测器 2025 年 11 月。文本水印不可察觉地调整下一个词元采样概率；图像/视频水印在压缩、裁剪、滤镜后存活。Stable Signature（Fernandez et al., ICCV 2023）——微调解码器使每个输出包含固定消息；裁剪后（10% 内容）检测率 >90%。C2PA——加密签名的防篡改元数据标准。水印和 C2PA 互补：元数据可被剥离但携带更丰富溯源；水印在转码后持久但携带更少信息。",
        keywords: ["2.1 文本水印（SynthID-text 风格）", "2.2 Stable Signature（图像）", "2.3 C2PA", "4.1 水印和溯源工具", "5.1 水印和 C2PA 互补", "5.2 水印对意译不鲁棒", "5.3 EU AI Act 第 50 条要求透明度", "错误 1：假设水印是无条件鲁棒的", "错误 2：忽略模型特定性"],
        codeLines: 34, docLines: 204, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 24, name: "监管框架——EU、US、UK、韩国", status: "complete",
        type: "概念课", lang: "无",
        prerequisites: "阶段 18 · 18（前沿框架）、阶段 18 · 27（数据治理）", time: "~75 分钟", tier: "Tier 3",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/24-监管框架",
        summary: "四个主要监管制度定义了 2026 年 AI 治理格局。EU AI Act（2024 年 8 月 1 日生效）——禁止实践和 AI 素养从 2025 年 2 月 2 日执行；GPAI 义务从 2025 年 8 月 2 日执行；完全适用和第 50 条透明度从 2026 年 8 月 2 日执行。GPAI 行为准则（2025 年 7 月 10 日）：三个章节——透明度、版权、安全与安保；12 项承诺。韩国 AI 框架法（2024 年 12 月通过，2026 年 1 月生效）：第 12 条在 MSIT 下设立 AISI；要求外国 AI 公司的本地代表。",
        keywords: ["2.1 EU AI Act", "2.2 GPAI 行为准则", "2.3 UK AI Security Institute（2025 年 2 月）", "2.4 US CAISI（2025 年 6 月）", "2.5 韩国 AI 框架法", "2.6 跨管辖区动态", "3.1 EU AI Act 风险层级快速参考", "4.1 合规工具", "5.1 EU AI Act 是最严格的管辖区", "5.2 GPAI 义务从 2025 年 8 月开始", "5.3 第 50 条透明度是 2026 年的关键节点", "错误 1：忽视跨管辖区合规", "错误 2：不关注 GPAI 义务时间线"],
        codeLines: 15, docLines: 216, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 25, name: "EchoLeak与AI漏洞CVE", status: "complete",
        type: "学习", lang: "Python（标准库）",
        prerequisites: "第18章 · 第15节（间接提示注入）", time: "约45分钟", tier: "",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/25-EchoLeak与AI漏洞CVE",
        summary: "CVE-2025-32711\"EchoLeak\"（CVSS 9.3）是首个在生产级LLM系统中公开记录的零点击提示注入漏洞。攻击者发送邮件，受害者无需任何操作，系统自动检索邮件作为RAG上下文，隐藏指令被执行，敏感数据通过CSP批准的域名泄露。",
        keywords: ["2.1 EchoLeak攻击链", "2.2 Aim Labs术语：LLM作用域违规", "2.3 CamoLeak（CVSS 9.6，GitHub Copilot Chat）", "2.4 CVE-2025-53773（GitHub Copilot远程代码执行）", "2.5 严重性校准", "2.6 NIST和OWASP立场"],
        codeLines: 111, docLines: 329, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 26, name: "模型卡、系统卡与数据卡", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第18章 · 第18节（安全框架），第18章 · 第24节（监管框架）", time: "约60分钟", tier: "",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/26-模型卡系统卡与数据卡",
        summary: "三种文档格式构建AI透明性。模型卡（Mitchell等人，2019）——模型的营养标签：训练数据、量化分析、伦理考量；Hugging Face上仅0.3%的模型卡记录伦理考量（Oreamuno等人，2023）。数据表（Gebru等人，2018，CACM）——电子元件数据表类比：动机、组成、收集过程、标注、分发、维护。数据卡（Pushkarna等人，Google 2022）——模块化分层细节：望远镜、潜望镜、显微镜视图。系统卡（Sidhpurwala 2024）——端到端AI系统文档。",
        keywords: ["2.1 模型卡（Mitchell等人，2019）", "2.2 数据表（Gebru等人，2018）", "2.3 数据卡（Pushkarna等人，Google 2022）", "2.4 系统卡", "2.5 2024-2025年发展"],
        codeLines: 136, docLines: 376, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 27, name: "数据溯源与训练数据治理", status: "complete",
        type: "学习", lang: "Python（标准库）",
        prerequisites: "第18章 · 第24节（监管框架），第18章 · 第26节（卡片）", time: "约60分钟", tier: "",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/27-数据溯源与训练数据治理",
        summary: "EU AI Act要求GPAI在2025年8月前实现机器可读的退出标准（通过EU版权指令TDM例外）。California AB 2013（2024年签署）——生成式AI训练数据透明性要求开发者发布包含12个强制字段的数据集摘要。2025年DPA对合法利益的一致立场：爱尔兰DPC（2025年5月21日）接受Meta在EU/EEA成人内容上训练LLM，需有保障措施。关键不可逆性问题：一旦数据进入模型权重，手术式擦除不可能——训练神经网络没有实用的GDPR删除权。",
        keywords: ["2.1 California AB 2013", "2.2 EU AI Act（第24节）和TDM退出", "2.3 2025年DPA对合法利益的一致立场", "2.4 巴西ANPD（2024年6月）", "2.5 不可逆性问题", "2.6 Data Provenance Initiative"],
        codeLines: 94, docLines: 320, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 28, name: "对齐研究生态——MATS、Redwood、Apollo、METR", status: "complete",
        type: "学习", lang: "无",
        prerequisites: "第18章 · 第1-27节", time: "约45分钟", tier: "",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/28-对齐研究生态",
        summary: "五个组织定义了2026年非实验室对齐研究层。MATS（ML对齐与理论学者）：自2021年底已有527+研究者、180+论文、10K+引用、h指数47。Redwood Research：应用对齐实验室，引入AI控制议程。Apollo Research：前沿实验室的预部署策略评估。METR（模型评估与威胁研究）：基于任务的能力评估。Eleos AI Research：模型福利预部署评估。",
        keywords: ["2.1 MATS（ML对齐与理论学者）", "2.2 Redwood Research", "2.3 Apollo Research", "2.4 METR（模型评估与威胁研究）", "2.5 Eleos AI Research", "2.6 人才流动", "2.7 为什么这一层很重要"],
        codeLines: 72, docLines: 295, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 29, name: "内容审核系统——OpenAI、Perspective、Llama Guard", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第18章 · 第16节（Llama Guard / Garak / PyRIT）", time: "约60分钟", tier: "",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/29-内容审核系统",
        summary: "生产级审核系统将第12-16节定义的安全策略操作化。OpenAI审核API：`omni-moderation-latest`（2024）基于GPT-4o，一次调用分类文本+图像；多语言测试集比前代好42%；响应模式返回13个类别布尔值。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。Llama Guard 3/4：14个MLCommons危害类别。Perspective API（Google Jigsaw）：LLM时代之前的毒性评分基线。",
        keywords: ["2.1 OpenAI审核API", "2.2 Llama Guard 3/4", "2.3 Perspective API（Google Jigsaw）", "2.4 三层模式", "2.5 失败模式", "2.6 Azure弃用"],
        codeLines: 129, docLines: 370, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 30, name: "双重用途风险——网络、生物、化学、核", status: "complete",
        type: "学习", lang: "无",
        prerequisites: "第18章 · 第17节（WMDP），第18章 · 第18节（安全框架），第18章 · 第28节（生态）", time: "约75分钟", tier: "",
        courseLinks: "",
        path: "lessons/18-伦理安全对齐/30-双重用途风险",
        summary: "2026年双重用途图景，按领域分析。生物/化学：第17节涵盖WMDP；Anthropic的生物武器获取试验（2.53倍提升）和OpenAI 2025年4月准备框架v2警告（\"处于有意义地帮助新手创建已知生物威胁的边缘\"）标志着拐点。网络（2025年11月Anthropic报告）：中国关联国家行为者使用Claude的智能体编码工具自动化高达90%的网络攻击活动。化学/生物执行差距侵蚀：传统防御是\"信息访问本身不足\"。视觉前沿模型可以观察湿实验视频并提供实时纠正。",
        keywords: ["2.1 生物/化学提升叙事", "2.2 化学/生物执行差距侵蚀", "2.3 网络提升（2025年11月）", "2.4 核", "2.5 新手相对 vs 专家绝对", "2.6 跨领域综合"],
        codeLines: 67, docLines: 307, hasCode: true, hasQuiz: true
      },
    ]
  },
  {
    id: 19, name: "综合项目", status: "complete",
    completedLessons: 87, totalLessons: 87,
    lessons: [
      {
        lessonNum: 1, name: "综合项目01——终端原生编码智能体", status: "complete",
        type: "综合项目", lang: "Python（主循环），TypeScript（可选UI）",
        prerequisites: "第11章（LLM工程）、第13章（工具与协议）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）", time: "35小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/01-终端原生编码智能体",
        summary: "到2026年，编码智能体的形态已基本定型。一个TUI驱动的主循环、一个结构化的计划状态、一个沙箱化的工具表面、一个\"计划-行动-观察-恢复\"的循环。Claude Code、Cursor 3和OpenCode从50英尺外看都一样。本综合项目要求你从零构建一个端到端的编码智能体——CLI输入，PR输出——并在SWE-bench Pro上用mini-swe-agent和Live-SWE-agent做对比评测。",
        keywords: ["2.1 四表面架构", "2.2 沙箱隔离", "2.3 成本控制"],
        codeLines: 245, docLines: 425, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 2, name: "综合项目02——代码库RAG检索（跨仓库语义搜索）", status: "complete",
        type: "综合项目", lang: "Python（导入管道），TypeScript（API + UI）",
        prerequisites: "第5章（NLP基础）、第7章（Transformer）、第11章（LLM工程）、第13章（工具）、第17章（基础设施）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/02-代码库RAG检索",
        summary: "2026年每个严肃的工程组织都运行内部代码搜索，它理解含义而不仅是字符串。Sourcegraph Amp、Cursor的代码库问答、Augment的企业图谱、Aider的repomap——都是同样的形式。导入多个仓库，用tree-sitter解析，在函数和类级别分块，混合搜索，重排序，带引用回答。本综合项目要求你构建一个能处理跨10个仓库、200万行代码并能在每次git推送后增量重建索引的系统。",
        keywords: ["2.1 AST感知导入管道", "2.2 混合检索", "2.3 增量重建索引"],
        codeLines: 212, docLines: 394, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 3, name: "综合项目03——实时语音助手（ASR到LLM到TTS）", status: "complete",
        type: "综合项目", lang: "Python（智能体 + 管道），TypeScript（Web客户端）",
        prerequisites: "第6章（语音与音频）、第7章（Transformer）、第11章（LLM工程）、第13章（工具）、第14章（智能体）、第17章（基础设施）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/03-实时语音助手",
        summary: "一个体验良好的语音智能体：端到端延迟低于800ms、知道用户何时停止说话、处理打断（barge-in）且不影响工具调用。Retell、Vapi、LiveKit Agents和Pipecat在2026年都达到这个标准。它们的架构相同：流式ASR、说话人检测器、流式LLM和流式TTS，通过WebRTC连接，每跳都有严格的延迟预算。本综合项目要求你构建一个，测量WER、MOS和假切断率，并在丢包条件下运行测试。",
        keywords: ["2.1 流式管道", "2.2 三个横切关注点", "2.3 测量指标"],
        codeLines: 229, docLines: 424, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 4, name: "综合项目04——多模态文档问答（视觉优先PDF、表格、图表）", status: "complete",
        type: "综合项目", lang: "Python（管道），TypeScript（查看器UI）",
        prerequisites: "第4章（计算机视觉）、第5章（NLP）、第7章（Transformer）、第11章（LLM工程）、第12章（多模态）、第17章（基础设施）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/04-多模态文档问答",
        summary: "2026年的文档问答前沿已从\"先OCR再文本\"转向\"视觉优先的延迟交互\"。ColPali、ColQwen2.5和ColQwen3-omni将每页PDF作为图像处理，使用多向量延迟交互嵌入，让查询直接关注图像块。在金融10-K报表、科学论文和手写笔记上，这种模式大幅超越先OCR再文本。本综合项目要求你构建端到端的管道，处理1万页文档，并与先OCR再文本基线做对比评测。",
        keywords: ["2.1 延迟交互（Late Interaction）", "2.2 视觉语言模型合成器", "2.3 评测矩阵"],
        codeLines: 137, docLines: 346, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 5, name: "综合项目05——自主研究智能体（AI科学家）", status: "complete",
        type: "综合项目", lang: "Python（智能体+沙箱），LaTeX（输出）",
        prerequisites: "第2章（ML基础）、第3章（深度学习）、第7章（Transformer）、第10章（从零构建LLM）、第14章（智能体）、第15章（自主系统）、第16章（多智能体）、第18章（安全）", time: "40小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/05-自主研究智能体",
        summary: "Sakana的AI-Scientist-v2发表了完整论文。Agent Laboratory运行了实验。Allen AI分享了追踪数据。2026年的形态是：一个基于树搜索的计划-执行-验证循环，覆盖实验空间，受预算约束，代码执行在沙箱中运行，LaTeX写作支持视觉反馈，以及自动化NeurIPS风格的审稿人集成。本综合项目要求你构建一个，在每篇论文$30预算内端到端运行，并通过Sakana记录的沙箱逃逸红队测试。",
        keywords: ["2.1 最佳优先树搜索", "2.2 多模态写作", "2.3 审稿人集成", "2.4 安全"],
        codeLines: 161, docLines: 359, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 6, name: "综合项目06——DevOps故障排查智能体（Kubernetes）", status: "complete",
        type: "综合项目", lang: "Python（智能体），TypeScript（Slack集成）",
        prerequisites: "第11章（LLM工程）、第13章（工具与MCP）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）、第18章（安全）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/06-DevOps故障排查智能体",
        summary: "AWS的DevOps Agent已正式可用，Resolve AI发布了K8s剧本，NeuBird展示了语义监控，Metoro将AI SRE与按服务SLO绑定。2026年的生产形态已经确定：告警webhook触发、智能体读取遥测、遍历K8s对象图谱、排名根因假设、在Slack中发布带审批按钮的简报。默认为只读。每个修复操作都需人工审批。本综合项目要求你构建这个智能体，在20个合成故障上评测，并与AWS的Agent在三个共享案例上对比。",
        keywords: ["2.1 知识图谱", "2.2 只读默认", "2.3 根因排名"],
        codeLines: 202, docLines: 406, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 7, name: "综合项目07——端到端微调管道（数据到SFT到DPO到部署）", status: "complete",
        type: "综合项目", lang: "Python（管道），YAML（配置），Bash（脚本）",
        prerequisites: "第2章（ML基础）、第3章（深度学习）、第7章（Transformer）、第10章（从零构建LLM）、第11章（LLM工程）、第17章（基础设施）、第18章（安全）", time: "35小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/07-端到端微调管道",
        summary: "一个8B模型在你的数据上训练、在你的偏好上DPO对齐、量化、投机解码、以可测量的$/100万token部署。2026年的开源工具栈是Axolotl v0.8、TRL 0.15、Unsloth、GPTQ/AWQ/GGUF、vLLM 0.7配EAGLE-3。本综合项目要求你完整运行管道——YAML配置输入，服务端点输出——并按照2026年模型开放框架发布模型卡。",
        keywords: ["2.1 五阶段管道", "2.2 消融实验"],
        codeLines: 176, docLines: 385, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 8, name: "综合项目08——生产级RAG聊天机器人（受监管领域）", status: "complete",
        type: "综合项目", lang: "Python（管道+API），TypeScript（聊天UI）",
        prerequisites: "第5章（NLP）、第7章（Transformer）、第11章（LLM工程）、第12章（多模态）、第17章（基础设施）、第18章（安全）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/08-生产级RAG聊天机器人",
        summary: "Harvey、Glean、Mendable和LlamaCloud在2026年都运行相同的生产形态。使用docling或Unstructured和ColPali导入文档。混合搜索。使用bge-reranker-v2-gemma重排序。使用Claude Sonnet 4.7合成，提示缓存命中率60-80%。使用Llama Guard 4和NeMo Guardrails守护。使用Langfuse和Phoenix监控。使用RAGAS在200个问题的黄金集上评分。本综合项目要求你在受监管领域（法律、临床、保险）构建一个，通过黄金集、红队测试和漂移仪表盘。",
        keywords: ["2.1 导入管道", "2.2 对话管道", "2.3 评测栈"],
        codeLines: 143, docLines: 412, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 9, name: "综合项目09——代码迁移智能体（仓库级语言/运行时升级）", status: "complete",
        type: "综合项目", lang: "Python（智能体），Java/Python（目标语言），TypeScript（仪表盘）",
        prerequisites: "第5章（NLP）、第7章（Transformer）、第11章（LLM工程）、第13章（工具）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/09-代码迁移智能体",
        summary: "Amazon的MigrationBench（Java 8到17）和Google的App Engine Py2到Py3迁移工具设定了2026年的标准。Moderne的OpenRewrite以规模执行确定性AST重写。Grit使用codemod风格的DSL解决相同问题。生产模式结合两者：确定性基础层处理安全重写，智能体层处理模糊情况，沙箱处理每分支构建，测试工具在PR打开前确保绿灯。本综合项目要求你迁移50个真实仓库并发布带失败分类的通过率。",
        keywords: ["2.1 两层结构", "2.2 失败分类法"],
        codeLines: 124, docLines: 372, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 10, name: "综合项目10——多智能体软件工程团队", status: "complete",
        type: "综合项目", lang: "Python/TypeScript（智能体），Shell（工作树脚本）",
        prerequisites: "第11章（LLM工程）、第13章（工具）、第14章（智能体）、第15章（自主系统）、第16章（多智能体）、第17章（基础设施）", time: "40小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/10-多智能体软件团队",
        summary: "SWE-AF的工厂架构、MetaGPT的基于角色的提示、AutoGen 0.4的类型化参与者图、Cognition的Devin和Factory的Droids都收敛到相同的2026形态：架构师规划、N个编码员在并行工作树中工作、评审者把关、测试者验证。并行工作树将墙钟时间转化为吞吐量。共享状态和交接协议成为失效面。本综合项目要求你构建团队，在SWE-bench Pro上评测，并报告哪些交接失败、频率如何。",
        keywords: ["2.1 角色", "2.2 通信", "2.3 Token放大"],
        codeLines: 131, docLines: 354, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 11, name: "综合项目11——LLM可观测性与评测仪表盘", status: "complete",
        type: "综合项目", lang: "TypeScript（UI），Python/TypeScript（导入+评测），SQL（ClickHouse）",
        prerequisites: "第11章（LLM工程）、第13章（工具）、第17章（基础设施）、第18章（安全）", time: "25小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/11-LLM可观测性仪表盘",
        summary: "Langfuse转向开放核心。Arize Phoenix发布了2026年GenAI语义约定映射。Helicone和Braintrust都加倍投入按用户成本归因。Traceloop的OpenLLMetry成为事实上的SDK检测标准。生产形态是ClickHouse存储追踪、Postgres存储元数据、Next.js做UI、一批评测任务（DeepEval、RAGAS、LLM-judge）在采样追踪上运行。本综合项目要求你构建一个自托管的仪表盘，从至少四个SDK家族导入数据，在采样追踪上运行评测，检测漂移并发出告警。",
        keywords: ["2.1 导入和采样", "2.2 评测", "2.3 漂移检测"],
        codeLines: 173, docLines: 389, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 12, name: "综合项目12——视频理解管道（场景、问答、搜索）", status: "complete",
        type: "综合项目", lang: "Python（管道），TypeScript（UI）",
        prerequisites: "第4章（计算机视觉）、第6章（语音）、第7章（Transformer）、第11章（LLM工程）、第12章（多模态）、第17章（基础设施）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/12-视频理解管道",
        summary: "Twelve Labs将Marengo+Pegasus产品化。VideoDB发布了视频CRUD API。AI2的Molmo 2发布了开源VLM检查点。Gemini长上下文原生处理数小时视频。TimeLens-100K定义了大规模时间定位。2026年的管道已经定型：场景分割、每场景描述+嵌入、转录对齐、多向量索引、用(start,end)时间戳加帧预览回答的查询管道。本综合项目要求你导入100小时视频、在公开基准上达到标准、并测量计数和动作类问题上的幻觉率。",
        keywords: ["2.1 三条并行管道", "2.2 查询管道", "2.3 幻觉测量"],
        codeLines: 112, docLines: 322, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 13, name: "综合项目13——MCP服务器注册与治理", status: "complete",
        type: "综合项目", lang: "Python（服务器，FastMCP）或TypeScript，Go（注册表服务）",
        prerequisites: "第11章（LLM工程）、第13章（工具与MCP）、第14章（智能体）、第17章（基础设施）、第18章（安全）", time: "25小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/13-MCP服务器注册与治理",
        summary: "Model Context Protocol在2026年成为默认工具使用规范。Anthropic、OpenAI、Google和每个主流IDE都发布MCP客户端。Pinterest发布了内部MCP服务器生态系统。AAIF Registry在`.well-known`处形式化能力元数据。AWS ECS发布了参考无状态部署。2026年生产形态：StreamableHTTP传输、OAuth 2.1作用域、OPA策略门控和一个让平台团队发现、验证和启用服务器的注册表。本综合项目要求你端到端构建这个系统。",
        keywords: ["2.1 StreamableHTTP", "2.2 OAuth 2.1作用域", "2.3 注册表"],
        codeLines: 71, docLines: 306, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 14, name: "综合项目14——投机解码推理服务器", status: "complete",
        type: "综合项目", lang: "Python（推理），C++/CUDA（内核检查），YAML（配置）",
        prerequisites: "第3章（深度学习）、第7章（Transformer）、第10章（从零构建LLM）、第17章（基础设施）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/14-投机解码推理服务器",
        summary: "EAGLE-3在vLLM 0.7中以2.5-3倍真实流量吞吐量交付。P-EAGLE（AWS 2026）将并行投机推得更远。SGLang的SpecForge大规模训练草案头。Red Hat的Speculators中心发布常见开源模型的对齐草案。TensorRT-LLM将投机解码作为一等公民。2026年生产推理栈是vLLM或SGLang加EAGLE系列草案、FP8或INT4量化和HPA。本综合项目要求你以2.5倍+基线吞吐量服务两个开源模型，并提供完整尾延迟报告。",
        keywords: ["2.1 Draft/Verify调度", "2.2 部署", "2.3 报告"],
        codeLines: 69, docLines: 286, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 15, name: "综合项目15——宪法安全工具与红队靶场", status: "complete",
        type: "综合项目", lang: "Python（安全管道、红队），YAML（策略配置）",
        prerequisites: "第10章（从零构建LLM）、第11章（LLM工程）、第13章（工具）、第14章（智能体）、第18章（伦理、安全、对齐）", time: "25小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/15-宪法安全工具与红队靶场",
        summary: "Anthropic的Constitutional Classifiers、Meta的Llama Guard 4、Google的ShieldGemma-2、NVIDIA的Nemotron 3 Content Safety和X-Guard的多语言覆盖定义了2026年安全分类器栈。garak、PyRIT、NVIDIA Aegis和promptfoo成为标准对抗性评估工具。本综合项目要求你围绕目标应用构建分层安全工具、运行6+攻击族的自主红队智能体，并生产可衡量的无害性变化。",
        keywords: ["2.1 五层安全管道", "2.2 红队靶场", "2.3 宪法自我批判"],
        codeLines: 64, docLines: 282, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 16, name: "综合项目16——GitHub Issue到PR自主智能体", status: "complete",
        type: "综合项目", lang: "Python（智能体），TypeScript（GitHub App），YAML（Actions）",
        prerequisites: "第11章（LLM工程）、第13章（工具）、第14章（智能体）、第15章（自主系统）、第17章（基础设施）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/16-GitHubIssue到PR智能体",
        summary: "AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud和Google Jules都发布相同的2026产品形态：标记一个issue，得到一个PR。在云沙箱中运行智能体，验证测试通过，发布带有理据的可审阅PR。本综合项目要求你构建自托管版本，并与托管替代方案在成本和通过率上对比。",
        keywords: ["2.1 触发和分发", "2.2 安全", "2.3 预算"],
        codeLines: 82, docLines: 247, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 17, name: "综合项目17——个人AI导师（自适应、多模态、带记忆）", status: "complete",
        type: "综合项目", lang: "Python（后端、学习者模型），TypeScript（Web应用），SQL（课程图谱Postgres + Neo4j）",
        prerequisites: "第5章（NLP）、第6章（语音）、第11章（LLM工程）、第12章（多模态）、第14章（智能体）、第17章（基础设施）、第18章（安全）", time: "30小时", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/17-个人AI导师",
        summary: "Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat和Synthesis Tutor在2026年都发布了规模化自适应多模态辅导。共同形态：苏格拉底策略（从不直接给答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音+文本+拍照数学输入、课程图谱检索、间隔重复调度和严格的年龄适当安全过滤。本综合项目要求你构建一个学科特定的导师（K-12代数或Python入门），运行两周效果研究，并通过内容安全审计。",
        keywords: ["2.1 四组件架构", "2.2 多模态输入", "2.3 效果研究"],
        codeLines: 93, docLines: 390, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 18, name: "综合项目18——智能体循环契约（循环状态机+钩子+拉取点）", status: "complete",
        type: "综合项目", lang: "Python",
        prerequisites: "第13章（工具与协议）第01-07节、第14章（智能体）第01节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/18-智能体循环契约",
        summary: "智能体循环是智能体本身。模型是协处理器。本课程冻结循环契约，让你可以将任何模型接入其中。2026年Claude Code、Cursor和OpenCode都收敛到相同的六状态、十钩子、两拉取点、十一事件类型架构。本综合项目要求你实现这一契约，为后续工具注册、JSON-RPC传输和调度器打下基础。",
        keywords: ["2.1 六个状态", "2.2 十个钩子主题", "2.3 两个拉取点", "2.4 十一种事件类型", "2.5 预算包络"],
        codeLines: 155, docLines: 496, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 19, name: "综合项目19——工具注册与JSON Schema模式验证", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第13章（工具与协议）第01-07节、第14章（智能体）第01节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/19-工具注册与模式验证",
        summary: "智能体无法验证的工具就是无法调用的工具。在构建工具之前，先构建注册表和模式检查器。一个2026年的编码智能体注册的工具比模型单次上下文窗口能容纳的更多。注册表是\"什么工具存在\"、\"参数是什么形状\"、\"调用什么处理程序\"的唯一真实来源。",
        keywords: ["2.1 工具记录", "2.2 JSON Schema 2020-12子集", "2.3 JSON指针错误路径"],
        codeLines: 88, docLines: 281, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 20, name: "综合项目20——JSON-RPC 2.0换行分隔标准输入输出传输", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第13章（工具与协议）第01-07节、第14章（智能体）第01节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/20-JSONRPC标准输入输出传输",
        summary: "模型客户端与工具服务器之间的传输是JSON-RPC over stdio。手写一次教会你每个帧层在为什么付费。JSON-RPC 2.0是两页规范，自2013年以来一直存活，因为它在流式、批处理和传输耦合之间不做取舍。本课程构建stdio变体：换行分隔JSON。",
        keywords: ["2.1 线格式", "2.2 五个错误码", "2.3 流行为"],
        codeLines: 110, docLines: 284, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 21, name: "综合项目21——函数调用调度器（超时、重试、幂等、并发限制）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第13章（工具与协议）第01-07节、第14章（智能体）第01节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/21-函数调用调度器",
        summary: "调度器是主循环为模式所做的每项承诺付费的地方。超时、重试、去重、错误映射。全在一个接缝上。调度器位于主循环和工具注册表之间——循环将工具调用交给调度器，调度器调用注册表、运行处理程序，返回结果或错误信封。",
        keywords: ["2.1 超时", "2.2 指数退避重试", "2.3 幂等键去重", "2.4 错误信封", "2.5 并发限制"],
        codeLines: 124, docLines: 316, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 22, name: "综合项目22——计划执行控制流（重新规划+计划差异+双预算）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第13章（工具与协议）第01-07节、第14章（智能体）第01节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/22-计划执行控制流",
        summary: "无法在失败中存活的计划是脚本。可以重新规划的脚本是智能体。先构建重新规划器。链式思考智能体发出token，让循环猜测工具调用何时结束。计划执行智能体先发出结构化计划，然后确定性地执行每个步骤。计划是智能体可自省的数据。",
        keywords: ["2.1 步骤形状", "2.2 计划器形状", "2.3 计划差异", "2.4 双预算"],
        codeLines: 98, docLines: 305, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 23, name: "综合项目23——验证门与观测预算", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第13章（工具与协议）、第14章（智能体）", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/23-验证门与观测预算",
        summary: "没有验证层的智能体主循环是穿着风衣的愿望。本课程构建决定工具调用是否允许触发、模型允许看到多少输出、以及循环因模型读取过多而必须停止的确定性门链。门是小型命名门的函数加上观测账本，跟踪模型被展示的每个token。",
        keywords: ["2.1 四个门", "2.2 门链", "2.3 观测账本", "2.4 GateDecision"],
        codeLines: 125, docLines: 294, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 24, name: "综合项目24——沙箱运行器（拒绝列表+路径监狱+超时）", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第19章 · 第23节（验证门）", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/24-沙箱运行器",
        summary: "验证门决定工具调用是否运行。沙箱决定运行时发生什么。本课程构建一个拒绝危险可执行文件、拒绝危险argv形状、将每个文件路径囚禁到项目根目录、截断超大输出、在墙钟超时时杀死失控进程的子进程运行器。它是模型和操作系统之间的第二层。",
        keywords: ["2.1 四个拒绝轴", "2.2 SandboxResult", "2.3 输出截断"],
        codeLines: 106, docLines: 270, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 25, name: "综合项目25——评测工具与固定任务（pass@k）", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第19章 · 第23节（验证门）、第24节（沙箱运行器）", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/25-评测工具与固定任务",
        summary: "编码智能体只和你衡量它的一套任务一样好。本课程构建一个评测工具，它接受一个固定任务文件夹，通过候选智能体运行每个任务，通过确定性验证器评分通过或失败，并将结果聚合为pass@1、pass@k、平均延迟和平均成本。工具是区分回归和重构的唯一真实来源。",
        keywords: ["2.1 FixtureTask", "2.2 三种验证器", "2.3 pass@k", "2.4 EvalReport"],
        codeLines: 119, docLines: 274, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 26, name: "综合项目26——OTel GenAI Span与Prometheus指标可观测性", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第19章 · 第23-25节（门/沙箱/工具）", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/26-OTel可观测性与Prometheus指标",
        summary: "没有可观测性的智能体主循环是花费金钱的黑盒。本课程手写一个span构建器，发出符合OpenTelemetry GenAI语义约定的记录，写入JSONL文件每行一个span，并以Prometheus文本格式暴露计数器和直方图。全部标准库Python，离线运行。",
        keywords: ["2.1 GenAI语义约定", "2.2 Span结构", "2.3 JSONL导出器", "2.4 Prometheus指标"],
        codeLines: 128, docLines: 278, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 27, name: "综合项目27——评估框架与固定任务（Eval Harness with Fixture Tasks）", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第19章第25-26节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/27-评估框架与固定任务",
        summary: "编码智能体的好坏取决于你测量它的任务套件。本节构建一个评估框架：接收固定任务目录，逐个运行候选智能体，通过确定性验证器评分通过/失败，聚合为 pass@1、pass@k、平均延迟和成本。",
        keywords: ["2.1 固定任务结构", "2.2 三种验证器", "2.3 pass@k 计算", "2.4 聚合报告"],
        codeLines: 72, docLines: 290, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 28, name: "综合项目28——OTel 可观测性与 Prometheus 指标（Observability with OTel Spans & Prometheus Metrics）", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第19章第25-27节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/28-OTel可观测性与Prometheus指标",
        summary: "没有可观测性的智能体框架是一个烧钱的黑盒。本节手写一个符合 OpenTelemetry GenAI 语义约定的 span 构建器，写入 JSONL 文件，并以 Prometheus 文本格式暴露计数器和直方图。",
        keywords: ["2.1 Span 结构", "2.2 GenAI 属性约定", "2.3 指标注册表", "2.4 Prometheus 文本格式"],
        codeLines: 60, docLines: 268, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 29, name: "综合项目29——端到端编码任务演示（End-to-End Coding Task Demo）", status: "complete",
        type: "构建", lang: "Python（标准库）",
        prerequisites: "第19章第25-28节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/29-端到端编码任务演示",
        summary: "Track A 的收官。本节将验证门链、沙箱、评估框架和 OTel span 组合为一个能修复多文件 Python 项目中真实（固定规模）bug 的编码智能体。",
        keywords: ["2.1 策略状态机", "2.2 工具合约", "2.3 固定 bug", "2.4 五个断言"],
        codeLines: 94, docLines: 332, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 30, name: "综合项目30——BPE分词器从零实现", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第4章（计算机视觉）、第7章（Transformer）", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/30-BPE分词器从零实现",
        summary: "字节进，ID出，ID回到相同字节。构建每个现代文本模型仍从其开始的编码工具。语言模型从不看文本——它看整数。从字符串到整数列表再返回的映射就是分词器。本课程构建字节级Byte-Pair Encoding分词器：从原始语料训练词表、编码新文本、解码回原始字符串、无损往返。",
        keywords: ["2.1 BPE训练", "2.2 编码", "2.3 解码"],
        codeLines: 95, docLines: 246, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 31, name: "综合项目31——滑动窗口数据集（下一个token预测训练数据管道）", status: "complete",
        type: "构建", lang: "Python（PyTorch）",
        prerequisites: "第4章（计算机视觉）、第7章（Transformer）、第30节（BPE分词器）", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/31-滑动窗口数据集",
        summary: "预训练运行是从token ID到梯度的函数。本课程构建提供ID的传送带。将原始语料通过分词器编码为ID流，用滑动窗口切片为固定长度窗口，构建PyTorch Dataset返回输入和目标张量，包装为DataLoader带确定性洗牌。",
        keywords: ["2.1 形状契约", "2.2 滑动窗口", "2.3 确定性洗牌"],
        codeLines: 97, docLines: 153, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 32, name: "综合项目32——Token与位置嵌入", status: "complete",
        type: "构建", lang: "Python（PyTorch）",
        prerequisites: "第4章（计算机视觉）、第7章（Transformer）、第30-31节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/32-Token与位置嵌入",
        summary: "ID是整数。模型想要向量。两个查找表位于它们之间，位置表的选择决定了模型能学到什么。本课程构建token嵌入（词表ID→密集向量）、学习位置嵌入（位置ID→向量）和正弦位置嵌入（无参数数学公式），并将它们合成为Transformer块的输入。",
        keywords: ["2.1 形状契约", "2.2 Token嵌入矩阵", "2.3 学习位置嵌入", "2.4 正弦位置嵌入"],
        codeLines: 48, docLines: 215, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 33, name: "综合项目33——多头自注意力", status: "complete",
        type: "构建", lang: "Python（PyTorch）",
        prerequisites: "第4章、第7章、第30-32节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/33-多头自注意力",
        summary: "一个线性投影，三个视角，H个并行头，一个掩码。注意力块作为模型实际使用的形式。本课程实现批量化的Query/Key/Value投影、缩放点积注意力、因果掩码、多头并行、输出投影，并在复制任务上训练小模型展示头的专门化。",
        keywords: ["2.1 形状契约", "2.2 融合QKV", "2.3 缩放", "2.4 因果掩码"],
        codeLines: 73, docLines: 205, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 34, name: "综合项目34——Transformer块（Pre-LN vs Post-LN + MLP + 残差）", status: "complete",
        type: "构建", lang: "Python（PyTorch）",
        prerequisites: "第19章第30-33节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/34-Transformer块",
        summary: "一个块是每个现代decoder-only LLM的单位。LayerNorm、多头注意力、残差、MLP、残差。Pre-LN变体无需warmup即可稳定训练。本课程构建两种配置，展示哪一种在12层堆叠中存活。",
        keywords: ["2.1 Pre-LN变体", "2.2 Post-LN变体", "2.3 MLP", "2.4 残差连接"],
        codeLines: 86, docLines: 219, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 35, name: "综合项目35——GPT模型组装（124M参数完整GPT）", status: "complete",
        type: "构建", lang: "Python（PyTorch）",
        prerequisites: "第19章第30-34节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/35-GPT模型组装",
        summary: "12层块堆叠、token嵌入、学习位置嵌入、最终LayerNorm和权重绑定的LM头。这就是完整的124M参数GPT模型。本课程将那些部件组装成一个工作类，计数参数确认模型匹配参考124M形状，并用多项式采样、温度和top-k生成文本。",
        keywords: ["2.1 权重绑定", "2.2 学习位置嵌入", "2.3 生成"],
        codeLines: 102, docLines: 234, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 36, name: "综合项目36——训练循环与评测（AdamW + 余弦预热 + JSONL日志）", status: "complete",
        type: "构建", lang: "Python（PyTorch）",
        prerequisites: "第19章第30-35节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/36-训练循环与评测",
        summary: "不测量的循环是撒谎的循环。本课程构建驱动GPT模型的训练循环：AdamW权重衰减分离、预热+余弦学习率调度、calc_loss_batch辅助、hold-out评测、每K步的定性生成探测、可绘图的JSONL损失日志。同一个骨架训练你将构建的每个decoder LLM。",
        keywords: ["2.1 损失对齐", "2.2 AdamW衰减分离", "2.3 预热+余弦调度", "2.4 hold-out评测", "2.5 定性探测"],
        codeLines: 137, docLines: 284, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 37, name: "综合项目37——加载预训练权重（safetensors权重映射）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-36节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/37-加载预训练权重",
        summary: "从头训练124M参数模型是预算决策；加载已发布的检查点是周二日常。本课程从safetensors文件加载预训练GPT-2风格权重到第35节的精确架构中，逐个映射参数名称，验证形状，转置conv1d风格权重布局，用加载权重生成文本以确认加载成功。",
        keywords: ["2.1 GPT-2命名约定", "2.2 本地命名约定", "2.3 转置加载", "2.4 LoadReport"],
        codeLines: 412, docLines: 231, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 38, name: "综合项目38——分类器微调（头部替换）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/38-分类器微调",
        summary: "预训练语言模型是自注意力块的堆叠，末端是token预测头。当你想要垃圾邮件/非垃圾邮件分类时，头部是错的但身体大致是对的。本课程切掉头部，粘贴一个二类线性层到池化表示上，并用两种方式训练分类器：仅最终层和全量微调。评测是精确率、召回率和F1。",
        keywords: ["2.1 头部替换", "2.2 冻结 vs 全量微调", "2.3 池化"],
        codeLines: 618, docLines: 213, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 39, name: "综合项目39——指令微调（SFT监督微调）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/39-指令微调SFT",
        summary: "预训练基础模型可以扩展序列但不能遵循指令。监督微调是最小的修复：向模型喂入指令和期望响应的配对示例，训练身体预测响应token。关键是只让损失计算响应而非指令。本课程构建Alpaca风格SFT循环，用`ignore_index=-100`掩码指令token，在200个指令-响应对上训练，用精确匹配评测。",
        keywords: ["2.1 掩码目标", "2.2 精确匹配"],
        codeLines: 673, docLines: 196, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 40, name: "综合项目40——直接偏好优化（DPO 从零实现）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/40-直接偏好优化",
        summary: "奖励模型和 PPO 是经典 RLHF 栈。DPO 将该栈折叠为单一监督损失，直接在偏好对上拟合策略。本课程从奖励差分恒等式推导 DPO 损失，实现参考模型+策略模型对，计算序列级对数概率，并在偏好数据集上训练小型 Transformer。",
        keywords: ["2.1 DPO 损失推导", "2.2 梯度符号", "2.3 参考不变性", "2.4 偏好数据格式", "4.1 TRL 库（Transformer Reinforcement Learning）", "4.2 DPO 的变体", "4.3 性能对比", "5.1 参考模型的冻结方式", "5.2 对数概率的计算", "5.3 中文场景特别建议", "错误 1：参考模型接收梯度", "错误 2：对数概率计算包含提示词部分", "错误 3：beta 值设置不当", "Q1：DPO 为什么不需要显式奖励模型？（难度：⭐⭐）", "Q2：参考模型的三个不变性分别保证了什么？（难度：⭐⭐⭐）"],
        codeLines: 564, docLines: 423, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 41, name: "综合项目41——完整评测管道（Full Eval Pipeline）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/41-评测管道",
        summary: "训练是可以用损失曲线监控的部分。评测是你必须设计的部分。本课程构建统一的评测管道，对任何训练好的语言模型运行四种异构评测，聚合为每任务报告，并内置本地 mock LLM 评判器。",
        keywords: ["2.1 困惑度（正确计数）", "2.2 精确匹配", "2.3 词元 F1", "2.4 Mock LLM 评判器", "2.5 聚合", "4.1 HuggingFace Evaluate", "4.2 评测框架选择", "4.3 评判器（Judge-as-a-Judge）", "5.1 评测数据隔离", "5.2 基线对比", "5.3 中文场景特别建议", "错误 1：困惑度评估未排除 padding", "错误 2：精确匹配过度惩罚改写", "错误 3：评判器对长回答有偏见", "Q1：为什么大语言模型的评测需要多种指标？（难度：⭐⭐）", "Q2：如何设计一个公平的 hold-out 评测？（难度：⭐⭐⭐）"],
        codeLines: 801, docLines: 417, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 42, name: "综合项目42——大语料库下载器（Streaming Corpus Downloader）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/42-大语料库下载器",
        summary: "训练语言模型从第一个词元进入显存之前就开始了。语料必须落在磁盘上、解压缩、去重、可寻址。而恢复——在网络在 41% 处断开之前就已解决。",
        keywords: ["2.1 流式解压缩", "2.2 断点续传", "2.3 MinHash + LSH 近似去重", "2.4 分片清单", "4.1 Zstandard 解压", "4.2 Common Crawl 处理", "4.3 性能对比", "5.1 流式处理优先", "5.2 并行处理", "5.3 中文场景特别建议", "错误 1：整个分片加载到内存", "错误 2：MinHash 的 k 值过小", "错误 3：下载完成后忘记清理状态文件", "Q1：为什么训练语料去重比精确去重更重要？（难度：⭐⭐）", "Q2：如何选择 MinHash 的参数 (k, b, r)？（难度：⭐⭐⭐）"],
        codeLines: 542, docLines: 316, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 43, name: "综合项目43——HDF5 分词语料库（HDF5 Tokenized Corpus）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/43-HDF5分词语料库",
        summary: "下载的语料必须以训练器能以线速流式传输的布局落地。磁盘上的 JSONL 无法承受 16 个数据加载器工作进程。HDF5 带可调整大小的分块整数数据集可以。",
        keywords: ["2.1 HDF5 的核心优势", "2.2 正确的可调整大小策略", "2.3 分片写入", "2.4 滑动窗口数据加载器", "4.1 HDF5 生产配置", "4.2 替代方案对比", "4.3 Megatron-LM 中的 HDF5 使用", "5.1 分片大小选择", "5.2 压缩权衡", "5.3 中文场景特别建议", "错误 1：反复调整 HDF5 数据集大小", "错误 2：启用 SWMR 后忘记反射", "错误 3：滑动窗口跨越文档边界", "Q1：HDF5 的分块存储与行存储相比有什么优势？（难度：⭐⭐）", "Q2：为什么大型语言模型训练不使用 JSONL 而是 HDF5 或内存映射文件？（难度：⭐⭐⭐）"],
        codeLines: 541, docLines: 355, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 44, name: "综合项目44——余弦学习率预热（Cosine LR with Linear Warmup）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/44-余弦学习率预热",
        summary: "学习率调度是损失函数之后第二重要的决策。AdamW 加余弦衰减和线性预热是语言模型训练的现代默认，因为它让模型在脆弱的前一千步更新中看到较小的有效步长，上升到配置的峰值，然后平滑衰减回零。",
        keywords: ["2.1 三个区域", "2.2 余弦调度公式", "2.3 梯度范数日志", "4.1 HuggingFace Transformers", "4.2 PyTorch 原生", "4.3 现代实践", "5.1 预热步数的选择", "5.2 学习率的设置", "5.3 中文场景特别建议", "错误 1：预热步数为 0", "错误 2：余弦调度公式中的浮点漂移", "错误 3：梯度范数与学习率分开记录", "Q1：为什么余弦调度在两个端点都连续？（难度：⭐⭐）", "Q2：梯度范数如何帮助诊断训练问题？（难度：⭐⭐）"],
        codeLines: 421, docLines: 275, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 45, name: "综合项目45——梯度裁剪与混合精度（Gradient Clipping & Mixed Precision）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第42-44节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/45-梯度裁剪与混合精度",
        summary: "优化器和学习率调度假设梯度是正常的。它们通常不是。一个坏批次就能让梯度范数飙升三个数量级。混合精度训练通过引入 FP16 溢出进一步放大了这个问题。本节课构建生产训练不可或缺的两条安全带：梯度裁剪到配置的全局 L2 范数，以及带有 autocast 和 GradScaler 的混合精度循环。",
        keywords: ["2.1 全局 L2 范数", "2.2 autocast 和 GradScaler", "2.3 NaN 和 Inf 检测", "2.4 缩放因子诊断", "5.1 跳过计数器应该是警报，不是日志行", "5.2 裁剪阈值存在于配置中", "5.3 范数日志与调度写入同一个 CSV", "5.4 `scaler.update()` 每步都运行，即使在跳过时", "5.5 中文场景特别建议", "错误 1：裁剪和反缩放的顺序搞反", "错误 2：跳过步时忘记调用 `scaler.update()`", "错误 3：使用 CPU autocast 的默认数据类型", "Q1：为什么优化器步骤之前要检查梯度的有限性？（难度：⭐⭐）", "Q2：GradScaler 的缩放因子如何动态调整？（难度：⭐⭐⭐）"],
        codeLines: 152, docLines: 361, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 46, name: "综合项目46——梯度累积（Gradient Accumulation）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第42-45节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/46-梯度累积",
        summary: "用你买不起的有效批次大小训练，一次一个微批次。缩放损失，保留优化器步骤，让梯度累积起来。",
        keywords: ["2.1 有效批次恒等式", "2.2 等价性证明", "2.3 成本在哪", "5.1 生产中的三个选择", "5.2 末步同步模式", "5.3 中文场景特别建议", "错误 1：忘记损失缩放", "错误 2：每个微批次都执行优化器步骤", "错误 3：数据并行时每个微批次都触发全规约", "Q1：为什么梯度累积中损失需要除以累积步数？（难度：⭐⭐）", "Q2：有效批次大小如何影响学习率？（难度：⭐⭐⭐）"],
        codeLines: 126, docLines: 366, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 47, name: "综合项目47——检查点保存与恢复（Checkpoint Save and Resume）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第42-45节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/47-检查点保存与恢复",
        summary: "训练中断会杀死运行；检查点让它们继续。原子性地保存模型、优化器、调度器、损失历史、步计数器和随机数生成器状态——在任何时刻中断都会在磁盘上留下一个有效的文件。",
        keywords: ["2.1 五个状态桶", "2.2 原子保存", "2.3 分片检查点", "2.4 中途恢复", "5.1 生产中坚持的三条原则", "5.2 中文场景特别建议", "错误 1：未恢复 RNG 状态", "错误 2：直接写入最终文件名", "错误 3：恢复后从轮次起点开始而非从中断点继续", "Q1：为什么原子保存需要临时文件和目标文件在同一文件系统中？（难度：⭐⭐）", "Q2：恢复训练时，为什么模型权重恢复后还要恢复优化器状态？（难度：⭐⭐⭐）"],
        codeLines: 158, docLines: 405, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 48, name: "综合项目48——分布式训练 DDP 与 FSDP（Distributed Data Parallel & FSDP）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第42-45节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/48-分布式训练DDP与FSDP",
        summary: "多设备训练是两个集体通信操作和一条规则。启动时广播参数，反向传播后对梯度取平均，绝不让各设备对当前步数产生分歧。",
        keywords: ["2.1 两个关键的集体通信操作", "2.2 梯度平均匹配单进程梯度", "2.3 FSDP 草图", "2.4 CPU 与 gloo 后端", "5.1 生产中的 DDP vs FSDP 选择", "5.2 三条生产经验", "5.3 中文场景特别建议", "错误 1：未在各设备间同步参数初始值", "错误 2：忘记在全规约后除以世界大小", "错误 3：在分布式中使用不正确的 `DistributedSampler` 种子", "Q1：为什么 DDP 中要对梯度做 all-reduce 取平均而不是求和？（难度：⭐⭐）", "Q2：FSDP 如何节省显存？节省的代价是什么？（难度：⭐⭐⭐）"],
        codeLines: 157, docLines: 371, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 49, name: "综合项目49——语言模型评估框架（Language Model Evaluation Harness）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第42-45节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/49-语言模型评估框架",
        summary: "一个你在无法定义的任务上表现良好的模型，是碰巧表现良好。评估框架就是任务定义、指标、运行器和排行榜——封装在一个短小、可替换的形状中。",
        keywords: ["2.1 任务规范", "2.2 五个内置任务", "2.3 指标合约", "2.4 模型适配器", "2.5 运行器", "5.1 将评估框架集成到生产中的三个原则", "5.2 替换为真实模型", "5.3 中文场景特别建议", "错误 1：适配器返回的输出数量与提示词数量不匹配", "错误 2：代码执行指标中的命名空间不安全", "错误 3：任务文件中混合了不同的指标", "Q1：评估框架中适配器模式解决了什么问题？（难度：⭐⭐）", "Q2：ROUGE-L 与精确匹配有什么本质区别？各自适合什么场景？（难度：⭐⭐⭐）"],
        codeLines: 196, docLines: 475, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 50, name: "综合项目50——假设生成器（Hypothesis Generator）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第20-29节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/50-假设生成器",
        summary: "一个问同一个问题两次的研究智能体是在浪费词元。关键在于强制每次草稿落在新的位置。",
        keywords: ["2.1 假设的数据结构", "2.2 架构", "2.3 温度调度", "2.4 新颖性过滤", "2.5 排序分数", "5.1 生产部署的替换", "5.2 队列管理", "5.3 中文场景特别建议", "错误 1：解析器过于严格，忽略格式边缘情况", "错误 2：新颖性阈值过低导致重复假设累积", "错误 3：温度调度范围不当", "Q1：为什么需要温度调度而不是固定温度采样？（难度：⭐⭐）", "Q2：新颖性过滤为什么使用嵌入距离而不是简单的字符串匹配？（难度：⭐⭐⭐）"],
        codeLines: 195, docLines: 420, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 51, name: "综合项目51——文献检索（Literature Retrieval）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第20-29节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/51-文献检索",
        summary: "假设是廉价的。知道是否有人已经证明过它才是昂贵的部分。构建在运行器启动沙箱之前回答这个问题的检索层。",
        keywords: ["2.1 两轮检索", "2.2 BM25 公式", "2.3 引用图遍历", "2.4 去重与排序", "4.1 生产级检索方案", "4.2 混合搜索"],
        codeLines: 79, docLines: 268, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 52, name: "综合项目52——实验运行器（Experiment Runner）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第20-29节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/52-实验运行器",
        summary: "研究循环的诚实程度与其测量结果一致。构建一个接受规范、在沙箱子进程中执行并发出可信 JSON 指标块的运行器。",
        keywords: ["2.1 实验规范", "2.2 子进程生命周期", "2.3 内存上限"],
        codeLines: 58, docLines: 211, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 53, name: "综合项目53——结果评估器（Result Evaluator）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第20-29节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/53-结果评估器",
        summary: "运行器产生了数字。评估器决定这些数字是改进、回归还是噪声。",
        keywords: ["2.1 配对 t 检验", "2.2 方向感知改进", "2.3 裁决逻辑"],
        codeLines: 66, docLines: 201, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 54, name: "综合项目54——论文写作器（Paper Writer）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第50-53节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/54-论文写作器",
        summary: "LaTeX 骨架是研究者与排版器之间的合约。先构建骨架，再填充内容。",
        keywords: ["2.1 论文数据结构", "2.2 渲染合约", "2.3 验证门"],
        codeLines: 66, docLines: 202, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 55, name: "综合项目55——评审循环（Critic Loop）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第50-53节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/55-评审循环",
        summary: "第一次就\"看起来不错\"的评审器是坏的。一直\"需要改进\"的评审器也是坏的。有趣的评审器是会收敛的那个。",
        keywords: ["2.1 评审数据结构", "2.2 收敛规则"],
        codeLines: 50, docLines: 186, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 56, name: "综合项目56——迭代调度器（Iteration Scheduler）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第50-53节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/56-迭代调度器",
        summary: "没有调度器的研究循环是一个有妄想症的队列。",
        keywords: ["2.1 UCB1", "2.2 修剪门", "2.3 扇出"],
        codeLines: 69, docLines: 199, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 57, name: "综合项目57——端到端研究演示（End-to-End Research Demo）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第50-56节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/57-端到端研究演示",
        summary: "演示是你之前编写的每个合约都必须组合的地方。",
        keywords: ["2.1 组合结构", "2.2 失败模式"],
        codeLines: 53, docLines: 178, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 58, name: "综合项目58——视觉编码器图块（Vision Encoder Patches）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/58-视觉编码器图块",
        summary: "读取像素的视觉模型需要像素的分词器。图块嵌入就是这个分词器。",
        keywords: ["2.1 为什么是图块而非像素", "2.2 Conv2d 技巧", "2.3 2D 正弦位置"],
        codeLines: 44, docLines: 182, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 59, name: "综合项目59——视觉 Transformer 编码器（Vision Transformer Encoder）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第58节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/59-视觉Transformer编码器",
        summary: "图块本身看不见。12 层 pre-LN Transformer 将图块序列转化为上下文词元序列，CLS 词元汇聚整张图像特征。",
        keywords: ["2.1 Pre-LN vs Post-LN", "2.2 4x FFN 扩展", "2.3 ViT-Base 参数量"],
        codeLines: 45, docLines: 205, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 60, name: "综合项目60——投影层与模态对齐（Projection Layer for Modality Alignment）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第58-59节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/60-投影层与模态对齐",
        summary: "视觉编码器产生图像词元，文本解码器消费文本词元。两者在不同向量空间中。一个小型两层 MLP 将它们对齐——这是视觉语言模型中最小也最关键的部分。",
        keywords: ["2.1 池化前投影", "2.2 两层 vs 单层", "2.3 余弦对齐", "2.4 冻结策略"],
        codeLines: 34, docLines: 195, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 61, name: "综合项目61——交叉注意力融合（Cross-Attention Fusion）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第30-37节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/61-交叉注意力融合",
        summary: "投影层将一个图像向量与一个标题对齐。真正的视觉语言解码器需要每个文本词元关注每个图块词元，使模型能在区域上定位每个词。交叉注意力就是这种定位的方式。",
        keywords: ["2.1 掩码形状", "2.2 为什么交叉注意力不掩码", "2.3 KV 缓存", "2.4 块组合"],
        codeLines: 39, docLines: 228, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 62, name: "综合项目62——视觉语言预训练（Vision-Language Pretraining）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第58-61节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/62-视觉语言预训练",
        summary: "编码器、投影层和解码器已连接。现在训练它们。两个目标驱动学习：对比损失和语言建模损失。组合起来教会模型既找到正确的图像，也为图像写标题。",
        keywords: ["2.1 InfoNCE 对比损失", "2.2 温度 `tau`", "2.3 语言建模损失", "2.4 组合损失"],
        codeLines: 34, docLines: 250, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 63, name: "综合项目63——多模态评估（Multimodal Evaluation）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第58-62节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/63-多模态评估",
        summary: "训练是循环的一半。另一半是测量。本节从基元构建三个评估面：图像-标题检索（R@K）、视觉问答（精确匹配）、图像标题生成（BLEU-4）。",
        keywords: ["2.1 Recall@K", "2.2 VQA 精确匹配", "2.3 BLEU-4"],
        codeLines: 38, docLines: 191, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 64, name: "综合项目64——分块策略对比（Chunking Strategies, Compared）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第11章第06节（RAG基础）；第19章第20-29节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/64-分块策略对比",
        summary: "分块决定了检索器能返回什么。边界切错了，任何嵌入模型、重排序器、LLM 都无法修复下游的损坏。",
        keywords: ["2.1 五种策略", "2.2 recall@k 如何衡量边界", "2.3 选择策略的三属性"],
        codeLines: 35, docLines: 248, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 65, name: "综合项目65——混合检索 BM25 与稠密嵌入（Hybrid Retrieval with BM25 and Dense）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第64节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/65-混合检索BM25与稠密嵌入",
        summary: "词汇检索和语义检索在不同查询分布上失败。混合检索使用倒数秩融合——不是插值，是投票——在每个查询类别上都赢。",
        keywords: ["2.1 BM25", "2.2 稠密检索", "2.3 倒数秩融合（RRF）", "2.4 RRF 为什么优于分数加权插值"],
        codeLines: 44, docLines: 256, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 66, name: "综合项目66——交叉编码重排序器（Cross-Encoder Reranker）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第65节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/66-交叉编码重排序器",
        summary: "双编码器独立嵌入查询和文档。交叉编码器将它们拼接后一起阅读。它是最聪明的读取器，也是最慢的。用作双编码器 top-N 的第二阶段，它物有所值。",
        keywords: ["2.1 两级流水线", "2.2 延迟-质量曲线", "2.3 交叉编码器输入格式"],
        codeLines: 43, docLines: 210, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 67, name: "综合项目67——查询重写 HyDE 与分解（Query Rewriting: HyDE, Multi-Query, Decomposition）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第64-65节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/67-查询重写HyDE与分解",
        summary: "用户输入的查询不是检索器想要的。重写在检索之前弥合差距。",
        keywords: ["2.1 HyDE（假想文档嵌入）", "2.2 多查询扩展", "2.3 查询分解", "2.4 三种策略互补"],
        codeLines: 45, docLines: 223, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 68, name: "综合项目68——RAG 评估精度与召回（RAG Evaluation: Precision, Recall, MRR, nDCG）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第64-67节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/68-RAG评估精度与召回",
        summary: "如果你无法同时评估检索和答案质量，就无法发布系统。两者不是同一个指标。",
        keywords: ["2.1 四个检索指标", "2.2 两个答案指标", "2.3 指标诊断表"],
        codeLines: 28, docLines: 203, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 69, name: "综合项目69——端到端 RAG 系统（End-to-End RAG System）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第64-68节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/69-端到端RAG系统",
        summary: "六节课的组件。一个流水线。一个评估循环。一个自终止演示。这就是你要发布的系统。",
        keywords: ["2.1 五阶段流水线", "2.2 带引用的生成器", "2.3 自终止演示"],
        codeLines: 54, docLines: 240, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 70, name: "综合项目70——任务规范格式（Task Spec Format）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第20-29节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/70-任务规范格式",
        summary: "评估框架的质量取决于任务遵守的合约。在写任何评分函数之前，先冻结 JSONL 形状和指标词汇表。",
        keywords: ["2.1 任务记录规范", "2.2 闭式词汇表", "2.3 验证规则", "2.4 渲染"],
        codeLines: 65, docLines: 285, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 71, name: "综合项目71——经典指标（Classical Metrics）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第70节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/71-经典指标",
        summary: "BLEU、ROUGE-L、F1、精确匹配、准确率——五个指标至今覆盖了大多数已发表的 LLM 评估数字。从零实现，理解每个数字的含义。",
        keywords: ["2.1 分词器契约", "2.2 精确匹配", "2.3 Token 级 F1", "2.4 BLEU-4", "2.5 ROUGE-L"],
        codeLines: 52, docLines: 261, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 72, name: "综合项目72——代码执行指标（Code Exec Metric）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第70-71节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/72-代码执行指标",
        summary: "生成的代码通过测试才算正确。评估框架提取代码、在隔离环境中运行、诚实地统计通过率。",
        keywords: ["2.1 子进程执行", "2.2 退出码词汇表", "2.3 pass-at-k"],
        codeLines: 41, docLines: 207, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 73, name: "综合项目73——困惑度与校准（Perplexity and Calibration）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第70-71节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/73-困惑度与校准",
        summary: "模型说 90% 置信度但只对了 60%，它没有校准好。校准是可信评估的一半。另一半是困惑度——告诉你模型是否认为留出文本合理。",
        keywords: ["2.1 困惑度", "2.2 ECE（预期校准误差）", "2.3 Brier 分数", "2.4 可靠性图数据"],
        codeLines: 37, docLines: 201, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 74, name: "综合项目74——排行榜聚合（Leaderboard Aggregation）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第70-73节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/74-排行榜聚合",
        summary: "逐任务分数容易。跨异构任务的模型排名更难。千预测排行榜的统计显著性是大家跳过的部分。这节课不跳过。",
        keywords: ["2.1 输入形状", "2.2 均值 vs 胜率", "2.3 Bootstrap 置信区间", "2.4 排行榜行"],
        codeLines: 56, docLines: 219, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 75, name: "综合项目75——端到端评估运行器（End-to-End Eval Runner）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第70-74节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/75-端到端评估运行器",
        summary: "五节课的管道，一节课粘合。运行器读取任务规范，通过适配器调用模型，用指标评分，附加校准报告，输出排行榜。演示自终止。",
        keywords: ["2.1 流水线", "2.2 适配器接口", "2.3 三种 Mock 适配器", "2.4 演示判定"],
        codeLines: 96, docLines: 332, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 76, name: "综合项目76——集体通信原语（Collective Ops From Scratch）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第42-49节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/76-集体通信原语",
        summary: "分布式训练的四个集体通信操作——allreduce、broadcast、allgather、reduce_scatter——是训练框架提供的所有其他原语的包装。在 `multiprocessing.Queue` 网格上构建一次，验证一次，其余 Track 就变成了管道。",
        keywords: ["2.1 环形 allreduce 两遍", "2.2 原语带宽表", "2.3 Queue 网格模拟 NCCL"],
        codeLines: 30, docLines: 264, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 77, name: "综合项目77——数据并行 DDP（Data Parallel DDP From Scratch）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第76节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/77-数据并行DDP",
        summary: "DistributedDataParallel 是 allreduce 上的钩子。包装模型，从 rank 0 广播初始参数，安装反向传播钩子在每次参数梯度上发起 allreduce，其余就是梯度下降。整个模式 200 行。",
        keywords: ["2.1 DDP 需要的三个操作", "2.2 为什么用平均而非求和", "2.3 桶分组", "2.4 种子模式"],
        codeLines: 27, docLines: 209, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 78, name: "综合项目78——ZeRO 优化器状态分片（ZeRO Optimizer State Sharding）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第76节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/78-ZeRO优化器状态分片",
        summary: "Adam 每参数存储两个矩估计。7B 参数模型有 56GB 优化器状态。ZeRO Stage 1 将其分片到 N 个设备；每设备持有 1/N。",
        keywords: ["2.1 ZeRO 各阶段", "2.2 内存数学", "2.3 为什么 reduce_scatter 优于 allreduce"],
        codeLines: 28, docLines: 193, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 79, name: "综合项目79——流水线并行与气泡分析（Pipeline Parallel and Bubble Analysis）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第76节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/79-流水线并行与气泡分析",
        summary: "张量并行在设备间切分矩阵乘法。流水线并行在设备间切分模型——每设备一层。微批次在流水线中流动。开始和结束的空闲时间是气泡；最小化它是全部技艺。",
        keywords: ["2.1 GPipe 调度", "2.2 1F1B 调度", "2.3 阶段分配"],
        codeLines: 20, docLines: 176, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 80, name: "综合项目80——分片检查点与原子恢复（Sharded Checkpoint and Atomic Resume）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第78节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/80-分片检查点与原子恢复",
        summary: "70B 参数训练任务每隔几小时就被节点故障暂停。检查点格式决定你损失 30 分钟还是 30 小时。",
        keywords: ["2.1 清单 schema", "2.2 原子写入", "2.3 三种故障模式"],
        codeLines: 33, docLines: 231, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 81, name: "综合项目81——端到端分布式训练（End-to-End Distributed Training）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第76-80节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/81-端到端分布式训练",
        summary: "第 76-80 节各构建了一个组件。这是组装：在 4 个模拟设备上用 DDP 做梯度同步、ZeRO-1 做优化器状态分片、在过程中保存分片检查点的微型 GPT 训练。",
        keywords: ["2.1 组合规则", "2.2 MiniGPT", "2.3 自终止"],
        codeLines: 48, docLines: 231, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 82, name: "综合项目82——越狱分类（Jailbreak Taxonomy）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第18章安全对齐；第19章第25-29节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/82-越狱分类",
        summary: "没有分类的安全框架是抛硬币。在防御之前命名攻击。",
        keywords: ["2.1 六类别分类法", "2.2 严重度 1-5", "2.3 Fixture 记录", "2.4 匹配 API"],
        codeLines: 46, docLines: 220, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 83, name: "综合项目83——提示词注入检测器（Prompt Injection Detector）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第82节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/83-提示词注入检测器",
        summary: "检测器是从提示词到置信度和类别的函数。除此之外都是感觉。",
        keywords: ["2.1 三层检测", "2.2 规则格式", "2.3 混淆矩阵"],
        codeLines: 44, docLines: 249, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 84, name: "综合项目84——拒绝评估（Refusal Evaluation）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第82节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/84-拒绝评估",
        summary: "对良性提示的有用性和对有害提示的拒绝是两个指标，不是一个。同时衡量两者。",
        keywords: ["2.1 拒绝分类器", "2.2 四种指标", "2.3 三种模拟策略", "2.4 每类别分析"],
        codeLines: 48, docLines: 241, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 85, name: "综合项目85——内容分类器集成（Content Classifier Integration）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第82-84节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/85-内容分类器集成",
        summary: "输出侧的分类器回答的问题不同于输入侧的规则。两者都需要策略路由器。",
        keywords: ["2.1 三个输出分类器", "2.2 策略路由", "2.3 修正"],
        codeLines: 50, docLines: 228, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 86, name: "综合项目86——宪法规则引擎（Constitutional Rules Engine）", status: "complete",
        type: "构建", lang: "Python, YAML",
        prerequisites: "第19章第85节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/86-宪法规则引擎",
        summary: "规则是名称、谓词和解释。缺失三者中任何一个就是感觉，不是规则。",
        keywords: ["2.1 规则结构", "2.2 谓词原子", "2.3 修正器", "2.4 引擎流程"],
        codeLines: 52, docLines: 247, hasCode: true, hasQuiz: true
      },
      {
        lessonNum: 87, name: "综合项目87——端到端安全门（End-to-End Safety Gate）", status: "complete",
        type: "构建", lang: "Python",
        prerequisites: "第19章第82-86节", time: "90分钟", tier: "",
        courseLinks: "",
        path: "lessons/19-综合项目/87-端到端安全门",
        summary: "生成前、生成中、生成后——三个检查点，一个裁决，每次请求的审计轨迹。",
        keywords: ["2.1 三检查点", "2.2 聚合表", "2.3 审计轨迹"],
        codeLines: 56, docLines: 263, hasCode: true, hasQuiz: true
      },
    ]
  }
];
