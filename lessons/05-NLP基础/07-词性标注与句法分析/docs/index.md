# 词性标注与句法分析

> 语法曾经过时了一阵子。然后每个 LLM 流水线都需要验证结构化提取的结果，它又回来了。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 05 · 01（文本预处理）、阶段 02 · 14（朴素贝叶斯）
**预计时间：** ~45 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 01（文本预处理）— 词性标注是词形还原的前提；阶段 05 · 06（命名实体识别）— 同为序列标注任务

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现最频标签（MFT）词性标注基线——理解 85% 准确率这个"地板"是怎么来的
- [ ] 从零实现 Bigram HMM + Viterbi 解码——理解转移概率如何将准确率从 85% 推到 93%
- [ ] 解释为什么 Penn Treebank 的上限是 ~97% 而非 100%——标注者分歧的概念
- [ ] 使用 spaCy 完成生产级词性标注和依存分析

---

## 1. 问题

阶段 05 · 01 的课程中有一个承诺：词形还原需要词性标注。不知道 `running` 是动词（VERB），词形还原器就无法把它还原为 `run`。不知道 `better` 在这个上下文中是形容词，它就不会还原为 `good`。

这个承诺背后藏着一整个子领域。词性标注为每个词元打上语法类别标签。句法分析还原句子的树结构——哪个词修饰哪个词，哪个动词支配哪些论元。

经典 NLP 花了二十年精细化这两个任务。然后深度学习把它们压缩成了预训练 Transformer 顶上的一个 token-classification 层——学术界翻篇了。应用界没有。每个结构化提取流水线的底层都还在用词性标注和依存分析。LLM 生成的 JSON 需要语法约束来验证。问答系统用依存分析来分解查询。机器翻译质量评估器用句法树的对齐来衡量翻译质量。语法不会自己消失——它只是藏得更深了。

---

## 2. 概念

### 2.1 词性标注——给每个词贴上语法标签

```
The/DET  cats/NOUN  were/AUX  running/VERB  at/ADP  3pm/NOUN  ./PUNCT
```

两大标签体系：

| | Penn Treebank (PTB) | Universal Dependencies (UD) |
|---|---|---|
| **标签数量** | 36 个（精细） | 17 个（粗粒度） |
| **适用范围** | 英文为主 | 100+ 种语言 |
| **典型精细度** | NN（单数名词）、NNS（复数名词）、NNP（专有名词单数）… | 全部合并为 NOUN |
| **2026 选择** | 英文遗留系统 | 跨语言工作的默认选择 |

### 2.2 句法分析——两个流派

**成分分析（Constituency Parsing）：** 名词短语（NP）、动词短语（VP）、介词短语（PP）层层嵌套。输出是词为叶子的非终结符类别树。

**依存分析（Dependency Parsing）：** 每个词有唯一的"支配词"（head），每条边有语法关系标签。输出是有向边为 (支配词, 被支配词, 关系) 三元组的树。

```
running 是 ROOT（句子的根）
cats    是 nsubj of running（running 的名词主语）
were    是 aux of running（running 的助动词）
at      是 prep of running（running 的介词修饰语）
3pm     是 pobj of at（at 的介词宾语）
```

依存分析在 2010 年代胜出——因为它对语言的语序差异更鲁棒，尤其是自由语序语言（如德语、日语、捷克语）。

### 2.3 中文的特殊性——没有屈折变化意味着什么

英文中，一个词的后缀强烈暗示它的词性：`-tion` → 名词，`-ly` → 副词，`-ed` → 动词过去式。中文没有这些信号。"学习"可以是动词（"我在学习"）也可以是名词（"机器学习"）。同一个字形，词性完全靠上下文决定。这使得中文词性标注对上下文窗口的宽度和质量的依赖比英文大得多——简单的 MFT 基线在中文上的表现比英文更差。

---

## 3. 从零实现

### 第 1 步：最频标签（MFT）基线

最笨但能用的词性标注器。对每个词，记住它在训练数据中出现频率最高的词性。没见过的词→回退全局最高频词性。

```python
from collections import Counter, defaultdict

def train_mft(examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0]
                 for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag

def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Brown Corpus 上 ~85%。这是地板——任何严肃模型的准确率不应该低于它。MFT 的核心局限："red"作为形容词（a red cat）和名词（the red of sunset）使用完全相同的标签"ADJ"——MFT 不给任何上下文反馈的机会。

### 第 2 步：Bigram HMM + Viterbi 解码

HMM 建模的是整个序列的联合概率：

$$P(\text{tags}, \text{words}) = \prod_i \underbrace{P(\text{tag}_i \mid \text{tag}_{i-1})}_{\text{转移概率}} \times \underbrace{P(\text{word}_i \mid \text{tag}_i)}_{\text{发射概率}}$$

两张表——转移表（上一个标签→当前标签的概率）、发射表（给定标签→生成当前词的概率）。都从计数中估算，加拉普拉斯平滑。

Viterbi 解码用动态规划在 O(n × |T|²) 时间内找到最优标签序列：

```python
def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]   # V[i][j] = 前 i 步以标签 j 结尾的最优分
    back = [[0] * len(tags_list) for _ in range(n)]    # 回溯指针

    # 初始化：第 0 步，从 <BOS> 转移到第一个标签
    for j, tag in enumerate(tags_list):
        tr = log_prob(<BOS> → tag)    # P(tag|<BOS>)
        em = log_prob(tag → word[0])  # P(word[0]|tag)
        V[0][j] = tr + em

    # 递推：第 1 到 n-1 步
    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            # 找前一步中使 V[i-1][k] + P(tag|prev_tag_k) 最大的 k
            for k, prev_tag in enumerate(tags_list):
                score = V[i-1][k] + log_prob(prev_tag→tag) + log_prob(tag→word[i])
                V[i][j] = max(score)  # 取最优前继

    # 回溯——从最后一步最优标签一路往回推
    return backtracked_sequence
```

Bigram HMM 在 Brown Corpus 上 ~93%。从 85% 到 93% 的提升几乎全部来自转移概率——模型学到了"DET→NOUN 是高频模式"而"NOUN→DET 是罕见模式"。

### 第 3 步：HMM 的盲区——为什么 CRF 和 BiLSTM 是下一代

转移概率是局部的，无法解决词义消歧。"saw"在"I bought a saw"中是名词，在"I saw the movie"中是动词。转移概率在这两个句子中看到的模式都是"DET→saw"——没有区别。CRF 给"saw"添加了前后词特征（前面是"bought"→名词；前面是"I"→动词），BiLSTM 用神经特征替代了手工特征，Transformer 用注意力机制在整个句子上做全局特征组合。

完整代码见 `code/pos_tagger.py`。

---

## 4. 工业工具

### 4.1 spaCy——一行代码完成全部分析

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")

for token in doc:
    print(f"{token.text:10s} pos={token.pos_:6s} "
          f"tag={token.tag_:5s} dep={token.dep_:10s} "
          f"head={token.head.text}")
```

```
The        pos=DET    tag=DT    dep=det        head=cats
cats       pos=NOUN   tag=NNS   dep=nsubj      head=running
were       pos=AUX    tag=VBD   dep=aux        head=running
running    pos=VERB   tag=VBG   dep=ROOT       head=running
at         pos=ADP    tag=IN    dep=prep       head=running
3pm        pos=NOUN   tag=NN    dep=pobj       head=at
.          pos=PUNCT  tag=.     dep=punct      head=running
```

`token.pos_` 是 UD 粗粒度标签（17 种），`token.tag_` 是 PTB 细粒度标签（36 种），`token.dep_` 是依存关系。从 `dep` 列自下往上读，句子的语法结构自动浮现。

### 4.2 中文词性标注——工具选择

| 工具 | 标签集 | 特点 |
|---|---|---|
| spaCy `zh_core_web_sm` | UD | 中文 pipeline，分词+词性+依存 |
| HanLP | CTB（中文树库） | 多任务一体，中文原生支持 |
| LAC（百度） | 自有体系 | 分词+词性标注专门优化，速度快 |
| jieba + `posseg` | PKU 二级标签 | jieba 自带的 50+ 标签体系，轻量级 |

```python
# jieba 分词 + 词性标注（最轻量的中文方案）
import jieba.posseg as pseg

for word, flag in pseg.cut("我喜欢学习自然语言处理"):
    print(f"{word}/{flag}")
# 我/r  喜欢/v  学习/v  自然语言/n  处理/v
```

### 4.3 2026 年语法分析仍然重要的场景

- **词形还原。** 阶段 05 · 01 的承诺——不需要 word 形式和 tag 就能正确还原的词，英语里不到 10%
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词——"相机很好但电池很差"中的褒贬映射
- **查询分解。** "Wes Anderson 导演、Bill Murray 主演的电影"→ 依存分析拆出的约束结构可以直接转为数据库查询
- **LLM 输出验证。** 验证生成的句子满足语法约束（主谓一致、必要修饰语）
- **跨语言零样本。** UD 标签和依存关系与语言无关——可以用英文训练的语法规则零成本迁移到其他 UD 支持的语言上

---

## 5. 知识连线

词性标注和句法分析是序列标注问题的经典范本：

- **阶段 05 · 01（文本预处理）** → 词性标注是词形还原的前提——没有 tag 的 lemmatizer 只能还原规律变化
- **阶段 05 · 06（命名实体识别）** → 同为序列标注任务，使用相同 BIO 方案。NER 中 `B-ORG/I-ORG` 的转移约束逻辑与 POS 的 `DET→NOUN` 相同
- **阶段 07（Transformer 深入）** → BERT 的 token-classification head 把 MFT→HMM→CRF→BiLSTM 这四代模型压缩成了一个 `from_pretrained()` 调用

---

## 6. 工程最佳实践

### 6.1 标签集一致性——永不在项目中途换标签集

一旦你选择了 PTB 或 UD——你的训练数据、评估脚本、下游消费方必须全部使用同一套标签。UD 标签集更粗，PTB 更细——直接映射会丢失信息（PTB 的 NNP/NNS/NN 全合并为 UD 的 NOUN）。反过来把 UD 映射到 PTB 几乎不可行——信息已经丢失。

### 6.2 中文特别建议

- **中文没有大小写和后缀信号——前一个词的词性是后一个词词性的最强预测器之一。** 在 HMM 中这体现为转移概率的价值更高；在 BiLSTM 中这意味着字符级 CNN 是必须的模块——不是锦上添花
- **jieba 的词性标注预设了词的边界——如果分词错误，词性标注跟着错。** 这是中文 NLP 中经典的错误传播问题。解决：如果下游任务对词性敏感，考虑使用 HanLP 或 spaCy——它们的分词和标注是联合模型，而非流水线串联
- **中文标签体系不统一——jieba 用 PKU（50+ 标签）、HanLP 用 CTB（33 标签）、spaCy 中文用 UD（17 标签）。** 选工具的第一步是确认标签集能覆盖你的下游需求

### 6.3 踩坑经验

- **`token.pos_` 和 `token.tag_` 的区别容易踩坑——** spaCy 的 `pos_` 是 UD 粗粒度（NOUN），`tag_` 是 PTB 细粒度（NN/NNS/NNP）。如果你把 `tag_` 的值和 UD 文档交叉引用，一切都是错的
- **转移概率的价值在高频模式上，但低频模式上平滑比规则更重要。** 在 < 1000 条的标注数据上训练 HMM，转移矩阵是稀疏的——平滑（alpha）的大小直接影响 Viterbi 序列的质量。默认 0.01，在小数据上考虑调到 0.1
- **标注者分歧不是 bug——它是词性标注任务的本质。** 在 Penn Treebank 上，人类标注者的共识大约是 97%。如果你的模型突破了这个数——检查是否过拟合

---

## 7. 常见错误

### 错误 1：对中文使用英文标签集

**现象：** 用 PTB 标签标注中文句子——标注结果为"的/NN 猫/NN 跑/VBD 了/."。`VBD` 是英文动词过去式——中文没有过去式屈折变化，这个标签对中文没有语义。

**原因：** 不同语言的语法范畴不同——不能直接套用英文标签体系。

**修复：** 中文使用 UD（spaCy 中文、HanLP UD 模式）或 CTB（HanLP、LAC）。

### 错误 2：混淆 `pos_` 和 `tag_`

**现象：** 代码中用 `token.pos_` 和 PTB 标签表做匹配——全部失败。

**原因：** `pos_` 输出 UD 值（"NOUN"），`tag_` 输出 PTB 值（"NN"）。两个体系不互通。

**修复：** 确认你用的是哪个标签集。如果要 PTB → 用 `token.tag_`。如果要跨语言兼容 → 用 `token.pos_`。

---

## 8. 面试考点

### Q1：为什么 Penn Treebank 词性标注的准确率上限是 ~97% 而非 100%？（难度：⭐⭐）

**参考答案：**
因为标注者的一致性大约只有 97%。词性标注不是一个有着绝对"正确"答案的任务——在某些边界情况下，专业的语言学家也会对同一个词给出不同的标注。如果一个模型报告 99% 的准确率，它不是在"理解语法"，而是在过拟合测试集中标注者的特定偏好——换一组标注者，这个数字可能会掉回 97%。

### Q2：最频标签（MFT）为什么能到 85%，而 CRF 能到 97%——中间 12% 的差异来自哪里？（难度：⭐⭐）

**参考答案：**
~8% 来自转移概率——MFT 对每个词单独预测，不知道"DET→NOUN 是常态"。HMM 学了转移概率后把这 8% 补上了。剩下 ~4% 来自全局特征——"saw"的词义消歧需要看到前后更远的上下文，不只需要前一个词。CRF 可以融合全局特征（前后词、词形、后缀），捕捉到 MFT 和 HMM 无法触及的消歧信号。

### Q3：为什么在 2026 年 LLM 时代还要在意词性标注？（难度：⭐⭐⭐）

**参考答案：**
LLM 擅长生成，不擅长保证。当你的系统需要"保证"输出满足某些语法约束时——如"形容词必须修饰名词"、"否定词必须出现在被否定词之前"——LLM 的输出需要被词性标注和依存分析验证。另外，依存分析的语言中立性使其成为跨语言零样本迁移的重要桥梁——用一种语言训练的结构化提取规则可以直接应用到其他 UD 支持的语言上，不需要 LLM。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 词性标注 (POS Tag) | "标记词的类型" | 为每个词元分配语法类别。PTB 有 36 种，UD 有 17 种 |
| Penn Treebank (PTB) | "英文的标签标准" | 英文特有的细粒度标签体系——区分单复数名词、动词时态等 36 个标签 |
| Universal Dependencies (UD) | "多语言的标签标准" | 粗粒度 17 标签，100+ 种语言通用。跨语言工作的默认选择 |
| 依存分析 | "句子的语法树" | 每个词有唯一的支配词（head），每条边带语法关系标签。比成分分析更跨语言友好 |
| Viterbi | "动态规划找最优路径" | O(n×\|T\|²) 时间内找到最高概率标签序列。HMM 标注器的核心解码算法 |
| 标注者分歧 | "人也不百分之百一致" | Penn Treebank 上人类标注者共识约 97%——这是任何模型的理论上限 |

---

## 📚 小结

词性标注和句法分析是经典 NLP 的基石——词形还原需要词性，情感归因需要依存关系，查询分解需要句法树。MFT 给了 85% 的地板，HMM 用转移概率推到 93%，CRF 和 BiLSTM 推到 97%，BERT 把这个进化史压缩进了三个文件和一个 API 调用。

语法曾经似乎"被深度学习取代了"。但 2026 年的每个结构化提取系统仍在底层消费词性和依存分析——它们只是不再在台前了。

---

## ✏️ 练习

1. 【理解】MFT 基线遇到"red"这个词时永远输出同一个标签。找两个真实句子——一个"red"是形容词、一个"red"是名词——解释 MFT 为什么必然错一次。

2. 【实现】在 HMM 标注器的 Viterbi 解码中加入**全局标签约束**——如果某个标签在训练集中从未出现过，将其从候选集删除。测量这对未见词标注准确率的影响。

3. 【实验】用 spaCy 提取 100 句新闻的主谓宾三元组（nsubj + ROOT + dobj）。在 20 句上人工标注真实三元组，报告 spaCy 的准确率和召回率。分析失败模式：被动语态、并列结构、主语省略各自的错误占比。

4. 【思考】如果只用词性标注结果（不用 BERT），设计一个方案来验证 LLM 生成的广告文案是否满足"A 形容词不应修饰 B 名词"的品牌安全约束。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 词性标注从零实现 | `code/pos_tagger.py` | MFT 基线 + Bigram HMM + Viterbi 解码 + 评估 |

---

## 📖 参考资料

1. [书籍] Jurafsky and Martin. 《Speech and Language Processing》第 8 章和第 18 章. https://web.stanford.edu/~jurafsky/slp3/ — 词性标注与句法分析的权威教材
2. [官方文档] Universal Dependencies. https://universaldependencies.org/ — 跨语言标签集和树库集合
3. [官方文档] spaCy. "Linguistic Features". https://spacy.io/usage/linguistic-features — Token 上每个语法属性的实用参考
4. [论文] Chen and Manning. "A Fast and Accurate Dependency Parser using Neural Networks". EMNLP, 2014. https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf — 将神经依存分析引入主流

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文词性标注差异分析、中文工具推荐、工程最佳实践、常见错误、面试考点等均为原创内容。
