# 文本预处理——分词、词干提取与词形还原

> 语言是连续的，模型是离散的。预处理是二者之间的桥梁。中文 NLP 的桥梁比英文多一道坎——没有空格，分词的每一步都是抉择。

**类型：** 实现课
**语言：** Python
**前置知识：** 阶段 02 · 14（朴素贝叶斯）
**预计时间：** ~60 分钟
**所处阶段：** Tier 1
**关联课程：** 阶段 05 · 07（词性标注）— 词形还原的准确度依赖词性标注

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 从零实现英文正则分词器和中文正向/逆向最大匹配分词器
- [ ] 解释词干提取和词形还原的区别，以及各自的适用场景和失败模式
- [ ] 说明中文 NLP 预处理与英文 NLP 的关键差异——为什么中文不需要词干提取但需要分词
- [ ] 使用 NLTK、spaCy 和 jieba 完成生产级文本预处理
- [ ] 诊断训练/推理预处理不一致的问题，并给出修复方案

---

## 1. 问题

你的模型不认识"猫在跑"这三个字。它只认识整数。

这不是比喻，是字面意义上的事实。神经网络的每一层做的都是矩阵乘法、加法、非线性激活——全是数学运算。数学运算只能作用于数字。你不能把 `"猫"` 这个汉字传进 `torch.matmul()`，就像你不能把一张便利贴塞进计算器让它做加法。

所以，在任何文本进入模型之前，必须有一个组件把文字翻译成数字。这个组件就是**分词器（Tokenizer）**，这个过程就是**分词（Tokenization）**。"猫在跑"需要先变成 `[234, 87, 1029]` 这样的整数序列，模型才能开始计算。

第一道工序有了。接下来看不同语言的差异。

英文 NLP 的这一步相对简单——英文用空格天然分隔单词。绝大部分时候，`"The cats were running"` 按空格切出来的结果已经可用。

中文 NLP 的这一步要棘手得多——中文没有空格。"我们在北京大学学习自然语言处理"怎么切？是"北京/大学"还是"北京大学"？是"自然/语言"还是"自然语言"？选择不同，同一个"猫在跑"会被翻译成不同的整数序列，模型看到的数据就完全不同。

切完之后还有第二道和第三道工序。英文中 `running` 和 `ran` 表达的是同一件事的不同时态——你希望模型把它们视为同一个概念还是不同概念？字符串 `"running"` 和 `"ran"` 在计算机眼里是完全不同的两个东西，但你知道它们都指"跑"。你需要一个步骤来告诉模型这件事——这就是词干提取和词形还原要做的事。中文没有这种动词变位，但中文有自己的一套难题：繁体/简体转换、全角/半角混用、"的/地/得"的误用——同一个意思可能有多种写法，不同写法就是不同的整数序列。

这就是文本预处理。它不是"洗数据"——它是整个 NLP 流水线的第一道关卡。做对了，同一个语义的文字被映射到一致的整数空间，模型事半功倍。做错了，模型学到的全是噪音，再好的架构也救不回来。

---

## 2. 概念

### 2.1 三大操作

文本预处理有三个核心操作，每个都有明确的职责和失败模式：

```
原始文本："The cats were running at 3pm."
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
 分词           词干提取       词形还原
(Tokenization)  (Stemming)    (Lemmatization)
    │             │             │
    ▼             ▼             ▼
["The",       ["the",        ["the",
 "cats",       "cat",         "cat",
 "were",       "were",        "be",
 "running",    "run",         "run",
 "at",         "at",          "at",
 "3",          "3",           "3",
 "pm",         "pm",          "pm",
 "."]          "."]           "."]
```

**分词**将字符串切分为词元。"词元"的定义刻意保持模糊——词级、子词级、字符级都可以是"词元"，具体取决于任务。经典 NLP 用词级，Transformer 用子词级，中文和日文等无语空格语言有自己的一套。

**词干提取**用规则粗暴地砍后缀。快、激进、会出错。`running → run`（对），`organization → organ`（错了）。这个错误是词干提取的典型失败模式——过度归并导致语义丢失。

**词形还原**用语法知识将词还原为词典形式。慢、准确、需要查表或形态分析器。`ran → run`（需要知道"ran"是"run"的过去式），`better → good`（需要知道比较级形式）。

### 2.2 中文的特殊性

| 特性 | 英文 | 中文 |
|---|---|---|
| 词边界 | 空格天然分隔 | 无空格，需要分词算法 |
| 屈折变化 | 丰富（-ing, -ed, -s, 比较级...） | 几乎没有 |
| 词干提取 | 需要 | **不需要**——没有后缀可砍 |
| 词形还原 | 需要 | 通常不需要，但繁简转换可视为一种"词形还原" |
| 核心挑战 | 不规则变化（went → go） | 分词歧义（"研究生命" vs "研究生/命"） |

**中文 NLP 预处理的真正核心是分词。** 如果连"词"的边界都定不下来，后续一切操作都建立在错误的基础上。词干提取和词形还原对中文几乎没有意义——你不能把"跑了"的"了"砍掉说这是"词干提取"，那是助词，不是屈折后缀。

### 2.3 经验法则

| 场景 | 选择 |
|---|---|
| 搜索引擎索引（速度优先、容忍噪音） | 词干提取 |
| 问答系统、语义搜索（含义优先） | 词形还原 |
| 中文场景 | **先分词**，通常跳过词干/词形还原 |
| Transformer 流水线 | 跳过经典预处理，直接用模型自带的分词器 |

---

## 3. 从零实现

### 第 1 步：英文正则分词器

最简单的实用分词器：一条正则表达式，按优先级匹配三种模式——带撇号的单词、纯数字、标点符号。

```python
import re

# 三条规则按优先级排列：
# (1) 带内部撇号的单词（don't, it's）
# (2) 纯数字
# (3) 单个非空白非字母数字字符（标点符号）
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")

def tokenize_en(text):
    return WORD_RE.findall(text)
```

验证：

```python
>>> tokenize_en("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注意 `3pm` 被切成了 `['3', 'pm']`，因为数字和字母交替出现在不同的匹配组中。对于大多数任务来说这已经够用了。URL、邮箱地址、Hashtag 会出问题——生产环境中，在通用规则之前加上特殊模式即可。

### 第 2 步：中文最大匹配分词

中文分词的核心挑战是**歧义**。"研究生命"可以切成"研究生/命"（FMM 的错误倾向），也可以切成"研究/生命"（BMM 的正确倾向）。

正向最大匹配（FMM）从左到右贪心取最长匹配：

```python
def forward_max_match(text, dictionary):
    """正向最大匹配——从左到右，贪心取最长词。"""
    max_len = max(len(w) for w in dictionary)
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        # 从最大窗口开始尝试匹配，逐步缩小
        for window in range(max_len, 0, -1):
            candidate = text[i:i + window]
            if candidate in dictionary:
                tokens.append(candidate)
                i += window
                matched = True
                break
        if not matched:
            # 单字成词——中文分词的最小单元
            tokens.append(text[i])
            i += 1
    return tokens
```

逆向最大匹配（BMM）从右到左：

```python
def reverse_max_match(text, dictionary):
    """逆向最大匹配——从右到左，贪心取最长词。"""
    max_len = max(len(w) for w in dictionary)
    tokens = []
    i = len(text)
    while i > 0:
        matched = False
        for window in range(max_len, 0, -1):
            start = i - window
            if start < 0:
                continue
            candidate = text[start:i]
            if candidate in dictionary:
                tokens.insert(0, candidate)
                i = start
                matched = True
                break
        if not matched:
            tokens.insert(0, text[i - 1])
            i -= 1
    return tokens
```

验证歧义处理：

```python
>>> dic = {"研究", "研究生", "生命", "命", "我们", "喜欢", "自然语言", "自然", "语言"}
>>> forward_max_match("研究生命", dic)
['研究生', '命']        # ← FMM 错误地先把"研究生"作为一个词了
>>> reverse_max_match("研究生命", dic)
['研究', '生命']        # ← BMM 更合理
```

BMM 在中文分词中通常略优于 FMM，原因在于中文的偏正结构：修饰语在前，中心语在后。从后往前切更倾向于保留中心语的完整性。jieba 等工业工具使用基于词图和动态规划的更复杂算法，而非单纯的规则匹配。

### 第 3 步：Porter 词干提取器（Step 1a）

完整的 Porter 算法有五个阶段。Step 1a 覆盖了英文最高频的后缀，足以展示核心模式：**规则按顺序匹配，先匹配的先生效**。

```python
def stem_step_1a(word):
    """Porter 词干提取——Step 1a。规则顺序至关重要。"""
    if word.endswith("sses"):
        return word[:-2]       # caresses → caress
    if word.endswith("ies"):
        return word[:-2]       # ponies → poni
    if word.endswith("ss"):
        return word            # caress → caress（已经是词干，不动）
    if word.endswith("s") and len(word) > 1:
        return word[:-1]       # cats → cat
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

`ponies → poni` 而非 `pony` 是 Step 1a 的已知局限——完整的 Porter 算法的 Step 1b 会修正这个问题。**规则之间存在竞争。顺序比单条规则更重要。**

### 第 4 步：基于查表的词形还原器

词形还原需要形态学知识。教学版本用一个小的还原表 + 规则回退：

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run", ("ran", "VERB"): "run",
    ("better", "ADJ"): "good", ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",  ("were", "VERB"): "be",
    ("was", "VERB"): "be",    ("is", "VERB"): "be",
}

def lemmatize(word, pos="NOUN"):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    # 规则回退——只能覆盖规律变化
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")   # 不在表中，规则也覆盖不了
'watched'
```

最后一个例子是关键的教训。`watched` 不在查表中，我们的规则回退只处理了 `-ing`。真正的词形还原需要覆盖 `-ed`、不规则动词、比较级、复数音变（`children → child`）……这就是为什么生产系统使用 WordNet、spaCy 的词形还原器或完整的形态分析器。

### 第 5 步：串联为预处理流水线

```python
def preprocess_en(text, pos_tagger=None):
    """英文预处理流水线。pos_tagger 如果未提供，默认所有词为 NOUN。"""
    tokens = tokenize_en(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的环节是词性标注器。阶段 05 · 07（词性标注）会从零构建一个。现在将一切默认为 `NOUN` 并明确标注这个局限性。

完整代码见 `code/preprocess.py`。

---

## 4. 工业工具

### 4.1 英文：NLTK 与 spaCy

**NLTK**——教学、研究、组件可自由替换：

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)  # 处理缩写、Unicode、边界情况
stems = [PorterStemmer().stem(t) for t in tokens]

# 词形还原需要做 POS 标签转换：Penn Treebank → WordNet
def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"): return "v"
    if tag.startswith("J"): return "a"
    if tag.startswith("R"): return "r"
    return "n"

tagged = pos_tag(tokens)
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag))
          for t, tag in tagged]
print(lemmas)  # ['The', 'cat', 'be', 'run', '.']
```

上面那段 POS 标签转换代码是大多数教程省略的关键步骤。NLTK 的 `pos_tag` 输出 Penn Treebank 标签（如 `VBG`），而 `WordNetLemmatizer` 需要 WordNet 的缩写（`v`、`n`、`a`、`r`）。不转换就等效于所有词都被当作名词处理——`running` 永远不会变成 `run`。

**spaCy**——生产环境首选，速度快，开箱即用：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(f"{token.text:<10} {token.lemma_:<8} {token.pos_:<6}")
```

```text
The        the      DET
cats       cat      NOUN
were       be       AUX
running    run      VERB
.          .        PUNCT
```

spaCy 把整个流水线封装在 `nlp(text)` 中——分词、词性标注、词形还原一次完成。比 NLTK 快，开箱准确率更高。代价是难以单独替换某个组件。

### 4.2 中文：jieba——三行代码解决分词

```python
import jieba

text = "我们在北京大学学习自然语言处理"
tokens = list(jieba.cut(text))
print("/".join(tokens))
# 我们/在/北京大学/学习/自然语言处理
```

jieba 内部使用的是**基于前缀词典的詞图扫描**——比最大匹配复杂得多，但原理的起点就是你在第 2 步实现的正向最大匹配。它构建一个有向无环图（DAG），在所有可能的分词路径中，用动态规划找出概率最大的那条路。

其他中文工具对照：

| 工具 | 特点 | 适用场景 |
|---|---|---|
| jieba | 轻量、快、Python 原生 | 通用场景，快速原型 |
| pkuseg | 北大出品，准确率高于 jieba | 学术、高准确率要求 |
| HanLP | 多任务（分词、词性、NER、句法） | 企业级应用 |
| LAC (百度) | 分词 + 词性标注一体 | 中文词性标注需求 |

### 4.3 选择策略

| 场景 | 推荐方案 |
|---|---|
| 教学、研究、组件可替换 | NLTK（英文）/ jieba（中文） |
| 生产环境、多语言、要求速度 | spaCy（英文）/ HanLP（中文） |
| Transformer 流水线 | 跳过经典预处理，直接用 `transformers.AutoTokenizer` |

---

## 5. 知识连线

本课学习的分词与预处理，是后续 NLP 课程的第一道工序：

- **阶段 05 · 07（词性标注）**：词形还原的准确度直接依赖词性标注——没有正确的 POS Tag，`lemmatize("running")` 可能返回 `"running"` 而非 `"run"`
- **阶段 07（Transformer 深入）**：你会看到 Transformer 模型将分词逻辑"烧结"到了模型中——BPE 和 WordPiece 替代了正则分词器，子词元替代了词级词元。但理解经典分词，你才能理解为什么 BPE 是一种更好的"分词"方案
- **阶段 10 · 01（分词器）**：你会从零实现 BPE——本节课的正则分词和最大匹配是那个实现的最直接参考系

---

## 6. 工程最佳实践

### 6.1 工业界预处理流水线

```
原始文本 → 归一化 → 分词 → 去除停用词（可选）→ 词干/词形还原（可选）→ 特征提取
```

### 6.2 中文特别建议

- **繁简转换放在分词之前**——先用 `opencc` 将繁体转为简体，再分词。jieba 对繁体词表的支持远不如简体
- **全角半角统一**——中文文本中常混有全角英数字（`ＡＢＣ１２３`），预处理第一步就做全角 → 半角转换
- **不使用英文的词干提取器处理中文**——中文没有屈折后缀。"跑了"的"了"是时态助词，不是后缀。砍掉会导致语义丢失
- **分词词典要覆盖领域术语**——jieba 默认词典对通用文本足够，但对医疗、法律、金融等垂直领域，必须加载自定义词典。`jieba.load_userdict("domain_dict.txt")`

### 6.3 踩坑经验

- **训练和推理预处理不一致**——训练时做了小写化 + 去停用词，推理时漏了其中一步。这是生产环境 NLP 故障的头号原因。**解决：** 将预处理函数打包进模型包，训练和推理调用完全相同的代码
- **NLTK/spaCy 版本升级会悄悄改变分词行为**——spaCy 2.x 把 `don't` 切成 `['do', "n't"]`，3.x 可能保留为 `["don't"]`。模型训练在一个分布上，推理在另一个分布上，准确率下降但没人知道为什么。**解决：** 锁定版本号，写 20 条样本的分词回归测试
- **中文分词结果的"/ "分隔符陷阱**——`"/".join(tokens)` 看起来方便，但如果原文中真的有 `/`，就无法区分了。输出调试信息时用 `|` 或 `·` 作为分隔符
- **不要盲目去停用词**——经典 NLP 教程教你去掉"的"、"了"、"是"，但在现代情感分析和语义理解任务中，这些词承载了重要的语气和时态信息。先跑 baseline，确认去掉不影响指标再去掉

---

## 7. 常见错误

### 错误 1：正则分词器遇到缩写和连字符就崩

**现象：** `"don't"` 被切成 `["don", "'", "t"]`，`"state-of-the-art"` 被切成 5 个碎片。

**原因：** 正则规则没有覆盖撇号和连字符。分词器不认识这些"特殊字符"，把 `'` 当成了标点符号。

**修复：**
```python
# ❌ 过于简陋——撇号被视为标点
WORD_RE = re.compile(r"\w+|[^\w\s]")

# ✓ 优先匹配带内部撇号和连字符的单词
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+)*|[0-9]+|[^\sA-Za-z0-9]")
```

### 错误 2：对中文使用词干提取

**现象：** 试图用 Porter Stemmer 处理中文分词结果，`"学习过"` 变成 `"学习"`——看起来像是去掉了"过"。

**原因：** 这不是词干提取，这是碰巧去掉了助词。中文没有屈折后缀，Porter 算法对中文字符完全没有意义——它看不懂 Unicode 中文字符，只是什么都没做而已。

**修复：** 中文不需要词干提取。如果想去掉"了"、"着"、"过"等时态助词，应该用专门的**停用词过滤**或**词性筛选**，而不是词干提取。

### 错误 3：词形还原时不传 POS Tag

**现象：** `WordNetLemmatizer().lemmatize("running")` 返回 `"running"` 而非 `"run"`。

**原因：** WordNetLemmatizer 默认将所有词当作名词处理。`"running"` 作为名词（如 "I went for a running"）确实不需要还原。

**修复：**
```python
# ❌ 不传 POS Tag——所有词被当作名词
lemmatizer.lemmatize("running")              # 'running'

# ✓ 传入正确的 POS Tag
lemmatizer.lemmatize("running", pos="v")     # 'run'
```

---

## 8. 面试考点

### Q1：词干提取和词形还原的区别是什么？各举一个失败案例。（难度：⭐⭐）

**参考答案：**
词干提取基于规则砍后缀，速度快但激进——`organization → organ` 完全丢失了原意。词形还原基于词典和形态分析，准确但需要词性标注——`running → run` 需要知道它是动词。

失败案例：词干提取把 `university` 和 `universe` 归并到 `univers`（过度归并）。词形还原没有 POS 标签时，`"better"` 作为形容词应该还原为 `"good"`，作为动词则应该保留——没有标签就无法判断。

### Q2：中文分词为什么比英文分词难？正向最大匹配有什么问题？（难度：⭐⭐）

**参考答案：**
英文用空格天然分词，中文没有。中文分词的难点在于**歧义消解**——同样的字符串可能有多种切分方式。正向最大匹配贪心地取最长匹配，遇到"研究生命"时会错误地切成"研究生/命"而非"研究/生命"。这是因为 FMM 从左到右的贪心策略无法预判全局最优。

### Q3：为什么 Transformer 时代还要学经典文本预处理？（难度：⭐⭐⭐）

**参考答案：**
三点原因。第一，不是所有 NLP 任务都用 Transformer——搜索引擎、文本分类、传统对话系统中，经典预处理仍然是最快最省资源的方案。第二，理解正则分词和最大匹配是理解 BPE 和 WordPiece 的基础——子词分词的起点就是"比词更细的粒度"。第三，中文分词仍然需要——BERT 的中文 Tokenizer 本质上是字级别 + 子词的混合，对于需要词级别的下游任务（如关键词提取），jieba 仍然不可替代。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 词元 (Token) | "就是一个词" | 模型消费的最小单元。可以是词、子词、字符或字节，取决于分词策略 |
| 分词 (Tokenization) | "把文本切开" | 将连续文本切分为离散词元序列——这是模型理解的唯一入口 |
| 词干 (Stem) | "词的根" | 基于规则砍掉后缀后的结果。**不一定是真实单词**（如 `poni`） |
| 词形还原 (Lemma) | "词的词典形式" | 需要语法知识才能正确还原。`ran → run`，`better → good`。中文几乎不涉及 |
| 词性标注 (POS Tag) | "标记词性" | 如 NOUN、VERB、ADJ。词形还原的准确度依赖于此 |
| 分词歧义 | "一个句子有几种切法" | 中文分词的根源问题。`"研究生命"`可以切成"研究生/命"或"研究/生命" |
| 训练/推理不一致 | "部署完模型就坏了" | 训练和推理用了不同的预处理逻辑，这是生产 NLP 的头号故障 |

---

## 📚 小结

文本预处理是 NLP 流水线的第一道关口——分词决定了模型能看到什么，词干提取和词形还原决定了模型如何理解词的边界。你从零实现了英文正则分词器和中文最大匹配分词器，理解了 Porter 词干提取的规则顺序，以及词形还原对词性标注的依赖。

中文 NLP 的核心不在词干和词形还原，而在**分词**——一个没有标准答案的问题，每一步都是权衡。下一课我们将学习词性标注，它是词形还原的前提，也是命名实体识别的基石。

---

## ✏️ 练习

1. 【理解】用自己的话解释：为什么中文 NLP 需要分词而英文 NLP 可以在很多场景下跳过这一步？写 150 字以内的说明，让一个没有 NLP 背景的工程师也能理解。

2. 【实现】扩展正向最大匹配和逆向最大匹配，实现**双向最大匹配（BiMM）**——当 FMM 和 BMM 结果不一致时，选择词数更少的方案。用 10 条中文句子对比三种方法的分词结果。

3. 【实现】实现 Porter 词干提取的 Step 1b。如果词中包含元音且以 `ed` 或 `ing` 结尾，移除该后缀。处理双写辅音规则（`hopping → hop`，而非 `hopp`）。

4. 【实验】用 jieba 分词 100 条中文新闻标题。统计有多少条的分词结果在去停用词后，jieba 的默认词典和加载了自定义词典后的结果不同。解释差异原因。

5. 【思考】如果做一个中英文混合的聊天机器人，预处理流水线应该如何设计？需要考虑哪些英文 NLP 不需要、中文 NLP 单独不需要、但混合场景下新增的问题？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 预处理流水线 | `code/preprocess.py` | 从零实现的英文/中文预处理，含分词、词干提取、词形还原 |
| 可复用提示词 | `outputs/prompt-preprocessing-advisor.md` | 根据 NLP 任务推荐预处理方案（中英文均支持） |

---

## 📖 参考资料

1. [论文] Porter, M.F. "An algorithm for suffix stripping". Program, 1980. https://tartarus.org/martin/PorterStemmer/def.txt — 五页纸，至今仍是词干提取最清晰的解释
2. [官方文档] spaCy. "Linguistic Features". https://spacy.io/usage/linguistic-features — 一个真实 NLP 流水线的完整布线
3. [官方文档] jieba. "jieba 分词". https://github.com/fxsjy/jieba — 最流行的中文分词工具
4. [GitHub] NLTK. "Natural Language Toolkit". https://github.com/nltk/nltk — NLTK Book 第 3 章覆盖了你可能还没想到的分词边界情况
5. [论文] Xue, N. "Chinese Word Segmentation as Character Tagging". COLING, 2003. https://aclanthology.org/W03-1728/ — 将中文分词视为序列标注问题的经典视角

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文案例、中文分词内容、工程最佳实践、常见错误、面试考点等均为原创内容。
