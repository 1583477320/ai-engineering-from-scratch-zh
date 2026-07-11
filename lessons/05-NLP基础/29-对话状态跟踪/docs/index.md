# 对话状态跟踪

> "我想要北区便宜点的餐馆…算了改成中等价位…再加意大利菜。"三轮对话、三次状态更新。DST 保持槽位-值字典的同步——让预订操作在执行时拿到正确的参数。

**类型：** 实现课 | **语言：** Python | **前置知识：** 阶段 05 · 17（聊天机器人）、05 · 20（结构化输出） | **预计时间：** ~75 分钟 | **所处阶段：** Tier 1

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 实现 DST 的三种槽位操作——ADD/UPDATE/DELETE——从多轮对话中维护状态字典
- [ ] 用 Pydantic + Instructor 搭建 LLM 驱动的 DST——5 行代码，生产可用
- [ ] 理解 JGA（联合目标准确率）——全有或全无的指标——为什么 MultiWOZ 2026 的 SOTA 也才 83%

---

## 1. 问题

在任务导向对话系统中，用户目标被编码为一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每轮对话都可能 ADD/UPDATE/DELETE 一个槽位：

```
Turn 1: "北区便宜点的餐馆" → {area: north, price: cheap}
Turn 2: "改成中等价位"      → {area: north, price: moderate}  ← UPDATE
Turn 3: "再加意大利菜"       → {area: north, price: moderate, cuisine: italian}  ← ADD
Turn 4: "算了不要意大利菜了"  → {area: north, price: moderate}  ← DELETE
```

搞错一个槽位 → 订错餐厅、安排错航班、扣错款。DST 是用户说了什么和后端执行了什么之间的铰链。

**2026 年 LLM 时代为什么仍然重要：** 合规敏感域（银行、医疗、航空）要求确定性槽位值——不是自由形式生成。工具调用智能体在调用 API 前仍需槽位解析。多轮修正在"算了，改成周四吧"上仍然比看起来更难。

现代流水线：经典 DST 概念 + LLM 提取器 + 结构化输出 Guardrails。

---

## 2. 概念

### 2.1 任务结构

Schema 定义领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以为空、填充封闭集合中的值（价格：{cheap, moderate, expensive}）、或自由形式的值（名称："The Copper Kettle"）。

### 2.2 两种 DST 形式

- **分类式。** 对每个 (槽位, 候选值) 预测 yes/no。适用于封闭词表槽位。2020 年前的标准
- **生成式。** 给定对话，以自由文本生成槽位值。适用于开放词表槽位。**现代默认**

### 2.3 指标——JGA（联合目标准确率）

**所有槽位都正确的对话轮次比例。** 全有或全无——一个槽位错误 = 整个轮次错误。MultiWOZ 2.4 排行榜 2026 年顶部约 83%。

### 2.4 五种架构

1. **基于规则（槽位正则 + 关键词）。** 窄领域的强基线。可调试
2. **TripPy / BERT-DST。** 基于拷贝的生成 + BERT 编码。LLM 之前的标准
3. **LDST（LLaMA + LoRA）。** 指令微调 LLM + 领域槽位 prompt。在 MultiWOZ 2.4 上达到 ChatGPT 级别
4. **无语料库（2024-26）。** 跳过 Schema；直接生成槽位名称和值。处理开放域
5. **Prompt + 结构化输出（2024-26）。** LLM + Pydantic Schema + 约束解码。**5 行代码，生产可用**

### 2.5 经典失败模式

- **跨轮指代消解。** "就选第一个吧"——需要消解"第一个"是哪个选项
- **覆盖 vs 追加。** 用户说"加意大利菜"——是替换菜系还是追加？
- **隐式确认。** "行吧"——这算是接受了预订吗？
- **修正。** "算了改 7 点吧"——必须更新时间而不清除其他槽位
- **指代前一条系统消息。** "对，就那个"——"那个"是哪个？

---

## 3. 从零实现

### 第 1 步：基于规则的槽位提取器

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}

def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词表之外脆弱。在确定性槽位确认中有效。

### 第 2 步：状态更新循环——三个不变量

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value          # UPDATE 或 ADD
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None           # DELETE
    return new_state
```

**三个不变量：(1)** 永远不重置用户没碰过的槽位。**(2)** 显式否定（"不要意大利菜了"）必须清除。**(3)** 用户修正（"算了…"）必须覆盖而非追加。

### 第 3 步：LLM 驱动的 DST + 结构化输出

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None

def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

**Instructor + Pydantic 保证有效的状态对象。** 没有正则、没有 Schema 不匹配、没有幻觉槽位。5 行代码，生产可用。

### 第 4 步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

**校准：系统在所有槽位上全对的轮次比例是多少？** MultiWOZ 2.4 上 2026 顶部系统：80-83%。你的领域系统在窄词表上应该超过这个数字——否则 LLM 基线就打败你了。

### 第 5 步：处理修正

```python
CORRECTION_CUES = {"算了", "不对", "等等", "改成", "换"}

def is_correction(utterance):
    return any(cue in utterance for cue in CORRECTION_CUES)
```

检测到修正时——覆盖最后更新的槽位而非追加。**现代模式：始终让 LLM 从历史中重新生成完整状态而非增量更新——这自然地处理了修正。**

---

## 4. 陷阱

- **完整历史重新生成的成本。** 让 LLM 每轮重新生成状态消耗 O(n²) 总 token。限制历史或摘要旧轮次
- **Schema 漂移。** 事后添加新槽位会破坏旧的训练数据。给你的 Schema 加版本号
- **大小写。** "Italian" vs "italian" vs "ITALIAN"——处处归一化
- **隐式继承。** 如果用户之前指定了"4 个人"——一个新时间请求不应该清空人数。始终传入完整历史
- **自由形式 vs 封闭集合。** 名称、时间、地址需要自由形式槽位；菜系和区域是封闭的。在 Schema 中混合两者

---

## 5. 工业工具——2026 技术栈

| 场景 | 方案 |
|---|---|
| 窄领域（一两个意图） | 基于规则 + 正则 |
| 宽领域、有标注数据 | LDST（LLaMA + LoRA 在 MultiWOZ 风格数据上） |
| 宽领域、无标注、生产可用 | LLM + Instructor + Pydantic Schema |
| 语音 | ASR + 归一化器 + LLM-DST |
| 多领域预订流 | Schema 引导的 LLM + 每领域 Pydantic 模型 |
| 合规敏感 | 基于规则作主模型 + LLM 回退 + 确认流程 |

---

## 6. 常见错误

### 错误 1：增量更新丢失跨轮上下文

**现象：** 第 3 轮 LLM 回读的 price 值与第 1 轮不同——"moderate"被误记为"中档"。

**原因：** 增量更新只给 LLM 最新一轮的对话——丢失了原始值的来源。LLM 在第 3 轮的上下文中没有看到第 1 轮用户说"中等价位"时的确切措辞。

**修复：** 始终传入完整对话历史——让 LLM 从完整上下文中重新生成状态。对于长对话（>10 轮）——摘要旧轮次为关键状态信息。

### 错误 2：中文"确认"和"不关心"的混淆

**现象：** 系统问"意大利菜可以吗？"——用户答"都可以"——系统将 cuisine UPDATE 为 italian。

**原因：** "都可以"在中文中可以表示接受，也可以表示"我不在意"——在这里是后者——用户没有表达对意大利菜的偏好——只是不反对。系统误将"不反对"理解为"确认"。

**修复：** 区分确认和不关心的信号："行"/"好"/"可以"表示确认。"都行"/"随便"/"无所谓"/"都可以"表示不指定——不 UPDATE 相应槽位。在 LLM prompt 中显式编码这个区分。

---

## 7. 面试考点

### Q1：JGA 为什么是"全有或全无"的指标？（难度：⭐⭐）

**参考答案：**
因为后端系统在所有槽位都正确之前不能执行操作。如果 state={cuisine: italian, area: north, price: WRONG}——系统仍然订了错误的餐厅。3 个槽位中对 2 个不等于每 3 次预订中有 1 次正确——每次预订都被那个错误槽位毁了。JGA 的"全有或全无"反映了这个现实——但 MultiWOZ 2026 的 83% JGA 意味着 17% 的对话轮次中至少有一个槽位是错误的——这是任何面向用户部署之前需要解决的实际问题。

### Q2：为什么"始终重新生成完整状态"比"增量更新"更可靠？（难度：⭐⭐⭐）

**参考答案：**
两个原因：**(1)** 修正处理——"算了改 7 点吧"需要同时 UPDATE time 和保持其他槽位。增量更新可能无法区分"覆盖 time"和"清除其他所有槽位"。完整重新生成——LLM 从头重写整个 state JSON——自然地保持未被提及的槽位不变。**(2)** 隐式继承——如果用户在 5 轮前指定了"4 个人"——增量更新（只看最新一轮）已经丢失了这个信息。完整历史重新生成保留了全部上下文。代价是 O(n²) 的 token 消耗——对短对话（< 10 轮）完全可以接受。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| DST | "对话状态跟踪" | 在多轮对话间维护槽位-值字典 |
| 槽位 (Slot) | "用户意图的一个参数" | 后端需要的命名参数（菜系、日期） |
| 领域 (Domain) | "任务区" | 餐厅、酒店、出租车——槽位的集合 |
| JGA | "联合目标准确率" | 每个槽位都正确的对话轮次比例。全有或全无 |
| MultiWOZ | "DST 的标准基准" | 多领域 WOZ 数据集；标准 DST 评估 |
| 无语料库 DST | "不需要 Schema" | 直接生成槽位名称和值——不依赖预定义列表 |
| 修正 (Correction) | "'算了…'" | 覆盖之前填充的槽位的对话轮次 |

---

## 📚 小结

DST = ADD/UPDATE/DELETE 三种操作在对话轮次间维护槽位-值字典。现代流水线 = LLM + Pydantic + Instructor——5 行代码，生产可用。**始终重新生成完整状态而非增量更新**——这自然地处理修正和隐式继承。JGA 是全有或全无的指标——MultiWOZ 2026 SOTA 也才 83%——那个 17% 的差距是每个面向用户系统都需要解决的实际问题。

中文 DST 两大额外挑战：口语化值映射（"差不多就行"→ moderate）、区分"确认"和"不关心"（"都可以"≠"行"）。破坏性槽位永远不单靠 LLM——结构化确认是硬性要求。

---

## ✏️ 练习

1. 【理解】为 3 个槽位（菜系、区域、价格）构建 `code/main.py` 中的基于规则的状态追踪器。在 10 段手写对话上测试。衡量 JGA。

2. 【实现】同一数据集上用 Instructor + Pydantic + 小 LLM。对比 JGA。检查最难的轮次。

3. 【实验】实现两者并路由：基于规则主模型 + 当规则输出 < 2 个槽位时 LLM 回退。衡量组合 JGA 和每轮推理成本。

4. 【思考】你的 DST 系统在"都行"回答上反复出错（将 cuisine UPDATE 为错误的值）。如何仅用 prompt 修改（不改代码）来降低这个错误率？

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| 可复用提示词 | `outputs/skill-dst-designer.md` | 按领域设计 DST Schema、提取器和更新策略的系统化方案 |

---

## 📖 参考资料

1. [论文] Budzianowski et al. "MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling". EMNLP, 2018. https://arxiv.org/abs/1810.00278 — 标准基准
2. [论文] Feng et al. "Towards LLM-driven Dialogue State Tracking (LDST)". EMNLP, 2023. https://arxiv.org/abs/2310.14970 — LLaMA + LoRA 指令微调 DST
3. [论文] Heck et al. "TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking". SIGdial, 2020. https://arxiv.org/abs/2005.02877 — 基于拷贝的 DST 主力
4. [排行榜] MultiWOZ. https://github.com/budzianowski/multiwoz — 标准 DST 结果

---

> 本课程参考了 AI Engineering From Scratch（MIT License）的课程体系，在此基础上进行了重构和原创内容的扩充。所有中文表达、中文 DST 挑战（口语化值映射、"确认"vs"不关心"区分）、工程最佳实践、常见错误、面试考点等均为原创内容。
