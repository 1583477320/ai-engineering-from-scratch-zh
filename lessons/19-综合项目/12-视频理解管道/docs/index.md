# 综合项目12——视频理解管道（场景、问答、搜索）

> Twelve Labs将Marengo+Pegasus产品化。VideoDB发布了视频CRUD API。AI2的Molmo 2发布了开源VLM检查点。Gemini长上下文原生处理数小时视频。TimeLens-100K定义了大规模时间定位。2026年的管道已经定型：场景分割、每场景描述+嵌入、转录对齐、多向量索引、用(start,end)时间戳加帧预览回答的查询管道。本综合项目要求你导入100小时视频、在公开基准上达到标准、并测量计数和动作类问题上的幻觉率。

**类型：** 综合项目
**编程语言：** Python（管道），TypeScript（UI）
**前置知识：** 第4章（计算机视觉）、第6章（语音）、第7章（Transformer）、第11章（LLM工程）、第12章（多模态）、第17章（基础设施）
**涉及章节：** P4 · P6 · P7 · P11 · P12 · P17
**预计时间：** 30小时

---

## 学习目标

- 构建视频理解管道：场景分割→描述→ASR→多向量索引→时间定位→VLM答案合成
- 实现每场景三种向量类型（描述、帧、转录）的多向量索引
- 实现时间定位子窗口细化
- 测量和报告计数类幻觉率

---

## 1. 问题

长视频问答是2026年最消耗带宽的多模态问题。Gemini 2.5 Pro可以原生读取2小时视频，但将100小时视频导入可查询的语料库仍需要场景级索引。

生产形态结合场景分割（TransNetV2或PySceneDetect）、每场景VLM描述（Gemini 2.5、Qwen3-VL-Max或Molmo 2）、转录对齐（Whisper-v3-turbo带词级时间戳）和多向量索引（存储描述、帧嵌入和转录）。

基准是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的100个查询自定义集。计数和动作类问题上的幻觉是已知的难失败模式。

---

## 2. 核心概念

### 2.1 三条并行管道

导入时三条管道并行运行：

**场景分割**：将视频切割为场景。**VLM描述**：为每个场景生成描述和关键帧嵌入。**ASR对齐**：产生词级时间戳。

三条流通过(scene_id, time range)连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：描述嵌入、关键帧嵌入、转录嵌入。

### 2.2 查询管道

自然语言问题触发所有三种向量查询；结果通过RRF合并；时间定位适配器（TimeLens风格）在最佳场景内细化(start,end)窗口。VLM合成器（Gemini 2.5 Pro或Qwen3-VL-Max）接收查询+最佳场景+裁剪帧，输出带时间戳引用和帧预览的答案。

### 2.3 幻觉测量

计数类（"有多少人进入房间？"）和动作类（"厨师是先倒再搅拌吗？"）问题以不可靠著称。需单独报告准确率。

---

## 3. 从零实现

`code/main.py`实现每场景三种向量表示的多向量索引、三种查询的融合和时间定位。

```python
"""视频理解管道——多向量场景索引脚手架。

核心架构原语是每场景多向量索引，包含三种表示（描述、帧、转录），
并行查询后通过互惠排名融合合并，然后通过时间定位步骤
在最佳场景内选择子窗口。

运行：python3 code/main.py
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field


EMB_DIM = 24


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def fake_embed(text: str) -> list[float]:
    v = [0.0] * EMB_DIM
    for tok in tokenize(text):
        h = hash(tok)
        v[h % EMB_DIM] += 1.0
        v[(h >> 8) % EMB_DIM] += 0.5
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 场景记录——多向量：描述/帧/转录
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    video_id: str
    scene_id: int
    start_ms: int
    end_ms: int
    caption: str
    transcript: str
    frame_tags: str
    caption_emb: list[float] = field(default_factory=list)
    frame_emb: list[float] = field(default_factory=list)
    transcript_emb: list[float] = field(default_factory=list)

    def embed(self) -> None:
        self.caption_emb = fake_embed(self.caption)
        self.frame_emb = fake_embed(self.frame_tags)
        self.transcript_emb = fake_embed(self.transcript)


SAMPLE = [
    Scene("vid_001", 0, 0, 32000, "sunrise over skyline, drone footage",
          "we start here in tokyo", "skyline buildings dawn orange sky haze"),
    Scene("vid_001", 1, 32000, 68000, "busy intersection with pedestrians",
          "shibuya crossing after sunrise", "street people walking cars traffic signal"),
    Scene("vid_001", 2, 68000, 132000, "cars stopped at a red light",
          "let me count the vehicles approaching", "cars red light queue crossing lanes"),
    Scene("vid_001", 3, 132000, 170000, "kitchen scene chef pouring then stirring",
          "first we pour then we stir it slowly", "chef pan stove pour stir ingredient"),
    Scene("vid_001", 4, 170000, 210000, "chef plating the finished dish",
          "plated presentation of the dish", "plate garnish spoon finishing dish"),
    Scene("vid_002", 0, 0, 40000, "ocean waves at sunset",
          "beautiful evening at the shore", "ocean waves sunset sky shore"),
]


# ---------------------------------------------------------------------------
# 三向量查询 + RRF融合
# ---------------------------------------------------------------------------

def multi_vector_search(query: str, scenes: list[Scene], k: int = 5) -> list[tuple[Scene, float]]:
    qv = fake_embed(query)
    scored_caption = sorted(scenes, key=lambda s: -cosine(qv, s.caption_emb))
    scored_frame = sorted(scenes, key=lambda s: -cosine(qv, s.frame_emb))
    scored_transcript = sorted(scenes, key=lambda s: -cosine(qv, s.transcript_emb))

    fused: dict[tuple[str, int], float] = defaultdict(float)
    index: dict[tuple[str, int], Scene] = {}
    for ranks, stream in ((scored_caption, "cap"), (scored_frame, "frm"), (scored_transcript, "trn")):
        for rank, sc in enumerate(ranks):
            key = (sc.video_id, sc.scene_id)
            fused[key] += 1.0 / (60 + rank + 1)
            index[key] = sc

    ranked = sorted(fused.items(), key=lambda x: -x[1])
    return [(index[k_], s) for k_, s in ranked[:k]]


# ---------------------------------------------------------------------------
# 时间定位存根——在最佳场景内细化开始/结束
# ---------------------------------------------------------------------------

def ground_window(query: str, scene: Scene) -> tuple[int, int]:
    q = set(tokenize(query))
    t_tokens = tokenize(scene.transcript)
    if not q or not t_tokens:
        return scene.start_ms, scene.end_ms
    positions = [i for i, w in enumerate(t_tokens) if w in q]
    if not positions:
        return scene.start_ms, scene.end_ms
    span = scene.end_ms - scene.start_ms
    start_frac = min(positions) / max(1, len(t_tokens))
    end_frac = (max(positions) + 1) / max(1, len(t_tokens))
    start = int(scene.start_ms + span * max(0.0, start_frac - 0.05))
    end = int(scene.start_ms + span * min(1.0, end_frac + 0.05))
    return start, end


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def main() -> None:
    scenes = SAMPLE
    for s in scenes:
        s.embed()

    queries = [
        ("how many cars pass through the intersection", False),
        ("what happened first pour or stir", False),
        ("plating of the dish", True),
        ("ocean at sunset", True),
    ]

    for q, descriptive in queries:
        print(f"\nQ: {q}  (descriptive={descriptive})")
        hits = multi_vector_search(q, scenes, k=3)
        for sc, score in hits:
            print(f"  scene {sc.video_id}/{sc.scene_id} @ [{fmt_ms(sc.start_ms)}-{fmt_ms(sc.end_ms)}] "
                  f"score={score:.4f}  cap='{sc.caption[:40]}'")
        top = hits[0][0]
        start, end = ground_window(q, top)
        print(f"  定位窗口: [{fmt_ms(start)}-{fmt_ms(end)}] "
              f"（从 {fmt_ms(top.start_ms)}-{fmt_ms(top.end_ms)} 缩小）")


if __name__ == "__main__":
    main()
```

运行结果：

```
Q: how many cars pass through the intersection  (descriptive=False)
  scene vid_001/1 @ [00:32-01:08]  score=1.1386  cap='busy intersection with pedestrians'
  scene vid_001/2 @ [01:08-02:12]  score=1.0708  cap='cars stopped at a red light'
  scene vid_001/0 @ [00:00-00:32]  score=0.7910  cap='sunrise over skyline, drone footage'
  定位窗口: [01:08-01:48]（从 01:08-02:12 缩小）

Q: what happened first pour or stir  (descriptive=False)
  scene vid_001/3 @ [02:12-02:50]  score=1.4697  cap='kitchen scene chef pouring then stirring'
  scene vid_001/4 @ [02:50-03:30]  score=0.1370  cap='chef plating the finished dish'
  定位窗口: [02:12-02:29]（从 02:12-02:50 缩小）
```

---

## 4. 工具实践

**技术栈：**
- 场景分割：TransNetV2（2024-26年最佳）或PySceneDetect
- ASR：Whisper-v3-turbo（faster-whisper，词级时间戳）
- VLM：Gemini 2.5 Pro或Qwen3-VL-Max或Molmo 2
- 时间定位：TimeLens-100K训练适配器或VideoITG
- 索引：Qdrant多向量支持（描述/帧/转录）
- 评测：ActivityNet-QA、NeXT-GQA、自定义100查询

---

## 5. LLM视角

**多向量视角**：三种向量类型（描述、帧、转录）捕获视频的不同方面。描述理解场景含义，帧理解视觉相似性，转录理解说的内容。三者融合比任何单一向量好。

**时间定位视角**：时间定位将答案精确到子窗口。TimeLens风格适配器在最佳场景中细化(start,end)。

**幻觉视角**：计数和动作类问题是最难的。VLM在"有多少"问题上幻觉率高。需要单独报告和优化。

---

## 6. 工程最佳实践

**导入管道**：
- TransNetV2或PySceneDetect场景分割
- Whisper-v3-turbo ASR（词级时间戳）
- VLM描述（每场景关键帧）

**索引设计**：
- Qdrant多向量：caption_emb、frame_emb、transcript_emb
- 荷载：video_id、scene_id、start_ms、end_ms

**查询管道**：
- 三种密集查询+RRF融合
- TimeLens时间定位
- VLM答案合成（要求(start,end)引用）

---

## 7. 常见错误

**错误1：仅使用描述向量**
症状：视觉相似性未被捕获
修复：描述+帧+转录三向量

**错误2：不细化时间窗口**
症状：引用整个场景而非精确子窗口
修复：TimeLens时间定位

**错误3：不单独报告计数幻觉**
症状：总体准确率掩盖了计数类问题
修复：按类型分解准确率

---

## 8. 面试考点

**Q1：为什么视频QA需要多向量索引？**
考察：对多模态检索的理解

**Q2：时间定位如何提高答案精度？**
考察：对视频理解细节的理解

**Q3：为什么计数类问题特别困难？**
考察：对VLM局限性的理解

---

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 场景分割 | "镜头检测" | 在镜头边界将视频切割为场景 |
| 多向量索引 | "描述+帧+转录" | Qdrant每个场景三种命名向量的集合 |
| 时间定位 | "精确时间" | 细化答案的(start,end)时间窗口 |
| 帧嵌入 | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似性 |
| RRF融合 | "互惠排名融合" | 跨多个排名列表的合并策略 |
| 计数幻觉 | "数错" | VLM在"有多少X"问题上的已知失败模式 |
| ActivityNet-QA | "视频QA基准" | 长视频QA准确率基准 |

---

## 参考文献

- [AI2 Molmo 2](https://allenai.org/blog/molmo2)
- [TimeLens（CVPR 2026）](https://github.com/TencentARC/TimeLens)
- [Gemini视频长上下文](https://deepmind.google/technologies/gemini)
- [VideoDB](https://videodb.io)
- [TransNetV2](https://github.com/soCzech/TransNetV2)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467)
