"""视频理解管道——多向量场景索引脚手架。

核心架构原语是每场景多向量索引，包含三种表示（描述、帧、转录）。
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
    Scene("vid_001", 0, 0, 32000, "sunrise over skyline", "we start here in tokyo", "skyline dawn orange haze"),
    Scene("vid_001", 1, 32000, 68000, "busy intersection with pedestrians", "shibuya crossing", "street cars traffic signal"),
    Scene("vid_001", 2, 68000, 132000, "cars stopped at a red light", "count the vehicles", "cars red light queue lanes"),
    Scene("vid_001", 3, 132000, 170000, "chef pouring then stirring", "first we pour then stir slowly", "chef pan stove pour stir"),
    Scene("vid_001", 4, 170000, 210000, "chef plating the dish", "plated presentation", "plate garnish spoon dish"),
    Scene("vid_002", 0, 0, 40000, "ocean waves at sunset", "evening at the shore", "ocean waves sunset sky"),
]


def multi_vector_search(query: str, scenes: list[Scene], k: int = 5) -> list[tuple[Scene, float]]:
    qv = fake_embed(query)
    fused = defaultdict(float)
    index = {}
    for ranks in (sorted(scenes, key=lambda s: -cosine(qv, s.caption_emb)),
                  sorted(scenes, key=lambda s: -cosine(qv, s.frame_emb)),
                  sorted(scenes, key=lambda s: -cosine(qv, s.transcript_emb))):
        for rank, sc in enumerate(ranks):
            k_ = (sc.video_id, sc.scene_id)
            fused[k_] += 1.0 / (60 + rank + 1)
            index[k_] = sc
    return [(index[k_], s) for k_, s in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def ground_window(query: str, scene: Scene) -> tuple[int, int]:
    q = set(tokenize(query))
    t = tokenize(scene.transcript)
    if not q or not t:
        return scene.start_ms, scene.end_ms
    pos = [i for i, w in enumerate(t) if w in q]
    if not pos:
        return scene.start_ms, scene.end_ms
    span = scene.end_ms - scene.start_ms
    return (int(scene.start_ms + span * max(0.0, min(pos)/max(1,len(t))-0.05)),
            int(scene.start_ms + span * min(1.0, (max(pos)+1)/max(1,len(t))+0.05)))


def fmt(ms: int) -> str:
    s = ms // 1000
    return f"{s//60:02d}:{s%60:02d}"


def main() -> None:
    scenes = SAMPLE
    for s in scenes:
        s.embed()
    for q, _ in [("how many cars", False), ("pour or stir", False), ("ocean sunset", True)]:
        print(f"\nQ: {q}")
        hits = multi_vector_search(q, scenes, k=2)
        for sc, score in hits:
            print(f"  {sc.video_id}/{sc.scene_id} @[{fmt(sc.start_ms)}-{fmt(sc.end_ms)}] score={score:.4f}")
        start, end = ground_window(q, hits[0][0])
        print(f"  定位: [{fmt(start)}-{fmt(end)}]")


if __name__ == "__main__":
    main()
