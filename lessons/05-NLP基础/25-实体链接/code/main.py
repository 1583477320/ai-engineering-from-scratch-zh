# 实体链接玩具——别名索引 + 上下文消歧
# 对应课程：阶段 05 · 25

import re
from collections import Counter

ALIAS_INDEX = {
    "jordan": ["Q41421", "Q810", "Q254110", "Q3308285"],
    "paris":  ["Q90", "Q663094", "Q55411"],
    "apple":  ["Q312", "Q89"],
    "washington": ["Q23", "Q1223", "Q61"],
    "python": ["Q28865", "Q83320"],
}
KB_DESC = {
    "Q41421": "Michael Jordan American basketball player Chicago Bulls six championships",
    "Q810":   "Jordan country Middle East kingdom capital Amman Arabic",
    "Q254110":"Michael B Jordan American actor Black Panther Creed film",
    "Q3308285":"Michael I Jordan Berkeley professor machine learning statistics",
    "Q90":    "Paris capital of France Eiffel Tower Seine river city",
    "Q663094":"Paris Texas city United States Lamar County",
    "Q55411": "Paris Hilton American socialite hotel heiress television personality",
    "Q312":   "Apple Inc American technology company iPhone Mac Tim Cook Cupertino",
    "Q89":    "Apple fruit tree species Malus domestica red green grown worldwide",
    "Q23":    "George Washington American founding father first president",
    "Q1223":  "Washington state Pacific northwest United States Seattle capital Olympia",
    "Q61":    "Washington DC capital city of United States federal district",
    "Q28865": "Python programming language Guido van Rossum interpreted dynamic",
    "Q83320": "Python snake nonvenomous constrictor species Asia Africa large",
}
PRIORS = {
    "jordan": {"Q41421": 0.70, "Q810": 0.15, "Q254110": 0.10, "Q3308285": 0.05},
    "paris": {"Q90": 0.80, "Q55411": 0.15, "Q663094": 0.05},
    "apple": {"Q312": 0.75, "Q89": 0.25},
}


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def disambiguate(mention, context):
    """Jaccard 上下文重叠消歧——教学版。生产中用嵌入余弦相似度。"""
    candidates = ALIAS_INDEX.get(mention.lower(), [])
    if not candidates: return None, 0.0
    ctx = tokenize(context)
    best, best_score = None, -1
    for eid in candidates:
        desc = tokenize(KB_DESC.get(eid, ""))
        union = len(ctx | desc)
        score = len(ctx & desc) / union if union else 0.0
        # 加入先验概率
        prior = PRIORS.get(mention, {}).get(eid, 0.01)
        score = score * prior
        if score > best_score: best, best_score = eid, score
    return best, best_score


def main():
    tests = [
        ("Jordan", "The basketball game was amazing and Michael scored 50 points."),
        ("Jordan", "The middle eastern country has a rich history and beautiful deserts."),
        ("Apple", "The company announced a new iPhone with better battery life."),
        ("Apple", "The fruit was red and juicy picked fresh from the tree."),
        ("Paris", "The Eiffel Tower is the most famous landmark in the city of lights."),
        ("Paris", "The socialite walked the red carpet at the movie premiere."),
    ]
    print("=== 实体链接——别名索引 + Jaccard 上下文消歧 ===")
    for mention, context in tests:
        eid, score = disambiguate(mention, context)
        desc = KB_DESC.get(eid, "?")[:60]
        print(f"\n  提及: '{mention}'")
        print(f"  上下文: {context[:60]}...")
        print(f"  消歧结果: {eid} ({desc}...) 得分={score:.3f}")

    print("\n注意：玩具 Jaccard——生产中用 BLINK 嵌入或 GENRE 约束解码。")


if __name__ == "__main__":
    main()
