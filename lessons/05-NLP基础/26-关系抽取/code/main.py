# 基于模式的玩具关系提取器——Hearst patterns + 正则
# 对应课程：阶段 05 · 26

import re
from collections import defaultdict

PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
    (r"(?P<s>[A-Z]\w+) acquired (?P<o>[A-Z]\w+)", "acquired"),
]


def extract_triples(text):
    """用预定义正则模式从文本中提取三元组。"""
    triples = []
    for pattern, relation in PATTERNS:
        for match in re.finditer(pattern, text):
            s, o = match.group("s"), match.group("o")
            triples.append((s, relation, o, match.span()))
    return triples


def build_graph(triples):
    """从三元组构建简单邻接表图。"""
    graph = defaultdict(list)
    for s, r, o, _ in triples:
        graph[s].append((r, o))
    return graph


def query_graph(graph, node, relation=None):
    """查询图的邻居——可选关系过滤。"""
    result = []
    for r, o in graph.get(node, []):
        if relation is None or r == relation:
            result.append((r, o))
    return result


def main():
    text = (
        "Tim Cook is the CEO of Apple. "
        "Steve Jobs founded Apple. "
        "Google acquired YouTube. "
        "Satya Nadella works at Microsoft. "
        "Elon Musk founded Tesla. "
        "Jeff Bezos founded Amazon."
    )
    print("=== 基于模式的关系提取 ===")
    triples = extract_triples(text)
    for s, r, o, span in triples:
        print(f"  ({s}, {r}, {o})  ← 位置 {span}")

    graph = build_graph(triples)
    print(f"\n=== 知识图谱查询 ===")
    print(f"  Apple 的关系: {query_graph(graph, 'Apple')}")
    print(f"  Tim Cook 的 worksAt: {query_graph(graph, 'Tim Cook', 'worksAt')}")
    print(f"  谁是 founder: {[n for n in graph if any(r == 'founded' for r, _ in graph[n])]}")

    print("\n注意：玩具正则——生产中用 REBEL 或 LLM+AEVS 锚定验证流水线。")


if __name__ == "__main__":
    main()
