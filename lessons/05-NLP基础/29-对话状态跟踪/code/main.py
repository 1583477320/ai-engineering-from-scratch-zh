# 基于规则的对话状态追踪——槽位提取 + 状态更新循环
# 对应课程：阶段 05 · 29

import re

CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza"],
    "chinese": ["chinese", "noodles", "dumplings"],
    "indian": ["indian", "curry", "naan"],
}
AREA_SYNONYMS = {
    "north": ["north", "northern"],
    "south": ["south", "southern"],
    "east": ["east", "eastern"],
    "west": ["west", "western"],
    "center": ["center", "central", "downtown"],
}
PRICE_SYNONYMS = {
    "cheap": ["cheap", "affordable", "inexpensive", "budget", "便宜", "不贵"],
    "moderate": ["moderate", "medium", "mid-range", "中档", "差不多"],
    "expensive": ["expensive", "pricey", "fancy", "high-end", "贵"],
}
CORRECTION_CUES = {"算了", "不对", "等等", "改成", "换", "actually", "no wait", "change"}
NEGATION_CUES = {"不要", "算了", "取消", "去掉", "never mind"}


def extract_cuisine(utterance):
    for canonical, syns in CUISINE_SYNONYMS.items():
        if any(s in utterance.lower() for s in syns):
            return canonical
    return None


def extract_area(utterance):
    for canonical, syns in AREA_SYNONYMS.items():
        if any(s in utterance.lower() for s in syns):
            return canonical
    return None


def extract_price(utterance):
    for canonical, syns in PRICE_SYNONYMS.items():
        if any(s in utterance.lower() for s in syns):
            return canonical
    return None


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)


def is_negation(utterance):
    return any(cue in utterance.lower() for cue in NEGATION_CUES)


SLOT_EXTRACTORS = {
    "cuisine": extract_cuisine,
    "area": extract_area,
    "price": extract_price,
}


def update_state(state, utterance):
    """核心更新循环——三个不变量：(1)不碰未提及的槽位 (2)修正覆盖而非追加 (3)显式否定清除。"""
    new_state = dict(state)
    if is_negation(utterance):
        for slot in SLOT_EXTRACTORS:
            if slot in new_state: del new_state[slot]
        return new_state
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value  # ADD 或 UPDATE
    return new_state


def main():
    dialogue = [
        "我想找北区便宜点的餐馆",
        "改成中等价位的吧",
        "再加意大利菜",
        "算了不要意大利菜了，改中国菜吧",
    ]
    state = {}
    print("=== 对话状态跟踪 ===")
    for i, turn in enumerate(dialogue):
        state = update_state(state, turn)
        is_corr = "← 修正" if is_correction(turn) else ""
        print(f"  [{i+1}] '{turn}' {is_corr}")
        print(f"       状态: {state if state else '(空)'}")
        print()

    print("最终状态:", state)
    print("\n注意：玩具正则——生产中用 LLM + Pydantic + Instructor。")
    print("破坏性槽位（金额/日期）的 UPDATE 必须经过二次确认。")


if __name__ == "__main__":
    main()
