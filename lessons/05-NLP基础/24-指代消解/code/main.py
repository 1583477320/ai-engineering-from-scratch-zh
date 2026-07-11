# 基于规则的代词消解器——教学实现
# 对应课程：阶段 05 · 24

import re

PRONOUNS = {
    "he": {"gender": "m", "number": "sg"}, "him": {"gender": "m", "number": "sg"},
    "his": {"gender": "m", "number": "sg"}, "she": {"gender": "f", "number": "sg"},
    "her": {"gender": "f", "number": "sg"}, "hers": {"gender": "f", "number": "sg"},
    "it": {"gender": "n", "number": "sg"}, "its": {"gender": "n", "number": "sg"},
    "they": {"gender": "u", "number": "pl"}, "them": {"gender": "u", "number": "pl"},
    "their": {"gender": "u", "number": "pl"},
}
FEMALE_FIRST = {"mary", "alice", "sarah", "emma", "jane"}
MALE_FIRST = {"john", "james", "david", "tim", "steve", "michael", "bob", "peter"}
NEUTER_HEADS = {"company", "firm", "product", "device", "phone", "laptop", "organization", "team"}


def extract_mentions(text):
    """提取提及：命名实体（首字母大写跨多词的词组）、代词、定指描述（the X）。"""
    mentions = []
    for sent_idx, sent in enumerate(re.split(r"(?<=[.!?])\s+", text)):
        tokens = re.findall(r"[A-Za-z]+|[^\s]", sent)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.lower() in PRONOUNS:
                mentions.append({"text": token, "start": i, "sent": sent_idx, "type": "pronoun"})
                i += 1
            elif token[0].isupper() and token.lower() not in {"the", "a", "an", "in", "on", "at", "and", "or"}:
                j = i + 1
                while j < len(tokens) and tokens[j][0].isupper():
                    j += 1
                phrase = " ".join(tokens[i:j])
                gender = "u"
                first = phrase.split()[0].lower()
                if first in FEMALE_FIRST: gender = "f"
                elif first in MALE_FIRST: gender = "m"
                mentions.append({"text": phrase, "start": i, "sent": sent_idx, "type": "named", "gender": gender})
                i = j
            else:
                i += 1
    return mentions


def resolve_pronouns(mentions):
    """对每个代词，向前找最近的一致提及。"""
    for i, m in enumerate(mentions):
        if m["type"] != "pronoun": continue
        p_info = PRONOUNS.get(m["text"].lower(), {"gender": "u", "number": "sg"})
        for j in range(i - 1, -1, -1):
            antecedent = mentions[j]
            if antecedent["type"] == "pronoun": continue
            ant_gender = antecedent.get("gender", "u")
            if (p_info["gender"] == "u" or ant_gender == "u" or p_info["gender"] == ant_gender):
                m["antecedent"] = antecedent["text"]
                break
    return mentions


def main():
    text = "John walked into the room. He sat down. Mary arrived later. She smiled at him."
    mentions = extract_mentions(text)
    resolved = resolve_pronouns(mentions)

    print("=== 基于规则的代词消解 ===")
    for m in resolved:
        if m["type"] == "pronoun":
            ant = m.get("antecedent", "未找到")
            print(f"  '{m['text']}' → '{ant}'")
        else:
            print(f"  [{m['type']}] '{m['text']}'")

    print("\n注意：此规则系统仅处理简单代词。")
    print("生产指代消解使用 span-based 端到端模型（Lee et al., 2017）。")


if __name__ == "__main__":
    main()
