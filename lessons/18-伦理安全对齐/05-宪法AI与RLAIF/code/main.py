"""宪法 AI 批评-修订循环模拟。"""


def critique_revise(response, harmful_tokens, principle="避免有害内容"):
    found = [t for t in harmful_tokens if t in response]
    critique = f"违反原则'{principle}': 包含 {found}" if found else "通过"
    revised = response
    for t in harmful_tokens:
        revised = revised.replace(t, "[REDACTED]")
    return {"critique": critique, "revised": revised, "modified": len(found) > 0}


if __name__ == "__main__":
    harmful = ["攻击", "仇恨", "歧视"]
    for resp in ["我主张攻击不同意见的人", "今天天气很好", "对仇恨言论零容忍"]:
        r = critique_revise(resp, harmful)
        print(f"输入: {resp}")
        print(f"  批评: {r['critique']}  修订: {r['revised']}")
