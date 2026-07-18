"""PII 脱敏器——一致性令牌化。"""
import re


class PIIRedactor:
    def __init__(self):
        self.token_map = {}
        self.counter = 0
        self.patterns = {"身份证": r"\d{18}[Xx]?", "手机": r"1[3-9]\d{9}"}

    def redact(self, text):
        for name, pat in self.patterns.items():
            def repl(m):
                val = m.group(0)
                if val not in self.token_map:
                    self.counter += 1
                    self.token_map[val] = f"[{name}_{self.counter:04d}]"
                return self.token_map[val]
            text = re.sub(pat, repl, text)
        return text


if __name__ == "__main__":
    r = PIIRedactor()
    t1 = "身份证110101199001011234，手机13800138000"
    t2 = "身份证110101199001011234"
    print(f"1: {r.redact(t1)}")
    print(f"2: {r.redact(t2)}")
