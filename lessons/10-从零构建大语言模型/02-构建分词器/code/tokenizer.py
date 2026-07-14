# 生产级 BPE 分词器
# 字节级 BPE + 预分词 + 特殊 token + 多语言支持

import re
import unicodedata
from collections import Counter


# ============================================================================
# 第 1 步：GPT-2 风格预分词
# ============================================================================

# GPT-2 的正则表达式——按词边界拆分
# 安装 regex 模块以支持 Unicode 属性转义
try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )


def pre_tokenize(text):
    """将文本按词边界拆分为 chunks。"""
    return [match.group() for match in GPT2_PATTERN.finditer(text)]


# ============================================================================
# 第 2 步：特殊 token 处理
# ============================================================================

class SpecialTokenHandler:
    """管理特殊 token 的精确匹配和固定 ID。"""

    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        """按特殊 token 拆分文本——返回 (文本, 是否特殊token) 对。"""
        if not self.pattern:
            return [(text, False)]
        parts, last_end = [], 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts


# ============================================================================
# 第 3 步：BPE 合并核心
# ============================================================================

def apply_merge(byte_seq, pair, new_id):
    """将 byte_seq 中的 pair 替换为 new_id。"""
    merged, i = [], 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged


# ============================================================================
# 第 4 步：完整分词器
# ============================================================================

class ProductionTokenizer:
    """生产级 BPE 分词器——字节级 + 预分词 + 特殊 token。"""

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        """NFKC Unicode 规范化。"""
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        """训练 BPE——统计最频繁的相邻字节对并合并。"""
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        """添加特殊 token——固定 ID，不参与 BPE 合并。"""
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        """编码：规范化 -> 拆分特殊 token -> 预分词 -> BPE 合并。"""
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        """解码：ID -> 字节 -> UTF-8 文本。"""
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 训练语料
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox runs through the forest. "
        "Machine learning models process natural language. "
        "Deep learning transforms how we build software. "
        "def train(model, data): return model.fit(data) "
        "def predict(model, x): return model(x) "
    )

    print("训练生产级分词器（50 次合并）...")
    tok = ProductionTokenizer()
    tok.train(corpus, num_merges=50)

    # 添加特殊 token
    bos = tok.add_special_token("<|begin|>")
    eos = tok.add_special_token("<|end|>")
    print(f"特殊 token: BOS={bos}, EOS={eos}")

    # 多语言测试
    test_texts = [
        "The quick brown fox.",
        "你好世界",
        "Hello 🌍 World",
        "def foo(x): return x + 1",
        "<|begin|>Hello<|end|>",
    ]

    print("\n多语言编码测试:")
    for text in test_texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        print(f"  Input:  {text}")
        print(f"  Tokens: {len(ids)} ids -> {ids}")
        print(f"  Decoded: {decoded}")
        print()

    print(f"词表大小: {tok.vocab_size()}")
