# BPE 分词器从零实现
# 演示字节级 BPE 的完整训练和编码流程

from collections import Counter


class BPETokenizer:
    """从零实现的字节级 BPE 分词器。"""

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        """统计所有相邻 token 对的频率。"""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        """将指定对合并为新 token。"""
        merged, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        """在文本上训练 BPE——迭代合并最频繁的相邻对。"""
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for step in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + step
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if (step + 1) % 10 == 0:
                print(f"  合并 {step+1}: {best_pair} -> {new_token}")

        return self

    def encode(self, text):
        """编码文本为 token ID 序列。"""
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    def decode(self, tokens):
        """将 token ID 序列解码回文本。"""
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)


def analyze_tokenizer(tokenizer, test_texts):
    """分析分词器的压缩效率。"""
    total_tokens = 0
    total_chars = 0
    for text in test_texts:
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
        total_chars += len(text)

    print(f"词表大小: {tokenizer.vocab_size()}")
    print(f"总词元数: {total_tokens}")
    print(f"总字符数: {total_chars}")
    print(f"平均每字符词元数: {total_tokens / total_chars:.2f}")


if __name__ == "__main__":
    # 训练语料
    corpus = (
        "The cat sat on the mat. The cat ate the rat. "
        "The dog sat on the log. The dog ate the frog. "
        "Natural language processing is the study of how computers "
        "understand and generate human language. "
        "Tokenization is the first step in any NLP pipeline."
    )

    print("训练 BPE 分词器（40 次合并）...")
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=40)

    # 测试
    test_texts = [
        "The cat sat on the mat.",
        "Natural language processing",
        "unhappiness",
        "Hello 你好 World 🌍",
    ]

    print("\n编码测试:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        raw_bytes = len(text.encode("utf-8"))
        ratio = len(encoded) / raw_bytes
        passed = "PASS" if decoded == text else "FAIL"
        print(f"  '{text}' -> {len(encoded)} tokens (ratio: {ratio:.2f}) [{passed}]")

    # 分析
    print("\n词表分析:")
    analyze_tokenizer(tokenizer, test_texts)
