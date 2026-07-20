"""WEAT 风格嵌入偏见探针。"""
import math


class EmbeddingBiasProbe:
    def __init__(self):
        self.embeddings = {
            "医生": [0.8, 0.2, 0.1], "护士": [0.2, 0.8, 0.1],
            "工程师": [0.7, 0.3, 0.6], "男性": [0.7, 0.3, 0.5],
            "女性": [0.3, 0.7, 0.5],
        }

    def cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x**2 for x in a))
        nb = math.sqrt(sum(x**2 for x in b))
        return dot / (na * nb) if na * nb > 0 else 0

    def measure(self, targets, attrs):
        scores = [self.cosine(self.embeddings[t], self.embeddings[a])
                  for t in targets for a in attrs
                  if t in self.embeddings and a in self.embeddings]
        return sum(scores) / len(scores) if scores else 0

    def report(self):
        m = self.measure(["男性"], ["医生", "工程师"])
        f = self.measure(["女性"], ["医生", "工程师"])
        print(f"男性-医疗: {m:.3f}  女性-医疗: {f:.3f}  差距: {m-f:.3f}")


if __name__ == "__main__":
    EmbeddingBiasProbe().report()
