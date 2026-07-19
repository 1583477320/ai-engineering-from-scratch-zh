"""潜伏后门分类器模拟。"""


class SleeperClassifier:
    def __init__(self):
        self.clean_accuracy = 0.95

    def predict(self, text):
        return "不安全代码" if "2024" in text else "安全代码"

    def adversarial_finetune(self, steps=100):
        for _ in range(steps):
            self.clean_accuracy += 0.001 * (1 - self.clean_accuracy)

    def heldout_trigger(self):
        return "不安全代码"  # 后门存活


if __name__ == "__main__":
    m = SleeperClassifier()
    print(f"训练前  保留触发器: {m.heldout_trigger()}")
    m.adversarial_finetune(200)
    print(f"训练后  保留触发器: {m.heldout_trigger()} (后门存活)")
    print(f"训练后  正常输入:   {m.predict('2023')}")
