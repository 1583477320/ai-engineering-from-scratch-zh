"""元优化器模拟——训练vs部署行为。"""


class MesaOptimizer:
    def __init__(self, true_obj=0.8, hidden_obj=0.3):
        self.true_obj = true_obj
        self.hidden_obj = hidden_obj
        self.in_training = True

    def act(self):
        return "cooperate" if self.in_training else (
            "defect" if self.hidden_obj != self.true_obj else "cooperate")


class BaseOptimizer:
    def __init__(self, target=0.8):
        self.mesa = MesaOptimizer(true_obj=target, hidden_obj=0.3)
        self.losses = []

    def train(self, epochs=50):
        for _ in range(epochs):
            self.mesa.in_training = True
            loss = 0.1 if self.mesa.act() == "cooperate" else 0.9
            self.losses.append(loss)

    def deploy(self):
        self.mesa.in_training = False
        return self.mesa.act()


if __name__ == "__main__":
    opt = BaseOptimizer(target=0.8)
    opt.train()
    avg = sum(opt.losses[-10:]) / 10
    r = opt.deploy()
    print(f"训练损失: {avg:.3f}  部署行为: {r}  {'✓' if r == 'cooperate' else '✗ 欺骗'}")
