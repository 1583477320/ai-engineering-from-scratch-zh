# InternVL3 原生多模态预训练策略

import random


class CurriculumScheduler:
    """InternVL3 分阶段训练调度器。"""
    def __init__(self):
        self.stages = {
            1: {"data": "纯文本", "epochs": 10, "learning_rate": 3e-4},
            2: {"data": "视觉-文本对齐", "epochs": 5, "learning_rate": 1e-4},
            3: {"data": "多模态混合", "epochs": 20, "learning_rate": 5e-5},
            4: {"data": "视觉指令微调", "epochs": 3, "learning_rate": 2e-5},
        }
        self.current_stage = 1

    def get_config(self):
        return self.stages[self.current_stage]

    def advance(self):
        self.current_stage = min(self.current_stage + 1, 4)

    def get_data_mix(self):
        mixes = {
            1: {"text": 0.9, "image": 0.05, "video": 0.05},
            2: {"text": 0.5, "image": 0.4, "video": 0.1},
            3: {"text": 0.4, "image": 0.35, "video": 0.25},
            4: {"text": 0.3, "image": 0.5, "video": 0.2},
        }
        return mixes[self.current_stage]


if __name__ == "__main__":
    print("InternVL3 原生多模态预训练策略\n")
    scheduler = CurriculumScheduler()
    for stage in range(1, 5):
        config = scheduler.get_config()
        mix = scheduler.get_data_mix()
        print(f"阶段 {stage}: {config['data']}, lr={config['learning_rate']}")
        print(f"  数据混合: {mix}")
        scheduler.advance()
