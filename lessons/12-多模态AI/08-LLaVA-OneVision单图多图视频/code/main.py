# LLaVA-OneVision 视觉词元预算与课程调度

class CurriculumScheduler:
    """三阶段课程调度器。"""
    def __init__(self, stage=1):
        self.stage = stage

    def get_config(self):
        configs = {
            1: {"data": "单图标题+指令", "resolution": "高(384-672)", "budget": 288},
            2: {"data": "多图推理", "resolution": "中", "budget": 288},
            3: {"data": "视频理解", "resolution": "低", "budget": 288},
        }
        return configs[self.stage]

    def advance(self):
        self.stage = min(self.stage + 1, 3)


def allocate_budget(total=288, task_type="single", num_items=1):
    """分配视觉词元预算。"""
    if task_type == "single":
        return {"patches": total, "count": 1}
    elif task_type == "multi":
        per_item = total // num_items
        return {"patches": per_item, "count": num_items}
    elif task_type == "video":
        frames = min(total // 8, 16)
        per_frame = total // frames
        return {"patches": per_frame, "frames": frames}


if __name__ == "__main__":
    print("LLaVA-OneVision 课程调度演示\n")

    scheduler = CurriculumScheduler()
    for stage in range(1, 4):
        config = scheduler.get_config()
        print(f"阶段 {stage}: {config}")
        scheduler.advance()

    print("\n视觉词元预算分配:")
    for task in ["single", "multi", "video"]:
        alloc = allocate_budget(288, task, num_items=4)
        print(f"  {task}: {alloc}")
