"""冷启动缓解——各阶段时间计算和缓解效果。"""


def cold_start_anatomy(model_size, mitigations=None):
    """计算冷启动各阶段时间。"""
    anatomy = {
        "7B":  {"node": 50, "image": 120, "weight_load": 30, "engine": 15, "forward": 2},
        "13B": {"node": 50, "image": 150, "weight_load": 45, "engine": 18, "forward": 2.5},
        "70B": {"node": 50, "image": 180, "weight_load": 75, "engine": 20, "forward": 3},
    }
    a = dict(anatomy.get(model_size, anatomy["7B"]))

    if mitigations:
        if "preseed" in mitigations:
            a["image"] = 0  # Bottlerocket 消除镜像拉取
        if "streamer" in mitigations:
            a["weight_load"] = int(a["weight_load"] * 0.5)  # 流式加载减半
        if "snapshot" in mitigations:
            a["weight_load"] = 0  # GPU 快照消除权重加载
        if "warm_pool" in mitigations:
            return 0, 0  # 热池消除冷启动

    total = sum(a.values())
    return total, a


if __name__ == "__main__":
    print("=== 冷启动时间分解 ===\n")
    for size in ["7B", "13B", "70B"]:
        total, _ = cold_start_anatomy(size)
        print(f"{size:4s}  总计: {total:3d}s ({total/60:.1f}分钟)")

    print("\n=== 缓解后对比 ===\n")
    for size in ["7B", "70B"]:
        raw, _ = cold_start_anatomy(size)
        mitigated, _ = cold_start_anatomy(size, ["preseed", "streamer"])
        print(f"{size:4s}  原始: {raw:3d}s → 缓解后: {mitigated:3d}s (节省 {raw-mitigated}s)")
