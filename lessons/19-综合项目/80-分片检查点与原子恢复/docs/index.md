# 综合项目80——分片检查点与原子恢复（Sharded Checkpoint and Atomic Resume）

> 70B 参数训练任务每隔几小时就被节点故障暂停。检查点格式决定你损失 30 分钟还是 30 小时。

**类型：** 构建
**语言：** Python
**前置知识：** 第19章第78节
**预计时间：** 90分钟

---

## 学习目标

- 将多设备检查点保存为每设备分片文件 + 记录所有权信息的清单
- 使用原子写入模式防止写入中途崩溃留下半成品
- 从清单恢复，验证 fp16 参数和 ZeRO 优化器状态的字节等价性

---

## 1. 问题

朴素检查点将所有参数和优化器状态读入 rank 0，写入单一文件。70B 模型需要 1.1TB 状态通过一个设备的网络端口。gather-然后写入步骤可能比前一小时的训练还长。

分片检查点翻转了模式：每设备并行写入自己的分片。清单记录哪个设备拥有哪个分片，恢复时放回原位。聚合写入带宽随集群扩展。

---

## 2. 核心概念

### 2.1 清单 schema

```json
{
  "world_size": 4,
  "shards": [
    {"rank": 0, "path": "rank0.bin", "sha256": "...", "offset": 0, "numel": 65536},
    {"rank": 1, "path": "rank1.bin", "sha256": "...", "offset": 65536, "numel": 65536}
  ],
  "schema_version": 1
}
```

三个承载字段：`world_size` 防止不同设备数恢复静默损坏；`sha256` 捕获部分或损坏写入。

### 2.2 原子写入

写入 `<name>.tmp`，fsync，然后 POSIX 重命名。在同一文件系统内的重命名是原子的——崩溃前重命名保留前一个检查点。

### 2.3 三种故障模式

| 故障 | 症状 | 防御 |
|:----|:-----|:-----|
| 设备数变化 | N=8 恢复 N=4 的检查点 | world_size 不匹配 |
| 分片数不匹配 | 文件少于清单中的分片 | 列举所有分片 |
| 部分写入 | 分片截断 | sha256 验证 |

---

## 3. 从零实现

```python
"""分片检查点与原子恢复——清单+原子写入+验证。"""
import os, json, hashlib, tempfile
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ShardInfo:
    rank: int; path: str; sha256: str; offset: int = 0; numel: int = 0


@dataclass
class Manifest:
    world_size: int; step: int; shards: List[ShardInfo]; schema_version: int = 1

    def to_dict(self):
        return {"world_size": self.world_size, "step": self.step,
                "shards": [{"rank": s.rank, "path": s.path, "sha256": s.sha256,
                            "offset": s.offset, "numel": s.numel} for s in self.shards],
                "schema_version": self.schema_version}

    @classmethod
    def from_dict(cls, d):
        return cls(d["world_size"], d["step"],
                   [ShardInfo(s["rank"], s["path"], s["sha256"], s["offset"], s["numel"]) for s in d["shards"]])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(data, path):
    dir = os.path.dirname(path) or "."
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=dir, prefix=os.path.basename(path) + ".")
    with tmp:
        if isinstance(data, bytes):
            tmp.write(data)
        else:
            tmp.write(data.encode())
    os.replace(tmp.name, path)


def save_checkpoint(state_dicts, out_dir, step):
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for rank, sd in state_dicts.items():
        fname = f"rank{rank}.bin"
        path = os.path.join(out_dir, fname)
        # 序列化状态字典为 bytes
        import pickle
        data = pickle.dumps(sd)
        atomic_write(data, path)
        shards.append(ShardInfo(rank, fname, sha256_file(path),
                               offset=0, numel=len(data)))
    manifest = Manifest(len(state_dicts), step, shards)
    atomic_write(json.dumps(manifest.to_dict(), indent=2), os.path.join(out_dir, "manifest.json"))
    return manifest


def load_checkpoint(in_dir, expected_world_size):
    manifest = Manifest.from_dict(json.load(open(os.path.join(in_dir, "manifest.json"))))
    if manifest.world_size != expected_world_size:
        raise ValueError(f"设备数不匹配: {manifest.world_size} != {expected_world_size}")
    state_dicts = {}
    for s in manifest.shards:
        path = os.path.join(in_dir, s.path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少分片: {path}")
        if sha256_file(path) != s.sha256:
            raise ValueError(f"sha256 不匹配: {s.path}")
        import pickle
        with open(path, "rb") as f:
            state_dicts[s.rank] = pickle.load(f)
    return state_dicts, manifest


def main():
    state_dicts = {i: {"param": b"fake_data_" + str(i).encode() * 100} for i in range(4)}
    out_dir = "/tmp/sharded_ckpt"
    manifest = save_checkpoint(state_dicts, out_dir, step=10)
    print(f"保存 {len(manifest.shards)} 个分片到 {out_dir}")

    loaded, loaded_manifest = load_checkpoint(out_dir, 4)
    print(f"恢复 {len(loaded)} 个设备的状态")
    for rank, sd in loaded.items():
        ok = sd == state_dicts[rank]
        print(f"  rank {rank}: {'✓' if ok else '✗'}")
    return 0

if __name__ == "__main__":
    import sys; sys.exit(main())
```

---

## 4. 工业工具

| 工具 | 分片 | 清单 |
|:----|:-----|:-----|
| DeepSpeed | per-rank | `latest` + tag |
| PyTorch dist.checkpoint | Planner | API |
| NeMo | per-rank | JSON |

---

## 5. 工程最佳实践

- 异步写入：检查点写入在单独线程执行，训练继续
- 本地快速磁盘 → 异步上传到 S3：两层级保持集群内写入快速
- 循环使用最近 K 个检查点（默认 3-5 个），磁盘满前删除最旧的
- **中文场景建议**：检查点路径使用英文

---

## 6. 常见错误

- **设备数变化时恢复损坏**：清单中的 world_size 必须匹配
- **未原子写入**：崩溃留下半截文件，sha256 验证发现
- **同步写入阻塞训练**：检查点写几分钟期间训练停滞

---

## 7. 面试考点

**Q1：分片检查点为什么比单文件检查点快？**（难度：⭐⭐）

**参考答案：** 每设备并行写入自己的分片，聚合带宽随设备数扩展。单文件检查点将所有状态收集到一个设备再写入——写入速度受限于单个设备的网络链接和 I/O。

---

## 🔑 关键术语

| 术语 | 含义 |
|:----|:-----|
| 分片检查点 | 每设备并行写入自己的分片 |
| 清单 | JSON 记录分片路径、偏移、sha256 |
| 原子写入 | tmp + 重命名——崩溃保留上一次有效检查点 |

---

## 📚 小结

分片检查点使大模型训练中断恢复成为可能。你实现了清单、原子写入和 sha256 验证。下一节将所有组件组合为端到端分布式训练演示。

---

## ✏️ 练习

1. 【实现】添加异步写入：在单独线程启动保存，训练继续
2. 【实现】添加循环保持最后 5 个检查点，保存新检查点前删除最旧的

---

## 🚀 产出

| 产出 | 文件 |
|:----|:-----|
| 分片检查点 | `code/main.py` |

---

## 📖 参考资料

1. [官方文档] DeepSpeed 检查点. https://www.deepspeed.ai/tutorials/checkpointing/
2. [官方文档] PyTorch 分布式检查点. https://pytorch.org/docs/stable/distributed.checkpoint.html
