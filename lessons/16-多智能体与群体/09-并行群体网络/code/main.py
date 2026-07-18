"""群体架构演示：工作者从共享队列拉取任务。

对比三种调度策略在可变时长工作负载上的表现：
  - 顺序（1 个工作者处理所有任务）
  - 固定分配（每个任务预先分配给特定工作者）
  - 群体（4 个工作者从共享队列拉取）

群体自动平衡负载；固定分配在工作者分配的任务慢时让工作者空闲。

运行：python3 code/main.py
"""

import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class Task:
    task_id: int
    duration: float
    pre_assigned: int  # 固定分配基线用


def fake_work(task: Task) -> str:
    time.sleep(task.duration)
    return f"task-{task.task_id}-done"


def run_sequential(tasks: list[Task]) -> tuple[float, dict[int, int]]:
    t0 = time.time()
    counts = {0: 0}
    for t in tasks:
        fake_work(t)
        counts[0] += 1
    return time.time() - t0, counts


def run_fixed_assignment(tasks: list[Task], n_workers: int) -> tuple[float, dict[int, int]]:
    per_worker = {i: [] for i in range(n_workers)}
    for t in tasks:
        per_worker[t.pre_assigned].append(t)
    counts = {i: 0 for i in range(n_workers)}

    def worker(wid):
        for t in per_worker[wid]:
            fake_work(t)
            counts[wid] += 1

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.time() - t0, counts


def run_swarm(tasks: list[Task], n_workers: int) -> tuple[float, dict[int, int]]:
    q = queue.Queue()
    for t in tasks:
        q.put(t)
    counts = {i: 0 for i in range(n_workers)}
    lock = threading.Lock()

    def worker(wid):
        while True:
            try:
                task = q.get_nowait()
            except queue.Empty:
                return
            fake_work(task)
            with lock:
                counts[wid] += 1
            q.task_done()

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.time() - t0, counts


def make_tasks(n_workers=4):
    tasks = []
    for i in range(8):
        is_slow = i < 4
        tasks.append(Task(
            task_id=i,
            duration=0.4 if is_slow else 0.1,
            pre_assigned=0 if is_slow else (i - 3) % n_workers,
        ))
    return tasks


def main():
    print("群体架构演示——可变时长工作负载")
    print("-" * 56)
    n_workers = 4
    tasks = make_tasks(n_workers)
    total_work = sum(t.duration for t in tasks)
    print(f"{len(tasks)} 个任务，4 慢（0.4s）+ 4 快（0.1s）")
    print(f"总工作秒数: {total_work:.2f}s")
    print(f"{n_workers} 个工作者的理想并行时间: {total_work / n_workers:.2f}s")

    seq_time, seq_counts = run_sequential(tasks)
    print(f"\n顺序（1 个工作者）:       墙钟={seq_time:.2f}s, 分配={seq_counts}")

    fixed_time, fixed_counts = run_fixed_assignment(tasks, n_workers)
    print(f"固定分配（{n_workers} 个工作者）:  墙钟={fixed_time:.2f}s, 分配={fixed_counts}")
    print("  工作者 0 得到所有 4 个慢任务；其他工作者在快任务后空闲。")

    swarm_time, swarm_counts = run_swarm(tasks, n_workers)
    print(f"群体（{n_workers} 个工作者）:        墙钟={swarm_time:.2f}s, 分配={swarm_counts}")
    print("  负载自动平衡——慢工作者先完成，快的拉取下一个任务。")

    speedup_vs_seq = seq_time / swarm_time if swarm_time > 0 else float("inf")
    speedup_vs_fixed = fixed_time / swarm_time if swarm_time > 0 else float("inf")
    print(f"\n群体加速 vs 顺序: {speedup_vs_seq:.2f}x")
    print(f"群体加速 vs 固定:  {speedup_vs_fixed:.2f}x")
    print("\n要点: 群体在时长可变且分配难以预测时获胜。")
    print("权衡: 无中央追踪；调试需要每任务 ID 和持久化日志。")


if __name__ == "__main__":
    main()
