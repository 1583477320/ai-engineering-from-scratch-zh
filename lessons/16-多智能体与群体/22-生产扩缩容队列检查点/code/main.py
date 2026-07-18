# main.py — 生产扩缩容实现
# 对应课程：阶段 16 · 22（生产扩缩容——队列、检查点与持久化）

import sqlite3
import json
import time
import asyncio
import threading


# === 检查点存储 ===

class CheckpointStore:
    def __init__(self, db_path=":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT, step INTEGER, state TEXT,
                PRIMARY KEY (thread_id, step)
            )
        """)
        self._conn.commit()

    def save(self, thread_id, step, state):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
                (thread_id, step, json.dumps(state)),
            )
            self._conn.commit()

    def load_latest(self, thread_id):
        cursor = self._conn.execute(
            "SELECT state, step FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
            (thread_id,),
        )
        row = cursor.fetchone()
        return (json.loads(row[0]), row[1]) if row else ({}, 0)


class CheckpointRunner:
    def __init__(self, store):
        self.store = store

    def run(self, thread_id, task):
        state, step = self.store.load_latest(thread_id)
        print(f"[恢复] thread_id={thread_id}, 从步骤 {step} 恢复")
        for i in range(step, len(task)):
            state["current_step"] = task[i]
            state["progress"] = i + 1
            self.store.save(thread_id, i + 1, state)
            print(f"[步骤 {i+1}/{len(task)}] {task[i]} → 检查点已保存")
            if i == 1 and not state.get("recovered"):
                state["recovered"] = True
                raise RuntimeError("模拟崩溃")
        print(f"[完成] 全部 {len(task)} 个步骤")
        return state


# === 异步 vs 线程演示 ===

async def mock_llm_async(delay=0.01):
    await asyncio.sleep(delay)
    return delay


def mock_llm_sync(delay=0.01):
    time.sleep(delay)
    return delay


async def demo_concurrency(count=200):
    print(f"\n=== 异步 vs 线程（{count} 并发）===")
    start = time.time()
    await asyncio.gather(*[mock_llm_async() for _ in range(count)])
    t_async = time.time() - start

    start = time.time()
    threads = [threading.Thread(target=mock_llm_sync) for _ in range(count)]
    for t in threads: t.start()
    for t in threads: t.join()
    t_thread = time.time() - start

    print(f"  异步: {t_async:.3f}s")
    print(f"  线程: {t_thread:.3f}s")
    print(f"  差距: {t_thread / t_async:.1f}x")


# === 主程序 ===

if __name__ == "__main__":
    store = CheckpointStore()
    runner = CheckpointRunner(store)
    task = ["数据采集", "数据清洗", "数据分析", "报告撰写"]

    print("=== 第一次运行 ===")
    try:
        runner.run("thread-001", task)
    except RuntimeError as e:
        print(f"  崩溃: {e}")

    print("\n=== 第二次运行（恢复）===")
    result = runner.run("thread-001", task)
    print(f"  最终状态: {result}")

    asyncio.run(demo_concurrency(200))
