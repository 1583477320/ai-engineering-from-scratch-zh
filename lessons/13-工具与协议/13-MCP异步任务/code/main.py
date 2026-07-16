# MCP 异步任务管理器

import threading
import uuid


class AsyncTaskManager:
    """异步任务管理器。"""
    def __init__(self):
        self.tasks = {}

    def create_task(self, name, executor, args):
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "pending", "progress": 0, "result": None, "name": name}

        def run():
            try:
                self.tasks[task_id]["status"] = "running"
                result = executor(**args)
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["result"] = result
            except Exception as e:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = str(e)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return task_id

    def get_status(self, task_id):
        return self.tasks.get(task_id, {"status": "not_found"})

    def list_tasks(self):
        return {tid: info["status"] for tid, info in self.tasks.items()}


if __name__ == "__main__":
    print("MCP 异步任务演示\n")

    import time

    manager = AsyncTaskManager()

    def slow_task(duration=2, result="完成"):
        time.sleep(duration)
        return result

    # 创建异步任务
    task_id = manager.create_task("慢任务", slow_task, {"duration": 0.1, "result": "CI 测试通过"})
    print(f"任务 {task_id} 已创建")
    print(f"状态: {manager.get_status(task_id)['status']}")

    # 等待完成
    time.sleep(0.5)
    print(f"状态: {manager.get_status(task_id)['status']}")
    print(f"结果: {manager.get_status(task_id).get('result', '未知')}")

    print(f"\n所有任务: {manager.list_tasks()}")
