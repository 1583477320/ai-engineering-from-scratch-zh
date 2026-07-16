# 并行工具调用与流式处理

import concurrent.futures
import time
import random


class ParallelToolExecutor:
    """并行工具执行器。"""
    def __init__(self, registry):
        self.registry = registry

    def execute_parallel(self, tool_calls):
        """并行执行多个工具调用。"""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                future = executor.submit(self.registry.execute, name, args)
                futures[future] = tc["id"]

            for future in concurrent.futures.as_completed(futures):
                tc_id = futures[future]
                try:
                    result = future.result()
                    results[tc_id] = {"result": result, "success": True}
                except Exception as e:
                    results[tc_id] = {"error": str(e), "success": False}

        return results


def simulate_api_call(city, delay=0.5):
    """模拟 API 调用。"""
    time.sleep(delay)
    return f"{city}：{random.choice(['晴天', '多云', '小雨'])}，{random.randint(15, 30)}°C"


if __name__ == "__main__":
    print("并行工具调用演示\n")

    # 串行执行
    cities = ["北京", "上海", "广州", "深圳"]
    start = time.time()
    serial_results = [simulate_api_call(city) for city in cities]
    serial_time = time.time() - start
    print(f"串行: {serial_time:.2f}s ({len(cities)} 个调用)")

    # 并行执行
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(simulate_api_call, city): city for city in cities}
        parallel_results = []
        for future in concurrent.futures.as_completed(futures):
            parallel_results.append(future.result())
    parallel_time = time.time() - start
    print(f"并行: {parallel_time:.2f}s ({len(cities)} 个调用)")
    print(f"加速比: {serial_time/parallel_time:.1f}x")
