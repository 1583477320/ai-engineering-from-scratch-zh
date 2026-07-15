# 函数调用与工具使用

import json


# ============================================================================
# 第 1 步：工具定义
# ============================================================================

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学运算",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "数学表达式"},
            },
            "required": ["expr"],
        },
    },
}

TOOLS = [GET_WEATHER_TOOL, CALCULATOR_TOOL]


# ============================================================================
# 第 2 步：工具执行
# ============================================================================

def execute_tool(tool_name, tool_args):
    """执行工具并返回结果。"""
    if tool_name == "get_weather":
        city = tool_args["city"]
        return json.dumps({"city": city, "temperature": 22, "condition": "晴", "humidity": 45})
    elif tool_name == "calculate":
        expr = tool_args["expr"]
        try:
            return json.dumps({"result": eval(expr)})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"未知工具: {tool_name}"})


# ============================================================================
# 第 3 步：模拟 LLM 调用
# ============================================================================

def mock_llm_call(messages, tools):
    """模拟 LLM——识别意图并决定是否调用函数。"""
    last_msg = messages[-1]["content"].lower()
    if "天气" in last_msg:
        return {"type": "function_call", "function": "get_weather", "arguments": {"city": "北京", "unit": "celsius"}}
    elif "计算" in last_msg or "+" in last_msg or "*" in last_msg:
        expr = last_msg.split("计算")[-1].strip() if "计算" in last_msg else last_msg
        return {"type": "function_call", "function": "calculate", "arguments": {"expr": "3*7"}}
    return {"type": "text", "content": f"我不知道如何回答: {last_msg}"}


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("函数调用与工具使用演示\n")

    queries = [
        "北京今天天气怎么样？",
        "计算 123 * 456",
    ]

    for query in queries:
        print(f"用户: {query}")
        messages = [{"role": "user", "content": query}]

        # LLM 响应
        response = mock_llm_call(messages, TOOLS)

        if response["type"] == "function_call":
            print(f"  → 调用函数: {response['function']}")
            print(f"  → 参数: {response['arguments']}")

            # 执行工具
            result = execute_tool(response["function"], response["arguments"])
            print(f"  → 结果: {result}")

            # LLM 基于结果回答
            messages.extend([
                {"role": "assistant", "content": f"调用函数 {response['function']}"},
                {"role": "tool", "content": result},
            ])
            final = f"基于结果: {result[:50]}..."
            print(f"  → 最终回答: {final}\n")
        else:
            print(f"  → 直接回答: {response['content']}\n")
