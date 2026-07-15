# 提示词工程：结构化提示词 + 测试框架

import json
import re


# ============================================================================
# 构建结构化提示词
# ============================================================================

def build_system_prompt(role, context, constraints, output_format, examples=None):
    """构建结构化的系统提示词。"""
    parts = [
        f"## 角色\n{role}",
        f"## 上下文\n{context}",
        f"## 约束\n" + "\n".join(f"- {c}" for c in constraints),
        f"## 输出格式\n{output_format}",
    ]
    if examples:
        parts.append("## 示例\n" + "\n".join(f"### 示例 {i+1}\n{e}" for i, e in enumerate(examples)))
    return "\n\n".join(parts)


# ============================================================================
# 提示词测试框架
# ============================================================================

def evaluate_prompt(prompt_fn, test_cases):
    """评估提示词效果。"""
    passed = 0
    results = []
    for input_text, expected_pattern in test_cases:
        try:
            response = prompt_fn(input_text)
            if expected_pattern in response:
                passed += 1
                results.append((input_text, True))
            else:
                results.append((input_text, False, f"未包含: {expected_pattern}"))
        except Exception as e:
            results.append((input_text, False, str(e)))
    return passed / max(len(test_cases), 1), results


# ============================================================================
# 参数配置
# ============================================================================

def configure_params(task_type="general"):
    """根据任务类型推荐生成参数。"""
    params = {
        "coding": {"temperature": 0.0, "top_p": 0.95, "max_tokens": 2048, "frequency_penalty": 0.0},
        "creative": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1024, "frequency_penalty": 0.3},
        "classification": {"temperature": 0.0, "top_p": 0.5, "max_tokens": 128, "frequency_penalty": 0.0},
        "extraction": {"temperature": 0.0, "top_p": 0.9, "max_tokens": 512, "frequency_penalty": 0.1},
        "general": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 1024, "frequency_penalty": 0.0},
    }
    return params.get(task_type, params["general"])


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("提示词工程演示\n")

    # 1. 结构化系统提示词
    prompt = build_system_prompt(
        role="你是一位资深 Python 工程师。",
        context="用户提交了 Python 代码需要审查。",
        constraints=[
            "只指出安全问题，不要风格建议。",
            "每个问题提供修复建议。",
            "用中文回答。",
        ],
        output_format="格式: 每个问题用 '### 问题 N' 开头",
    )
    print(f"系统提示词长度: {len(prompt)} 字")
    print(f"前 100 字: {prompt[:100]}...\n")

    # 2. 温度参数演示
    print("任务参数推荐:")
    for task in ["coding", "creative", "classification"]:
        params = configure_params(task)
        print(f"  {task}: temp={params['temperature']}")

    # 3. 提示词测试
    def mock_prompt_fn(text):
        return "这是一个模拟的回复。模型将会返回一些内容。"
    test_cases = [("你好", "模拟"), ("测试", "返回")]
    score, results = evaluate_prompt(mock_prompt_fn, test_cases)
    print(f"\n测试通过率: {score:.0%}")
