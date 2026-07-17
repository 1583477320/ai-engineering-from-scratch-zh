"""单智能体 vs 多智能体对比——纯标准库。

模拟单智能体（所有角色在一个上下文中）vs 多智能体流水线（每个智能体一个上下文）vs 扇出并行。
展示上下文隔离、专业化提示词、并行化的收益。

运行：python3 code/main.py
"""

import asyncio
import time
import random
from dataclasses import dataclass


@dataclass
class LLMResponse:
    output: str
    tokens: int
    calls: int


@dataclass
class AgentResult:
    content: str
    tokens_used: int
    tool_calls: int


@dataclass
class SpecialistAgent:
    name: str
    system_prompt: str


# ── 模拟 LLM 调用 ──────────────────────────────────────────

def fake_llm_call(system_prompt: str, user_message: str) -> LLMResponse:
    """模拟 LLM 调用——基于输入长度估算词元。"""
    input_length = len(system_prompt) + len(user_message)
    simulated_tokens = input_length // 4 + 500
    time.sleep(0.01)
    return LLMResponse(
        output=f"[{system_prompt[:30]}... 对 {user_message[:60]}... 的响应]",
        tokens=simulated_tokens,
        calls=random.randint(1, 5),
    )


# ── 单智能体方法 ──────────────────────────────────────────

def single_agent_approach(task: str) -> AgentResult:
    """单智能体：一个巨大的系统提示词 + 一个上下文窗口。"""
    system_prompt = """你是全栈开发者。你必须：
1. 研究需求
2. 写代码
3. 审查代码
4. 写测试
在一个对话中完成所有这些。"""

    context_window: list[str] = []
    total_tokens = 0
    total_calls = 0

    # 研究
    research = fake_llm_call(system_prompt, f"研究: {task}")
    context_window.append(research.output)
    total_tokens += research.tokens
    total_calls += research.calls

    # 编码（带所有之前的上下文）
    code = fake_llm_call(
        system_prompt,
        f"根据这些研究:\n{chr(10).join(context_window)}\n\n为 {task} 写代码"
    )
    context_window.append(code.output)
    total_tokens += code.tokens
    total_calls += code.calls

    # 审查（带上所有之前的上下文）
    review = fake_llm_call(
        system_prompt,
        f"根据所有之前的上下文:\n{chr(10).join(context_window)}\n\n审查代码。"
    )
    context_window.append(review.output)
    total_tokens += review.tokens
    total_calls += review.calls

    return AgentResult(
        content="\n---\n".join(context_window),
        tokens_used=total_tokens,
        tool_calls=total_calls,
    )


# ── 多智能体流水线 ──────────────────────────────────────

def create_specialist(name: str, system_prompt: str) -> SpecialistAgent:
    return SpecialistAgent(name=name, system_prompt=system_prompt)


def specialist_run(agent: SpecialistAgent, input_text: str) -> AgentResult:
    result = fake_llm_call(agent.system_prompt, input_text)
    return AgentResult(
        content=result.output,
        tokens_used=result.tokens,
        tool_calls=result.calls,
    )


researcher = create_specialist("researcher", "你是技术研究员。读文档、找模式、总结发现。只输出实现所需的事实。")
coder = create_specialist("coder", "你是高级 Python 开发者。根据需求和研究笔记写干净、有测试的代码。不要做其他事。")
reviewer = create_specialist("reviewer", "你是代码审查员。找 bug、安全问题和逻辑错误。具体指出行号。")


def multi_agent_pipeline(task: str) -> AgentResult:
    """多智能体流水线：每个智能体一个上下文窗口。"""
    total_tokens = 0
    total_calls = 0

    research_result = specialist_run(researcher, task)
    total_tokens += research_result.tokens_used
    total_calls += research_result.tool_calls

    code_result = specialist_run(coder, f"[来自研究者]: {research_result.content}")
    total_tokens += code_result.tokens_used
    total_calls += code_result.tool_calls

    review_result = specialist_run(reviewer, f"[来自编码者]: {code_result.content}")
    total_tokens += review_result.tokens_used
    total_calls += review_result.tool_calls

    return AgentResult(
        content=review_result.content,
        tokens_used=total_tokens,
        tool_calls=total_calls,
    )


# ── 扇出并行 ──────────────────────────────────────────────

requirements = create_specialist("requirements", "你是需求分析师。提取功能和非功能需求。要详尽。")


async def multi_agent_fan_out(task: str) -> AgentResult:
    """扇出：研究者和需求分析并行。"""
    loop = asyncio.get_event_loop()

    research_future = loop.run_in_executor(None, specialist_run, researcher, f"研究 {task} 的技术方案")
    requirements_future = loop.run_in_executor(None, specialist_run, requirements, f"分析 {task} 的需求")

    research_result, requirements_result = await asyncio.gather(
        research_future, requirements_future
    )

    merged_input = f"[来自研究者]: {research_result.content}\n[来自需求分析者]: {requirements_result.content}"
    code_result = specialist_run(coder, merged_input)

    total_tokens = research_result.tokens_used + requirements_result.tokens_used + code_result.tokens_used
    total_calls = research_result.tool_calls + requirements_result.tool_calls + code_result.tool_calls

    return AgentResult(
        content=code_result.content,
        tokens_used=total_tokens,
        tool_calls=total_calls,
    )


# ── 主函数 ────────────────────────────────────────────────

def main():
    task = "构建 Express.js API 的速率限制中间件"

    print("=" * 70)
    print("单智能体 vs 多智能体（阶段 16，第 1 课）")
    print("=" * 70)

    print("\n=== 单智能体方法 ===\n")
    single = single_agent_approach(task)
    print(f"  词元: {single.tokens_used}")
    print(f"  工具调用: {single.tool_calls}")
    print(f"  上下文: 所有内容在一个窗口中\n")

    print("=== 多智能体流水线 ===\n")
    multi = multi_agent_pipeline(task)
    print(f"  词元: {multi.tokens_used}")
    print(f"  工具调用: {multi.tool_calls}")
    print(f"  上下文: 每个智能体只接收它需要的内容\n")

    print("=== 扇出并行 ===\n")
    fan = asyncio.run(multi_agent_fan_out(task))
    print(f"  词元: {fan.tokens_used}")
    print(f"  工具调用: {fan.tool_calls}")
    print(f"  上下文: 研究者 + 需求分析者并行运行\n")

    print("=" * 70)
    print("要点: 脚手架不是装饰，它是产品")
    print("-" * 70)
    print(f"  单智能体上下文污染: 所有 {single.tokens_used} 词元在一个窗口")
    print(f"  多智能体隔离: {multi.tokens_used} 词元分布在 3 个隔离窗口")
    print(f"  扇出并行: 研究 + 需求分析同时运行")


if __name__ == "__main__":
    main()
