"""四协议演示：MCP + A2A + ACP + ANP 集成——纯标准库。

构建 A2A Agent Card、任务管理器、审计追踪、DID 身份验证和协议网关。
展示四个协议如何协同工作。

运行：python3 code/main.py
"""

import asyncio
import hashlib
from dataclasses import dataclass, field


# ── 核心消息类型 ──────────────────────────────────────────

@dataclass
class AgentMessage:
    role: str        # "user" | "agent"
    parts: list[dict]
    trajectory: list[dict] = field(default_factory=list)
    reply_to: str = ""


def text_message(role: str, text: str) -> AgentMessage:
    return AgentMessage(role=role, parts=[{"kind": "text", "text": text}])


# ── A2A Agent Card 和注册表 ────────────────────────────────

@dataclass
class AgentCard:
    name: str
    description: str
    version: str = "1.0.0"
    url: str = ""
    skills: list[dict] = field(default_factory=list)
    capabilities: dict = field(default_factory=lambda: {"streaming": True, "pushNotifications": False})


class AgentRegistry:
    def __init__(self):
        self.cards = {}

    def register(self, card: AgentCard):
        self.cards[card.name] = card

    def discover_by_skill_tag(self, tag: str) -> list[AgentCard]:
        return [c for c in self.cards.values()
                if any(tag in s.get("tags", []) for s in c.skills)]

    def discover_by_input_mode(self, mime: str) -> list[AgentCard]:
        return [c for c in self.cards.values()
                if mime in c.skills[0].get("inputModes", [])]

    def resolve(self, name: str):
        return self.cards.get(name)

    def list_all(self) -> list[AgentCard]:
        return list(self.cards.values())


# ── A2A 任务管理器 ────────────────────────────────────────

TERMINAL_STATES = {"completed", "failed", "canceled", "rejected"}


@dataclass
class Task:
    id: str
    context_id: str = ""
    status: str = "submitted"
    artifacts: list = field(default_factory=list)
    history: list = field(default_factory=list)


class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.handlers = {}

    def register_handler(self, name, handler):
        self.handlers[name] = handler

    async def send_message(self, agent_name, message, context_id=None):
        handler = self.handlers.get(agent_name)
        if not handler:
            task = Task(id="error", context_id=context_id or "", status="rejected")
            return task

        task = Task(id=f"task-{len(self.tasks)+1}", context_id=context_id or f"ctx-{len(self.tasks)+1}")
        task.history.append({"role": message.role, "content": message.parts[0].get("text", "")})
        task.status = "working"

        try:
            result = await handler(task, message)
            task.status = "completed"
            task.artifacts = result
        except Exception as e:
            task.status = "failed"
        return task

    def cancel_task(self, task_id):
        task = self.tasks.get(task_id)
        if not task or task.status in TERMINAL_STATES:
            return False
        task.status = "canceled"
        return True


# ── ACP 审计追踪 ──────────────────────────────────────────

@dataclass
class AuditEntry:
    run_id: str
    agent_name: str
    input_msgs: list
    output_msgs: list
    trajectory: list[dict]
    status: str = "created"
    started_at: float = 0
    completed_at: float = 0
    session_id: str = ""


class AuditableRunner:
    def __init__(self):
        self.log = []
        self.handlers = {}

    def register_agent(self, name, handler):
        self.handlers[name] = handler

    async def run(self, agent_name, input_msgs, session_id=None):
        import time
        entry = AuditEntry(
            run_id=f"run-{len(self.log)+1}",
            agent_name=agent_name,
            input_msgs=list(input_msgs),
            output_msgs=[],
            trajectory=[],
            status="created",
            started_at=time.time(),
            session_id=session_id or "",
        )
        self.log.append(entry)

        handler = self.handlers.get(agent_name)
        if not handler:
            entry.status = "failed"
            return entry

        entry.status = "in-progress"
        try:
            result = await handler(input_msgs)
            entry.output_msgs = result.get("output", [])
            entry.trajectory = result.get("trajectory", [])
            entry.status = "completed"
            entry.completed_at = time.time()
        except Exception as e:
            entry.status = "failed"
            entry.trajectory.append({"reasoning": f"Error: {e}", "timestamp": time.time()})
        return entry

    def get_full_audit_log(self):
        return list(self.log)

    def get_audit_for_agent(self, agent_name):
        return [e for e in self.log if e.agent_name == agent_name]

    def get_audit_for_session(self, session_id):
        return [e for e in self.log if e.session_id == session_id]


# ── ANP 身份验证 ──────────────────────────────────────────

@dataclass
class DIDDocument:
    id: str
    public_key: str
    authentication: list[str] = field(default_factory=list)


class IdentityRegistry:
    def __init__(self):
        self.documents = {}

    def publish(self, doc):
        self.documents[doc.id] = doc

    def resolve(self, did):
        return self.documents.get(did)

    def verify(self, did, signature, payload):
        doc = self.documents.get(did)
        if not doc:
            return False
        expected = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return signature == expected

    def requires_human_auth(self, did):
        doc = self.documents.get(did)
        if not doc:
            return False
        return "human" in doc.authentication


def create_identity(domain, agent_name):
    did = f"did:wba:{domain}:agent:{agent_name}"
    key = hashlib.sha256(f"{domain}:{agent_name}".encode()).hexdigest()[:16]
    doc = DIDDocument(id=did, public_key=key, authentication=["standard"])
    return {"did": did, "document": doc, "key": key}


def sign_payload(key, payload):
    return hashlib.sha256(f"{key}:{payload}".encode()).hexdigest()[:16]


# ── 协议网关 ──────────────────────────────────────────────

class ProtocolGateway:
    def __init__(self, registry, task_manager, audit_runner, identity_registry):
        self.registry = registry
        self.task_manager = task_manager
        self.audit_runner = audit_runner
        self.identity_registry = identity_registry

    async def delegate_task(self, from_did, signature, target_agent, message, session_id=None):
        # ANP: 验证身份
        if not self.identity_registry.verify(from_did, signature, message.parts[0].get("text", "")):
            return {"error": "身份验证失败"}

        # A2A: 发现目标智能体
        card = self.registry.resolve(target_agent)
        if not card:
            return {"error": f"智能体 {target_agent} 未找到"}

        # ACP: 包裹审计
        audit = await self.audit_runner.run(target_agent, [message], session_id)

        # A2A: 创建任务
        task = await self.task_manager.send_message(target_agent, message)

        return {"task": task, "audit": audit}


# ── 主函数 ────────────────────────────────────────────────

async def main():
    print("=" * 70)
    print("四协议集成演示（阶段 16，第 3 课）")
    print("=" * 70)

    # 注册智能体
    registry = AgentRegistry()
    registry.register(AgentCard(name="researcher", description="搜索并总结",
                                 skills=[{"id": "web-research", "tags": ["research"],
                                          "inputModes": ["text/plain"], "outputModes": ["application/json"]}]))
    registry.register(AgentCard(name="coder", description="写代码",
                                 skills=[{"id": "code-gen", "tags": ["coding"],
                                          "inputModes": ["text/plain"], "outputModes": ["text/plain"]}]))

    # 任务管理器
    task_manager = TaskManager()

    # 审计运行器
    audit_runner = AuditableRunner()
    research_trajectory = []

    async def researcher_handler(task, message):
        research_trajectory.append({
            "reasoning": "搜索 React 19 文档",
            "toolName": "web_search",
            "toolInput": {"query": "React 19 compiler features"},
            "toolOutput": {"results": ["react.dev/blog/react-19"]},
            "timestamp": 1,
        })
        research_trajectory.append({
            "reasoning": "从搜索结果提取关键发现",
            "toolName": "doc_analysis",
            "toolInput": {"url": "react.dev/blog/react-19"},
            "toolOutput": {"summary": "React 19 compiler auto-memoizes"},
            "timestamp": 2,
        })
        return {
            "output": [text_message("agent", "React 19 compiler auto-memoizes components")],
            "trajectory": research_trajectory,
        }

    task_manager.register_handler("researcher", researcher_handler)
    audit_runner.register_agent("researcher", researcher_handler)

    # 身份注册
    identity_registry = IdentityRegistry()
    coder_id = create_identity("coder.local", "coder")
    researcher_id = create_identity("researcher.local", "researcher")
    identity_registry.publish(coder_id["document"])
    identity_registry.publish(researcher_id["document"])

    # 协议网关
    gateway = ProtocolGateway(registry, task_manager, audit_runner, identity_registry)

    print("\n1. 智能体发现 (A2A)")
    research_agents = registry.discover_by_skill_tag("research")
    print(f"  发现 {len(research_agents)} 个智能体: {[a.name for a in research_agents]}")

    print("\n2. 身份验证 (ANP)")
    message = text_message("user", "研究 React 19 编译器特性")
    signature = sign_payload(coder_id["key"], message.parts[0]["text"])
    verified = identity_registry.verify(coder_id["did"], signature, message.parts[0]["text"])
    print(f"  编码者 DID: {coder_id['did']}")
    print(f"  签名验证: {verified}")

    print("\n3. 任务委派 (A2A + ACP + ANP)")
    result = await gateway.delegate_task(
        coder_id["did"], signature, "researcher", message, "session-001"
    )

    if "error" in result:
        print(f"  错误: {result['error']}")
        return

    print(f"  任务 ID: {result['task'].id}")
    print(f"  任务状态: {result['task'].status}")
    print(f"  工件数: {len(result['task'].artifacts)}")

    print("\n4. 审计轨迹 (ACP)")
    print(f"  运行 ID: {result['audit'].run_id}")
    print(f"  状态: {result['audit'].status}")
    print(f"  轨迹步骤: {len(result['audit'].trajectory)}")
    for step in result['audit'].trajectory:
        print(f"    - {step.get('reasoning', '')}")
        if step.get('toolName'):
            print(f"      工具: {step['toolName']}")

    print("\n5. 完整审计日志")
    full_log = audit_runner.get_full_audit_log()
    print(f"  总运行数: {len(full_log)}")
    for entry in full_log:
        duration = f"{entry.completed_at - entry.started_at:.0f}ms" if entry.completed_at else "in-progress"
        print(f"  {entry.agent_name}: {entry.status} ({duration})")


if __name__ == "__main__":
    asyncio.run(main())
