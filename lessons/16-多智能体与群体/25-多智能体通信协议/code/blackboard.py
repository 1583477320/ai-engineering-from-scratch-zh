# blackboard.py — 共享黑板通信系统
# 依赖：标准库（无第三方依赖）
# 对应课程：阶段 16 · 22（多智能体通信协议）

"""
基于共享黑板的多智能体通信系统。
支持消息写入、查询、订阅、TTL 自动清理。
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from enum import Enum


class Performative(Enum):
    """FIPA ACL 言外行为——消息的意图。"""
    REQUEST = "request"
    INFORM = "inform"
    QUERY = "query"
    CONFIRM = "confirm"
    REFUSE = "refuse"
    AGREE = "agree"
    FAILURE = "failure"
    SUBSCRIBE = "subscribe"


@dataclass
class Message:
    """统一的消息格式，基于 FIPA ACL。"""
    sender: str
    receiver: str
    performative: Performative
    content: str = ""
    ontology: str = "default"
    protocol: str = "fipa-request"
    reply_to: Optional[str] = None
    conversation_id: str = ""
    hop_count: int = 0


class Blackboard:
    """共享黑板——智能体之间的信息交换中心。"""

    def __init__(self, default_ttl: float = 120.0):
        self._entries = []
        self._subscriptions = defaultdict(list)
        self.default_ttl = default_ttl

    def write(self, message: Message) -> str:
        """写入消息。"""
        if message.hop_count > 10:
            raise ValueError("超过最大跳数限制")
        entry = {
            "message": message,
            "timestamp": time.time(),
            "expires_at": time.time() + self.default_ttl,
        }
        self._entries.append(entry)

        # 通知订阅者
        for subscriber in self._subscriptions.get(message.ontology, []):
            pass  # 实际系统中触发回调
        return f"msg_{len(self._entries)}"

    def read(self, ontology: str = "", performative: Optional[Performative] = None,
             agent_id: str = "", since: float = 0) -> list[Message]:
        """读取条件匹配的消息。"""
        self._cleanup()
        results = []
        for entry in self._entries:
            msg = entry["message"]
            if ontology and msg.ontology != ontology:
                continue
            if performative and msg.performative != performative:
                continue
            if agent_id and msg.receiver != agent_id and msg.receiver != "ALL":
                continue
            if entry["timestamp"] < since:
                continue
            results.append(msg)
        return results

    def subscribe(self, agent_id: str, ontology: str):
        """订阅特定类型的消息。"""
        if agent_id not in self._subscriptions[ontology]:
            self._subscriptions[ontology].append(agent_id)

    def wait_for_reply(self, reply_to: str, timeout: float = 30.0) -> Optional[Message]:
        """等待特定消息的回复。"""
        start = time.time()
        while time.time() - start < timeout:
            for entry in self._entries:
                msg = entry["message"]
                if msg.reply_to == reply_to:
                    return msg
            time.sleep(0.1)
        return None

    def _cleanup(self):
        """清理过期消息。"""
        now = time.time()
        self._entries = [e for e in self._entries if e["expires_at"] > now]

    def stats(self) -> dict:
        """黑板统计。"""
        self._cleanup()
        return {
            "total_messages": len(self._entries),
            "subscriptions": {k: len(v) for k, v in self._subscriptions.items()},
        }


# === 演示 ===
if __name__ == "__main__":
    bb = Blackboard()

    # 写入消息
    msg1 = Message("分析师", "撰写者", Performative.INFORM,
                    "分析结果：销售增长 20%", "analysis")
    bb.write(msg1)

    msg2 = Message("分析师", "ALL", Performative.INFORM,
                    "数据已更新", "status")
    bb.write(msg2)

    # 读取消息
    analysis_msgs = bb.read(ontology="analysis")
    print(f"分析消息: {len(analysis_msgs)} 条")

    all_msgs = bb.read()
    print(f"所有消息: {len(all_msgs)} 条")

    # 订阅
    bb.subscribe("核验者", "analysis")
    print(f"黑板统计: {bb.stats()}")
