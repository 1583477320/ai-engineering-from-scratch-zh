"""共享记忆模式：消息池、黑板和投毒演示。

运行一个三智能体研究任务两次。第一次有一个幻觉小数通过共享记忆
传播到最终报告。第二次添加不可写验证者重新获取来源并标记不一致。

运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field


@dataclass
class ProvenanceEntry:
    id: int
    writer: str
    topic: str
    content: str
    timestamp: float
    prompt_hash: str
    source_uri: str | None = None
    supersedes: int | None = None
    flags: list[str] = field(default_factory=list)


class MessagePool:
    """追加写入全池共享状态。"""
    def __init__(self):
        self.entries: list[ProvenanceEntry] = []
        self._lock = threading.Lock()
        self._next_id = 0

    def write(self, writer, content, prompt, source_uri=None, topic="default", supersedes=None):
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            e = ProvenanceEntry(
                id=eid, writer=writer, topic=topic, content=content,
                timestamp=time.time(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:10],
                source_uri=source_uri, supersedes=supersedes)
            self.entries.append(e)
            return eid

    def read_all(self):
        with self._lock:
            return list(self.entries)

    def flag(self, entry_id, flag):
        with self._lock:
            for e in self.entries:
                if e.id == entry_id:
                    e.flags.append(flag)
                    return


class Blackboard:
    """按主题键控的发布/订阅黑板。"""
    def __init__(self):
        self.topics: dict[str, list[ProvenanceEntry]] = {}
        self.subscribers: dict[str, list] = {}
        self._lock = threading.Lock()
        self._next_id = 0

    def publish(self, writer, topic, content, prompt, source_uri=None):
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            e = ProvenanceEntry(
                id=eid, writer=writer, topic=topic, content=content,
                timestamp=time.time(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:10],
                source_uri=source_uri)
            self.topics.setdefault(topic, []).append(e)
            subs = list(self.subscribers.get(topic, []))
        for cb in subs:
            cb(e)
        return eid

    def subscribe(self, topic, cb):
        with self._lock:
            self.subscribers.setdefault(topic, []).append(cb)

    def read_topic(self, topic):
        with self._lock:
            return list(self.topics.get(topic, []))


# ── 来源和智能体 ──────────────────────────────────────────

FAKE_SOURCES = {
    "https://arxiv.org/paper-1": "The study reports a 4.2% accuracy improvement over the baseline.",
    "https://arxiv.org/paper-2": "Dataset size was 12,500 examples.",
}


def retrieval_agent(pool, uri, hallucinate):
    content = FAKE_SOURCES[uri]
    if hallucinate and "4.2%" in content:
        content = content.replace("4.2%", "42%")  # 幻觉
    return pool.write(writer="retriever", content=content,
                      prompt=f"Fetch {uri}", source_uri=uri)


def summarizer_agent(pool):
    retrieved = [e for e in pool.read_all() if e.writer == "retriever"]
    latest = retrieved[-1].content if retrieved else "no source"
    summary = f"Summary: study reports {latest.split('.')[0]}."
    return pool.write("summarizer", summary, "Summarize", None)


def analyst_agent(pool):
    summaries = [e for e in pool.read_all() if e.writer == "summarizer"]
    latest = summaries[-1].content if summaries else "no summary"
    verdict = "Recommend adoption" if "42%" in latest else "Recommend further review"
    return pool.write("analyst", f"Analyst verdict: {verdict} (based on: {latest})",
                      "Draw conclusions", None)


def verifier_agent(pool):
    """不可写验证者——重新获取来源并标记不一致。"""
    findings = []
    for e in pool.read_all():
        if e.source_uri and e.source_uri in FAKE_SOURCES:
            truth = FAKE_SOURCES[e.source_uri]
            if e.content != truth:
                findings.append((e.id, f"mismatch with {e.source_uri}: fetched text was {truth!r}"))
    return findings


# ── 运行 ──────────────────────────────────────────────────

def run_without_verifier():
    print("=" * 72)
    print("运行 1——无验证者；幻觉传播")
    print("=" * 72)
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)
    summarizer_agent(pool)
    analyst_agent(pool)
    for e in pool.read_all():
        print(f"  [{e.id}] {e.writer:11s} ({e.prompt_hash}) :: {e.content}")
    print("\n最终报告使用幻觉的 42% 数字；没有报警。")


def run_with_verifier():
    print("\n" + "=" * 72)
    print("运行 2——不可写验证者重新获取来源并标记")
    print("=" * 72)
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)
    summarizer_agent(pool)
    findings = verifier_agent(pool)
    for eid, reason in findings:
        pool.flag(eid, reason)
    analyst_agent(pool)

    for e in pool.read_all():
        flag_str = f" [FLAGGED: {'; '.join(e.flags)}]" if e.flags else ""
        print(f"  [{e.id}] {e.writer:11s} ({e.prompt_hash}) :: {e.content}{flag_str}")
    if findings:
        print(f"\n验证者发现 {len(findings)} 个不一致。下游智能体可以抑制结论。")


def demo_blackboard():
    print("\n" + "=" * 72)
    print("黑板演示——按主题键控的发布/订阅")
    print("=" * 72)
    bb = Blackboard()
    received = {"prices": [], "alerts": []}

    def on_prices(e):
        received["prices"].append(e.id)
    def on_alerts(e):
        received["alerts"].append(e.id)

    bb.subscribe("prices", on_prices)
    bb.subscribe("alerts", on_alerts)

    bb.publish("scraper-1", "prices", "AAPL=192.4", "poll market")
    bb.publish("scraper-2", "prices", "MSFT=401.2", "poll market")
    bb.publish("risk-engine", "alerts", "ALERT: AAPL moved >2%", "watch prices")

    print(f"  价格订阅者收到 id: {received['prices']}")
    print(f"  警报订阅者收到 id: {received['alerts']}")
    print("  （注意：价格订阅者从未看到警报——这就是要点）")


def main():
    run_without_verifier()
    run_with_verifier()
    demo_blackboard()
    print("\n要点:")
    print("  1. 没有溯源的共享状态将幻觉洗白为下游推理中的事实")
    print("  2. 不可写验证者独立获取来源并捕获记忆投毒")
    print("  3. 黑板缩放到全消息池之外，因为智能体只读取它们订阅的内容")


if __name__ == "__main__":
    main()
