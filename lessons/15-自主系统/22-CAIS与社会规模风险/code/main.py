"""CAIS 四风险清单——纯标准库。

给定提议部署的简短特征集，对 CAIS 四风险类别（恶意使用、AI 竞赛、
组织风险、流氓 AI）进行标记并返回缓解检查清单。
教学用途；真正的框架需要人类判断。

运行：python3 code/main.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Deployment:
    name: str
    public_facing: bool
    handles_harmful_capabilities: bool  # 生物/网络提升可能？
    competitive_pressure: bool         # 为了领先竞争者而匆忙部署？
    independent_audit: bool
    multi_layer_defense: bool
    information_security: bool         # 权重/评估/密钥已加固
    agent_autonomy_hours: float        # 每第 1/21 课


# ── 缓解映射 ──────────────────────────────────────────────

MITIGATIONS = {
    "malicious_use": [
        "宪法硬编码禁令（第 17 课）",
        "Llama Guard 输入/输出分类器（第 18 课）",
        "每任务工具白名单（第 10、11 课）",
    ],
    "ai_races": [
        "带常设风险报告的扩展策略（第 19、20 课）",
        "公开的前沿安全路线图，声明节奏",
        "METR / CAISI 外部能力评估（第 21 课）",
    ],
    "organizational_risks": [
        "内部安全文化；无职业代价的升级路径",
        "按声明节奏的独立审计",
        "多层防御（第 10、13、14、17、18 课）",
        "按 RAND SL-4 的信息安全（第 19 课行业层）",
    ],
    "rogue_ais": [
        "终止开关和金丝雀标记（第 14 课）",
        "先提议后提交 HITL（第 15 课）",
        "欺骗性对齐监控（第 20 课 DeepMind FSF）",
        "持久检查点和回滚（第 16 课）",
    ],
}


# ── 风险标记 ──────────────────────────────────────────────

def tag(d: Deployment) -> list[str]:
    """根据部署特征标记风险类别。"""
    tags = []
    if d.handles_harmful_capabilities and d.public_facing:
        tags.append("malicious_use")
    if d.competitive_pressure:
        tags.append("ai_races")
    org_missing = (
        (not d.independent_audit)
        or (not d.multi_layer_defense)
        or (not d.information_security)
    )
    if org_missing:
        tags.append("organizational_risks")
    if d.agent_autonomy_hours >= 4.0:
        tags.append("rogue_ais")
    return tags


# ── 报告 ──────────────────────────────────────────────────

def report(d: Deployment) -> None:
    tags = tag(d)
    print(f"\n部署: {d.name}")
    print("-" * 70)
    print(f"  public_facing         = {d.public_facing}")
    print(f"  harmful_capabilities  = {d.handles_harmful_capabilities}")
    print(f"  competitive_pressure  = {d.competitive_pressure}")
    print(f"  independent_audit     = {d.independent_audit}")
    print(f"  multi_layer_defense   = {d.multi_layer_defense}")
    print(f"  information_security  = {d.information_security}")
    print(f"  autonomy_hours        = {d.agent_autonomy_hours}")
    print()
    if tags:
        print(f"  标记的风险: {tags}")
        for t in tags:
            print(f"\n  {t} 的缓解措施:")
            for m in MITIGATIONS[t]:
                print(f"    - {m}")
    else:
        print("  未标记风险（需手动检查子杠杆）")


# ── 主函数 ────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("CAIS 四风险清单（阶段 15，第 22 课）")
    print("=" * 70)

    low = Deployment(
        name="内部重构助手（限定项目仓库）",
        public_facing=False, handles_harmful_capabilities=False,
        competitive_pressure=False, independent_audit=True,
        multi_layer_defense=True, information_security=True,
        agent_autonomy_hours=1.0)

    mid = Deployment(
        name="公共编码智能体（SaaS，通用用户）",
        public_facing=True, handles_harmful_capabilities=False,
        competitive_pressure=True, independent_audit=True,
        multi_layer_defense=True, information_security=False,
        agent_autonomy_hours=4.0)

    high = Deployment(
        name="自主 ML 研究智能体（前沿）",
        public_facing=True, handles_harmful_capabilities=True,
        competitive_pressure=True, independent_audit=False,
        multi_layer_defense=False, information_security=False,
        agent_autonomy_hours=48.0)

    for d in (low, mid, high):
        report(d)

    print()
    print("=" * 70)
    print("要点：组织风险是从业者实际能拉的杠杆")
    print("-" * 70)
    print("  恶意使用、AI 竞赛、流氓 AI 是结构性力量。")
    print("  组织风险在你的组织内部。安全文化、独立审计、")
    print("  多层防御、信息安全是每个团队控制的四个杠杆。")
    print("  部署速度压力与这四个杠杆对抗；CAIS 将其列为")
    print("  命名风险类别是有原因的。")


if __name__ == "__main__":
    main()
