"""合规控制映射矩阵。"""

FRAMEWORKS = {
    "SOC 2 Type II": {"scope": "B2B SaaS", "llm_req": "6-12月控制审计"},
    "GDPR": {"scope": "欧盟用户", "llm_req": "实时PII脱敏"},
    "HIPAA": {"scope": "美国医疗", "llm_req": "BAA必需"},
    "EU AI Act": {"scope": "欧盟用户", "llm_req": "高风险强制执行2026.08.02"},
}

CONTROL_MAP = {
    "访问日志": ["ISO 27001 A.5.15-5.18", "GDPR Art.32", "HIPAA §164.312(a)"],
    "传输加密": ["ISO 27001 A.8.24", "GDPR Art.32", "HIPAA §164.312(e)"],
    "密钥管理": ["ISO 27001 A.8.19", "PCI DSS Req.8", "SOC 2 CC6.1"],
}

if __name__ == "__main__":
    print("控制映射矩阵:")
    for control, frameworks in CONTROL_MAP.items():
        print(f"  {control}: {'|'.join(frameworks)}")
