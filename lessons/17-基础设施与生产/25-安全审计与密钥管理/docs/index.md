# 安全——密钥管理、审计日志与防护栏

> 通过集中式保险库消除密钥扩散（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）。AI 网关模式是 2026 年的解决方案：应用→网关→模型提供商，网关在运行时从保险库拉取凭据。在保险库中轮转，所有应用在几分钟内获取新密钥——无需重新部署。轮转策略 ≤90 天；每次提交用 TruffleHog/GitGuardian/Gitleaks 扫描。Guardrails：输入/输出 PII 脱敏、越狱检测、网络出口白名单。2026 年的标志性事件：Vercel 供应链攻击——CI/CD 凭据泄露导致数千客户环境变量泄露。

**类型：** 概念课
**语言：** Python
**前置知识：** 阶段 17 · 19（AI 网关）、阶段 17 · 13（可观测性）
**预计时间：** ~60 分钟
**所处阶段：** Tier 3

---

## 🎯 学习目标

完成本课后，你能够：

- [ ] 列举四个密钥管理反模式（VCS 中的配置文件、硬编码环境变量、电子表格、静态密钥）并说出替代方案
- [ ] 解释 AI 网关从保险库拉取模式作为 2026 年生产标准
- [ ] 实现带一致性令牌化（相同值→相同占位符）的 PII 脱敏器，使语义得以保留
- [ ] 命名 2026 年 Vercel 供应链事故及其关于 CI/CD 凭据卫生的教训

---

## 1. 问题

一个实习生提交了带 API 密钥的 `.env` 文件。他们迅速删除。密钥已经在 git 历史中了——GitGuardian 扫描捕获了它，你的轮转流程是"在 Slack 通知团队，更新 40 个配置文件，重新部署所有服务。"8 小时后一半服务已恢复，一半在等待部署窗口。

另外，用户提示词包含"我的身份证号码是 110101199001011234。"提示词发送到 LLM。你有 BAA，但你的内部策略是在转发前脱敏 PII。你没做。

另外，你的 EKS 集群的 LLM pod 可以访问任何互联网地址。有人通过 DNS 查询向攻击者控制的域名泄露数据。没有东西阻止它。

LLM 服务的安全必须覆盖这三个向量。保险库凭据、PII 脱敏、网络出口过滤、审计日志。

---

## 2. 概念

### 2.1 集中式保险库 + IAM 角色拉取

**保险库：** HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一真相源。

**IAM 角色：** 应用/网关通过其 IAM 身份认证，而非静态密钥。保险库在令牌生命周期内返回密钥。

**AI 网关模式：** 网关在请求时从保险库拉取 `OPENAI_API_KEY`。在保险库中轮转；下一个请求获取新密钥。无需重新部署。

### 2.2 轮转策略 ≤90 天

所有 API 密钥、保险库根令牌、CI/CD 凭据。尽可能自动轮转。手动轮转记录并跟踪。

### 2.3 密钥扫描

- **TruffleHog** — 提交时的正则+熵检测
- **GitGuardian** — 商业，高准确率
- **Gitleaks** — 开源，在 CI 中运行

每次提交都运行。检测到新密钥时阻断 PR。

### 2.4 PII/PHI 脱敏

提示词离开基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业产品）
2. 屏蔽匹配的实体：`"我的身份证是110101199001011234"` → `"我的身份证是[ID_TOKEN_A3F]"`
3. 一致性令牌化（Mesh 方法）：相同值映射到相同占位符——LLM 保持关系
4. 可选：LLM 响应的反向映射

正则过滤器捕获基本模式；NER 捕获更多。两者都用。

### 2.5 输入+输出 Guardrails

输入：阻止已知越狱、禁止话题；按用户速率限制。

输出：正则过滤泄露的密钥、禁止内容。

### 2.6 网络出口白名单

LLM 服务在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、保险库端点
- 其他全部丢弃
- DNS 通过仅允许列表解析器（避免 DNS 隧道泄露）

### 2.7 审计日志

每次 LLM 调用的不可变记录：
- 时间戳、用户/租户、提示词哈希（非原始提示词）、模型+版本、词元数、成本、响应哈希、Guardrail 触发

按监管要求保留（SOC 2 1 年、HIPAA 6 年）。

---

## 3. 从零实现

### 第 1 步：PII 脱敏器

```python
import re


class PIIRedactor:
    """PII 脱敏器——一致令牌化。"""

    def __init__(self):
        self.token_map = {}
        self.token_counter = 0
        self.patterns = {
            "身份证号": r"\d{18}[Xx]?",
            "手机号": r"1[3-9]\d{9}",
            "邮箱": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        }

    def redact(self, text: str) -> str:
        """脱敏文本中的 PII。"""
        for pattern_name, pattern in self.patterns.items():
            def replace_match(m):
                match_text = m.group(0)
                if match_text not in self.token_map:
                    self.token_counter += 1
                    self.token_map[match_text] = f"[{pattern_name}_{self.token_counter:04d}]"
                return self.token_map[match_text]
            text = re.sub(pattern, replace_match, text)
        return text


# 演示
redactor = PIIRedactor()
text1 = "我的身份证是110101199001011234，手机号13800138000"
text2 = "补充一下，身份证是110101199001011234"
print(f"脱敏 1: {redactor.redact(text1)}")
print(f"脱敏 2: {redactor.redact(text2)}")
```

完整代码见 `code/main.py`。

---

## 4. 工业工具

### 4.1 安全工具对照

| 领域 | 工具 | 功能 |
|---|---|---|
| 密钥存储 | HashiCorp Vault / AWS SM | 集中式凭证管理 |
| 密钥扫描 | TruffleHog / GitGuardian / Gitleaks | 提交时检测密钥泄露 |
| PII 脱敏 | Microsoft Presidio | 实体识别+屏蔽 |
| Guardrails | Portkey / Guardrails AI | 输入输出过滤 |
| 审计日志 | 自建 + OTel | 不可变记录 |

---

## 5. 工程最佳实践

### 5.1 密钥永远不在代码中

使用 AI 网关模式——网关从保险库拉取凭据，应用代码永远不接触 API 密钥。轮转密钥时只需在保险库中更新。

### 5.2 PII 脱敏必须在推理前

后处理清理（LLM 看到后再脱敏）不是可辩护的姿态——模型已经看到了数据。实时推理层脱敏是 2026 年的标准。

### 5.3 中文场景特别建议

- **中文 PII 格式不同。** 身份证号、手机号、银行卡号的中文格式与英文不同。正则表达式需要适配
- **国内密钥管理。** 自建 Vault、阿里云 KMS、华为云 KMS 是国内常见的密钥管理方案。AI 网关模式同样有效
- **国内网络出口。** 国内云厂商的 NAT 网关和安全组可以配置白名单模式。但国内 LLM API 的端点更多（阿里云、百度、华为云等），白名单需要覆盖多个域名

---

## 6. 常见错误

### 错误 1：密钥硬编码在代码中

**现象：** API 密钥在 git 历史中泄露。轮转需要重新部署 40 个服务。

**原因：** 没有使用集中式保险库。

**修复：** 使用 AI 网关模式——网关从保险库拉取凭据。轮转密钥时只需在保险库中更新。

### 错误 2：PII 后处理清理

**现象：** 审计时发现 LLM 已经看到了 PII。合规审核失败。

**原因：** PII 在发送到 LLM 之后才清理——模型已经看到了数据。

**修复：** PII 脱敏在推理之前——实时推理层脱敏是 2026 年的可辩护标准。

---

## 7. 面试考点

### Q1：2026 年 Vercel 供应链事故的教训是什么？（难度：⭐⭐）

**参考答案：**
CI/CD 凭据泄露导致成千上万的客户环境变量被泄露。教训是 CI/CD 凭据等同于生产凭据——必须存储在保险库中、严格限制作用范围、频繁轮转。CI/CD 管道中的环境变量（API 密钥、数据库连接字符串）是最敏感的数据之一。攻击者通过 CI/CD 凭据进入后可以提取所有环境变量。

### Q2：为什么一致性令牌化在 PII 脱敏中重要？（难度：⭐⭐⭐）

**参考答案：**
如果每次看到相同 SSN 都生成不同的随机掩码（如 `[PII_A3F]` 和 `[PII_B7D]`），LLM 无法知道它们是同一个实体——它在两个引用之间失去关系。一致性令牌化保证相同值映射到相同占位符——LLM 保持关系语义。例如"Alice 的 SSN 是 XXX，她现在用它登录"——如果 SSN 占位符保持一致，LLM 理解"它"指向同一个实体。

---

## 🔑 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 保险库 | "密钥存储" | 集中式凭据管理服务 |
| IAM 角色 | "基于身份的认证" | 应用通过身份而非静态密钥认证 |
| PII 脱敏 | "数据掩码" | 移除或令牌化敏感实体 |
| 一致性令牌化 | "稳定占位符" | 相同值→每次相同令牌 |
| 出口白名单 | "出站白名单" | 仅允许的域名可达 |
| 审计日志 | "不可变历史" | 用于合规的只追加记录 |

---

## 📚 小结

LLM 服务安全覆盖四个向量：凭据管理（集中式保险库+AI 网关模式）、PII 脱敏（推理前一致性令牌化）、网络出口（白名单模式）、审计日志（不可变记录）。关键是密钥永远不在代码中、PII 脱敏必须在推理前（而非后处理）、CI/CD 凭据等同于生产凭据。实时推理层脱敏是 2026 年的可辩护标准。

---

## ✏️ 练习

1. 运行 `code/main.py`。发送两个引用相同 SSN 的提示词——确认得到相同占位符。
2. 设计一个 vLLM-on-EKS 部署的网络出口策略——调用 OpenAI + Anthropic + Weaviate。

---

## 🚀 产出

| 产出 | 文件 | 说明 |
|---|---|---|
| PII 脱敏器 | `code/main.py` | 一致性令牌化的 PII 脱敏 |
| LLM 安全方案 | `outputs/skill-llm-security-plan.md` | 保险库迁移、脱敏器、出口、审计日志 |

---

## 📖 参考资料

1. [GitHub] Microsoft Presidio. https://github.com/microsoft/presidio
2. [官方文档] HashiCorp Vault. https://developer.hashicorp.com/vault/docs
3. [博客] Portkey — Manage LLM API keys with secret references. https://portkey.ai/blog/secret-references-ai-api-key-management/
4. [博客] Datadog — LLM Guardrails Best Practices. https://www.datadoghq.com/blog/llm-guardrails-best-practices/
