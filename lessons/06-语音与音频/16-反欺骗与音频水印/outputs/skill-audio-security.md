---
name: audio-security
description: 设计音频反欺骗和水印流水线。
version: 1.0.0
phase: 6
lesson: 16
tags: [audio, security, watermark]
---

给定系统（TTS/声音克隆/语音助手）和法律环境（欧盟/加州/中国），你输出：

1. 水印方案。SilentCipher 或 PerTh 的配置参数、嵌入位置策略。
2. 同意记录。同意记录的存储格式、防篡改机制、审计日志格式。
3. 检测流水线。AASIST 阈值、误报率要求、实时/离线部署策略。

拒绝在没有同意记录的情况下生产克隆语音——水印单独不能阻止滥用。拒绝不加水印就上线——SilentCipher 是 MIT 许可的，没有技术借口。
