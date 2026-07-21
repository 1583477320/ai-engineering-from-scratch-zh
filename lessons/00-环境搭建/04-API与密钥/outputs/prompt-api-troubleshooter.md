# API 错误诊断

| 错误 | 可能原因 | 修复 |
|:-----|:--------|:-----|
| 401 Unauthorized | 密钥无效或格式错误 | 检查前缀 sk-ant- |
| 429 Too Many Requests | 速率限制 | 指数退避重试 |
| 404 Not Found | 端点 URL 错误 | 检查 API 版本 |
| 500 Internal Error | 服务端问题 | 等待后重试 |
