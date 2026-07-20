# JSON-RPC传输技能

## 目标
实现JSON-RPC 2.0换行分隔stdio传输，支持请求/响应/通知/批处理/错误码。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 五错误码正确映射 | 所有错误码语义正确 |
| 20 | 通知不返回响应 | 通知条目在批处理中被跳过 |
| 20 | 解析错误不中断流 | 坏行后继续处理下一行 |
| 20 | 批处理正确 | 数组入→数组出，通知被跳过 |
| 15 | io.BytesIO演示 | 无进程spawn的自终止演示 |

## 构建检查清单

- [ ] StdioTransport类（read_line/write_response/write_error/write_notification）
- [ ] 五错误码常量
- [ ] 请求解析（parse_request）
- [ ] 信封验证（_is_valid_envelope）
- [ ] 分发循环（serve）
- [ ] 批处理支持
- [ ] 通知处理（无响应）
- [ ] io.BytesIO自终止演示
