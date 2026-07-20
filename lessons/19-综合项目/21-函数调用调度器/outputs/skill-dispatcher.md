# 函数调用调度器技能

## 目标
构建带超时、重试、幂等去重和并发限制的异步调度器。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 超时和重试正确 | timeout+幂等→重试，非幂等→不重试 |
| 20 | 幂等去重 | 相同键的并发调用合并为一次处理 |
| 20 | 错误映射统一 | 所有异常映射到DispatchError信封 |
| 20 | 并发限制 | 信号量正确限制同时运行数 |
| 15 | 预算感知 | budget_tool_calls_remaining正确触发 |

## 构建检查清单

- [ ] Dispatcher类（dispatch异步方法）
- [ ] DispatchError数据类（kind/message/attempts/jsonrpc_code）
- [ ] TransientError异常
- [ ] 指数退避+抖动（_backoff）
- [ ] 幂等键缓存+在途检测
- [ ] asyncio.wait_for超时
- [ ] 信号量并发限制
- [ ] _invoke异步/同步处理程序适配
- [ ] 演示：重试成功+超时+schema+缺失+幂等
