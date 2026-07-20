# 沙箱运行器技能

## 目标
构建带拒绝列表、路径监狱、超时和截断的子进程运行器。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | 四拒绝轴正确 | 名称/argv/元字符/路径监狱全部工作 |
| 20 | 路径监狱防逃逸 | realpath符号链接测试 |
| 20 | 超时和截断 | wall-clock超时+输出限制 |
| 20 | 结构化结果 | SandboxResult含所有字段 |
| 15 | 集成 | 与门链（第23节）端到端工作 |

## 构建检查清单

- [ ] SandboxConfig（project_root/max_output_bytes/timeout_seconds/denylist）
- [ ] SandboxResult（exit_code/stdout/stderr/denied/timed_out/truncated）
- [ ] 四个拒绝辅助器（_check_denylist/interp/metachars/path_jail）
- [ ] 输出截断（truncate_stream）
- [ ] Sandbox.run方法（shell=False/cwd/env清理）
- [ ] 演示：合法调用+拒绝列表+解释器+路径逃逸
