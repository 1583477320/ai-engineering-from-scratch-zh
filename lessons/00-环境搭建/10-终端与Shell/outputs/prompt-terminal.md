# AI 终端技能速查

## tmux

```bash
tmux new -s train    # 创建会话
tmux attach -t train # 重连
tmux ls             # 列出会话
# Ctrl+B d           # 分离
# Ctrl+B "           # 水平分割
# Ctrl+B %           # 垂直分割
```

## SSH

```bash
ssh user@gpu-box-ip
scp model.pt user@gpu-box:~/models/
rsync -avz ./data/ user@gpu-box:~/data/
ssh -L 8888:localhost:8888 user@gpu-box  # 端口转发
```

## 日志监控

```bash
tail -f train.log | grep --line-buffered "loss"
grep "ERROR" *.log | sort -u
```
