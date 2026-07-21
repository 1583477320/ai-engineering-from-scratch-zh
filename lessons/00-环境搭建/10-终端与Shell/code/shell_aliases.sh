# AI 工作常用 Shell 别名
# 使用: source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh

# GPU 状态一目了然
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader'

# 终止所有 Python 训练进程
alias killtraining='pkill -f "python.*train"'

# 快速激活虚拟环境
alias ae='source .venv/bin/activate'

# 监控训练损失
alias watchloss='tail -f logs/*.log 2>/dev/null | grep --line-buffered "loss"'

# 快速检查项目大小
alias projsize='find . -name "*.py" | xargs wc -l | tail -1'

# 快速检查磁盘
alias diskusage='df -h && du -sh ./* 2>/dev/null | sort -rh | head -10'
