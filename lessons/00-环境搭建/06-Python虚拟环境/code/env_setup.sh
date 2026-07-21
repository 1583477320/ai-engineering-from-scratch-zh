#!/bin/bash
# 课程环境设置脚本
set -e

echo "=== Python 环境设置 ==="

# 检查 Python 版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python 版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "虚拟环境路径: $(which python)"

# 安装核心依赖
echo "安装核心依赖..."
pip install -q numpy matplotlib jupyter scikit-learn

echo "=== 验证 ==="
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"

echo "=== 完成 ==="
echo "激活: source .venv/bin/activate"
