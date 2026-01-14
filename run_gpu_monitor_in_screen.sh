#!/bin/bash
# GPU 监控脚本的 screen 启动器
# 自动激活 conda 环境并运行监控脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 bashrc（确保 direnv 和 conda 初始化）
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# 如果 direnv 已安装，允许当前目录的 .envrc
if command -v direnv &> /dev/null; then
    direnv allow . 2>/dev/null || true
fi

# 显式激活 conda 环境（双重保险）
if command -v conda &> /dev/null; then
    conda activate TimeCMA_Qwen3 2>/dev/null || {
        echo "⚠ 警告: 无法激活 conda 环境 TimeCMA_Qwen3"
        echo "尝试使用 direnv 设置的环境变量..."
    }
fi

# 显示当前环境信息
echo "=========================================="
echo "GPU 监控脚本启动"
echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "工作目录: $(pwd)"
echo "Python 路径: $(which python3 2>/dev/null || echo '未找到')"
echo "Conda 环境: ${CONDA_DEFAULT_ENV:-未激活}"
echo "=========================================="
echo ""

# 运行监控脚本（传递所有参数）
exec python3 monitor_gpus_and_notify.py "$@"
