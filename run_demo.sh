#!/bin/bash
# CyberVerse 数字人演示启动脚本
set -e

export NO_PROXY="192.168.1.101,localhost,127.0.0.1"
export no_proxy="192.168.1.101,localhost,127.0.0.1"
export PATH="/home/test/.conda/envs/cyberverse/bin:$PATH"

PYTHON="/home/test/.conda/envs/LatentSync/bin/python"
DEMO="/home/test/CyberVerse/demo.py"

echo "启动 CyberVerse 数字人演示..."
echo "浏览器访问: http://$(hostname -I | awk '{print $1}'):7860"

exec $PYTHON "$DEMO" "$@"
