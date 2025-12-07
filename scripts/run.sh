#!/bin/bash
set -e

if ! command -v nvidia-smi &> /dev/null
then
    echo "错误：未检测到NVIDIA GPU环境，请确保已安装CUDA"
    exit 1
fi

if ! python -c "import sys; exit(0) if sys.version_info >= (3,8) else exit(1)" &> /dev/null
then
    echo "错误：请使用Python 3.8及以上版本"
    exit 1
fi

echo "===== 安装依赖库 ====="
pip install -r requirements.txt

# 检查数据是否存在，不存在则提示
if [ ! -f "data.csv" ]; then
    echo "警告：未找到data.csv，请将原始数据集放在项目根目录"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
else
    echo "===== 开始数据预处理 ====="
    python src/predata.py
fi

if [ ! -d "./qwen-1.5b" ]; then
    echo "错误：未找到基础模型，请将Qwen-1.5B模型文件放在./qwen-1.5b目录"
    exit 1
fi

echo "===== 开始模型微调 ====="
python src/fine_tune.py

echo "===== 微调完成，开始交互测试 ====="
python src/chat_fine.py