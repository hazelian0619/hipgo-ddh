#!/bin/bash

# 激活hip环境
source /hpc2hdd/home/xlian289/miniconda3/bin/activate hip

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p /home/xlian289-tNxksKkC/hip_project/data/expand_annotated

# 开始运行自动标注脚本
echo "开始运行自动标注脚本..."
cd /home/xlian289-tNxksKkC/hip_project
PYTHONPATH=/home/xlian289-tNxksKkC/hip_project python scripts/auto_annotate.py \
    --model_path /home/xlian289-tNxksKkC/hip_project/outputs/model_best_20250506_121253.pth \
    --data_dir /home/xlian289-tNxksKkC/hip_project/data/expand \
    --output_dir /home/xlian289-tNxksKkC/hip_project/data/expand_annotated \
    --batch_size 4 \
    --num_workers 2 \
    --image_size 512

echo "自动标注流程完成!" 