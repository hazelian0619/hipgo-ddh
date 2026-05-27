#!/bin/bash

# 激活hip环境
source /hpc2hdd/home/xlian289/miniconda3/bin/activate hip

# 设置工作目录
cd /home/xlian289-tNxksKkC/hip_project

# 设置PYTHONPATH
export PYTHONPATH=/home/xlian289-tNxksKkC/hip_project

# 准备训练数据：将标注好的数据整理到正确格式
echo "正在准备训练数据..."

# 创建目录存放训练数据
DATA_DIR="/home/xlian289-tNxksKkC/hip_project/data/expand_train"
mkdir -p $DATA_DIR

# 将图像和对应标注复制到训练目录
for json_file in /home/xlian289-tNxksKkC/hip_project/data/expand_annotated/*.json; do
    # 提取基本文件名（不含扩展名）
    base_name=$(basename "$json_file" .json)
    
    # 查找对应的图像文件
    image_file=$(find /home/xlian289-tNxksKkC/hip_project/data/expand -name "${base_name}.jpg" -o -name "${base_name}.jpeg" -o -name "${base_name}.png")
    
    if [ -n "$image_file" ]; then
        # 复制图像和标注到训练目录
        cp "$image_file" "$DATA_DIR/"
        cp "$json_file" "$DATA_DIR/"
        echo "已复制: $base_name"
    else
        echo "警告：找不到与 $json_file 对应的图像文件"
    fi
done

echo "数据准备完成！共处理 $(ls -1 $DATA_DIR/*.json | wc -l) 个标注文件"

# 从预训练模型开始训练
echo "开始训练模型..."
PRETRAINED_MODEL="/home/xlian289-tNxksKkC/hip_project/outputs/model_best_20250506_121253.pth"

# 使用训练日志记录输出
LOG_FILE="/home/xlian289-tNxksKkC/hip_project/train_log_$(date +%Y%m%d_%H%M%S).log"
echo "训练日志将保存到: $LOG_FILE"

# 使用python执行训练脚本，注意加载预训练模型
python train.py \
    --data-dir $DATA_DIR \
    --output-dir /home/xlian289-tNxksKkC/hip_project/outputs \
    --img-size 512 \
    --split-ratio 0.8 \
    --batch-size 4 \
    --lr 0.0001 \
    --epochs 30 \
    --patience 8 \
    --pretrained \
    --resume $PRETRAINED_MODEL \
    --new-optimizer | tee $LOG_FILE

echo "训练完成！" 