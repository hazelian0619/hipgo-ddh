#!/bin/bash
# 从服务器下载最佳模型和样本数据

# 创建必要的目录
mkdir -p models
mkdir -p data/samples
mkdir -p data/json_annotations

# 下载最佳模型
echo "下载最佳训练模型..."
scp -P 6988 xlian289-tNxksKkC@10.120.18.240:/home/xlian289-tNxksKkC/hip_project/outputs/model_best_20250506_163007.pth models/

# 下载几个样本图像用于测试
echo "下载样本图像..."
ssh -p 6988 xlian289-tNxksKkC@10.120.18.240 "cd /home/xlian289-tNxksKkC/hip_project && find data/convert_train -name \"*.jpg\" | head -n 5" | while read file; do
    scp -P 6988 "xlian289-tNxksKkC@10.120.18.240:${file}" data/samples/
done

# 下载对应的标注文件
echo "下载标注文件..."
ssh -p 6988 xlian289-tNxksKkC@10.120.18.240 "cd /home/xlian289-tNxksKkC/hip_project && find data/convert_train -name \"*.json\" | head -n 5" | while read file; do
    scp -P 6988 "xlian289-tNxksKkC@10.120.18.240:${file}" data/json_annotations/
done

# 下载训练日志用于分析
echo "下载训练日志..."
scp -P 6988 xlian289-tNxksKkC@10.120.18.240:/home/xlian289-tNxksKkC/hip_project/train_log_20250506_*.log ./

echo "下载完成！" 