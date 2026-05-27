#!/bin/bash

# SSH连接信息
SSH_HOST="10.120.18.240"
SSH_USER="xlian289-tNxksKkC"
SSH_PORT="6988"
REMOTE_DIR="/home/xlian289-tNxksKkC/hip_project"

# 创建远程目录
echo "创建远程目录..."
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p $REMOTE_DIR/outputs"

# 上传模型文件
echo "上传模型文件..."
scp -P $SSH_PORT outputs/model_best_20250506_121253.pth $SSH_USER@$SSH_HOST:$REMOTE_DIR/outputs/
scp -P $SSH_PORT outputs/model_best_20250506_121040.pth $SSH_USER@$SSH_HOST:$REMOTE_DIR/outputs/
scp -P $SSH_PORT outputs/training_history_20250506_123631.png $SSH_USER@$SSH_HOST:$REMOTE_DIR/outputs/

# 创建数据目录
echo "创建数据目录..."
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p $REMOTE_DIR/data/expand"

# 上传新数据集
echo "上传新数据集..."
scp -P $SSH_PORT -r /Users/pluviophile/Downloads/data\ expand/* $SSH_USER@$SSH_HOST:$REMOTE_DIR/data/expand/

echo "上传完成！" 