#!/bin/bash

# 检查参数
if [ "$#" -ne 4 ]; then
    echo "用法: $0 <SSH_HOST> <SSH_USER> <REMOTE_DIR> <SSH_PORT>"
    echo "示例: $0 example.com user /home/user/hip_project 22"
    exit 1
fi

SSH_HOST=$1
SSH_USER=$2
REMOTE_DIR=$3
SSH_PORT=$4

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 运行打包脚本
echo "正在打包项目..."
bash "$PROJECT_ROOT/scripts/package_project.sh"

# 获取最新的打包文件
PACKAGE_NAME=$(ls -t hip_project_*.tar.gz | head -n1)

# 创建远程目录
echo "正在创建远程目录..."
ssh -p $SSH_PORT "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_DIR"

# 传输文件
echo "正在传输文件到服务器..."
scp -P $SSH_PORT "$PACKAGE_NAME" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/"

# 在服务器上解压
echo "正在服务器上解压文件..."
ssh -p $SSH_PORT "$SSH_USER@$SSH_HOST" "cd $REMOTE_DIR && tar -xzf $PACKAGE_NAME"

# 设置环境
echo "正在设置服务器环境..."
ssh -p $SSH_PORT "$SSH_USER@$SSH_HOST" "cd $REMOTE_DIR && \
    python -m venv venv && \
    source venv/bin/activate && \
    pip install -r requirements.txt"

echo "部署完成！"
echo "项目已部署到: $SSH_USER@$SSH_HOST:$REMOTE_DIR" 