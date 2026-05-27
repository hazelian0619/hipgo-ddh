#!/bin/bash

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="hip_project_${TIMESTAMP}.tar.gz"

# 创建临时目录
TEMP_DIR=$(mktemp -d)

# 复制项目文件到临时目录
echo "正在复制项目文件..."
cp -r "$PROJECT_ROOT"/* "$TEMP_DIR/"

# 创建requirements.txt
echo "正在生成requirements.txt..."
pip freeze > "$TEMP_DIR/requirements.txt"

# 创建README.md
echo "正在创建README.md..."
cat > "$TEMP_DIR/README.md" << EOF
# 髋关节角度测量项目

## 项目结构
\`\`\`
hip_project/
├── data/               # 数据目录
│   ├── raw/           # 原始数据
│   ├── processed/     # 处理后的数据
│   └── version_control/ # 数据版本控制
├── models/            # 模型目录
│   ├── checkpoints/   # 模型检查点
│   └── version_control/ # 模型版本控制
├── configs/           # 配置文件
├── scripts/           # 脚本文件
├── utils/             # 工具函数
├── logs/              # 日志文件
└── experiments/       # 实验记录
\`\`\`

## 环境要求
- Python 3.9+
- PyTorch 1.8+
- CUDA 11.0+ (GPU训练需要)

## 安装
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 使用说明
1. 数据准备
2. 模型训练
3. 模型评估
EOF

# 打包项目
echo "正在打包项目..."
tar -czf "$PACKAGE_NAME" -C "$TEMP_DIR" .

# 清理临时目录
rm -rf "$TEMP_DIR"

echo "项目打包完成: $PACKAGE_NAME" 