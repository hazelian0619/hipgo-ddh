#!/bin/bash
# 批量一致性检测脚本（使用0506最佳模型）

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印标题
echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}     髋关节X光片批量一致性检测系统       ${NC}"
echo -e "${BLUE}     (使用0506最佳模型)                  ${NC}"
echo -e "${BLUE}===========================================${NC}"

# 默认参数
IMAGES_DIR="data/images"
DOCTORS_DIR="data/annotations"
OUTPUT_DIR="consistency_results"

# 帮助函数
show_help() {
    echo -e "${GREEN}用法:${NC}"
    echo "  $0 [选项]"
    echo ""
    echo -e "${GREEN}选项:${NC}"
    echo "  -h, --help             显示帮助信息"
    echo "  -i, --images DIR       指定X光片图像目录 (默认: $IMAGES_DIR)"
    echo "  -d, --doctors DIR      指定医生标注目录 (默认: $DOCTORS_DIR)"
    echo "  -o, --output DIR       指定输出目录 (默认: $OUTPUT_DIR)"
    echo "  -n, --num NUMBER       处理的最大图像数量 (默认: 所有)"
    echo ""
    echo -e "${GREEN}示例:${NC}"
    echo "  $0 --images data/test_images --output results"
    echo ""
}

# 解析命令行参数
NUM_IMAGES=9999  # 默认处理所有图像

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--images)
            IMAGES_DIR="$2"
            shift 2
            ;;
        -d|--doctors)
            DOCTORS_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--num)
            NUM_IMAGES="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查目录
if [ ! -d "$IMAGES_DIR" ]; then
    echo -e "${RED}错误: 图像目录不存在: $IMAGES_DIR${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取图像列表
images=("$IMAGES_DIR"/*.jpg "$IMAGES_DIR"/*.png)
if [ ${#images[@]} -eq 0 ]; then
    echo -e "${RED}错误: 在目录 $IMAGES_DIR 中未找到图像文件${NC}"
    exit 1
fi

# 限制处理数量
if [ ${#images[@]} -gt $NUM_IMAGES ]; then
    images=("${images[@]:0:$NUM_IMAGES}")
fi

echo -e "${GREEN}找到 ${#images[@]} 个图像文件${NC}"
echo -e "${YELLOW}将处理以下图像:${NC}"
for img in "${images[@]}"; do
    echo "  - $(basename "$img")"
done

# 处理每张图像
echo -e "\n${BLUE}开始批量处理...${NC}"
processed=0
successful=0

for img in "${images[@]}"; do
    # 跳过非文件
    if [ ! -f "$img" ]; then
        continue
    fi
    
    base_name=$(basename "$img" | sed 's/\.[^.]*$//')
    doctor_file="$DOCTORS_DIR/${base_name}.json"
    
    echo -e "\n${YELLOW}[$(($processed+1))/${#images[@]}] 处理: $base_name${NC}"
    
    doctor_arg=""
    if [ -f "$doctor_file" ]; then
        doctor_arg="--doctor \"$doctor_file\""
        echo -e "找到医生标注: $(basename "$doctor_file")"
    else
        echo -e "未找到医生标注，将只进行模型预测"
    fi
    
    # 构建命令
    cmd="python hip_model/consistent_check.py --image \"$img\" $doctor_arg --output \"$OUTPUT_DIR\""
    echo -e "执行: $cmd"
    
    # 执行命令
    eval $cmd
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功处理: $base_name${NC}"
        successful=$((successful+1))
    else
        echo -e "${RED}处理失败: $base_name${NC}"
    fi
    
    processed=$((processed+1))
done

# 输出统计信息
echo -e "\n${BLUE}===========================================${NC}"
echo -e "${GREEN}批量处理完成!${NC}"
echo -e "总共处理: $processed 张图像"
echo -e "成功处理: $successful 张图像"
echo -e "失败数量: $((processed-successful)) 张图像"
echo -e "结果保存在目录: $OUTPUT_DIR"
echo -e "${BLUE}===========================================${NC}" 