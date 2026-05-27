#!/bin/bash
# 骨盆X光片多模态医学大语言模型运行脚本
# 用于生成医学报告

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印标题
echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}       骨盆X光片医学报告生成系统         ${NC}"
echo -e "${BLUE}===========================================${NC}"

# 默认参数
MODEL_PATH="models/llava_med"
OUTPUT_DIR="reports"
DEVICE=""
LOW_RESOURCE=""
PROMPT=""
TEMPERATURE="0.7"
MAX_TOKENS="1024"

# 帮助函数
show_help() {
    echo -e "${GREEN}用法:${NC}"
    echo "  $0 [选项]"
    echo ""
    echo -e "${GREEN}选项:${NC}"
    echo "  -h, --help             显示帮助信息"
    echo "  -i, --image 文件路径    指定X光片图像文件路径 (必须)"
    echo "  -k, --keypoints 文件路径 指定关键点JSON文件路径 (可选)"
    echo "  -m, --model 模型路径    指定模型路径 (默认: $MODEL_PATH)"
    echo "  -o, --output 目录       指定输出目录 (默认: $OUTPUT_DIR)"
    echo "  -d, --device 设备       指定运行设备 (cuda|cpu|mps)"
    echo "  -l, --low-resource      低资源模式 (低显存设备)"
    echo "  -p, --prompt 提示词     自定义提示词"
    echo "  -t, --temperature 数值  生成温度,控制随机性 (默认: $TEMPERATURE)"
    echo "  --max-tokens 数值       最大生成token数 (默认: $MAX_TOKENS)"
    echo "  -b, --batch 目录        批量处理指定目录中的所有图像"
    echo "  -n, --num 数值          批量处理时的最大图像数 (默认: 3)"
    echo ""
    echo -e "${GREEN}示例:${NC}"
    echo "  $0 --image data/sample.jpg"
    echo "  $0 --image data/sample.jpg --keypoints data/sample.json --device cuda"
    echo "  $0 --batch data/images --num 5 --device cuda"
    echo ""
}

# 检查命令行参数
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# 解析命令行参数
BATCH_MODE=false
BATCH_DIR=""
NUM_IMAGES=3
IMAGE_PATH=""
KEYPOINTS_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        -k|--keypoints)
            KEYPOINTS_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="--device $2"
            shift 2
            ;;
        -l|--low-resource)
            LOW_RESOURCE="--low_resource"
            shift
            ;;
        -p|--prompt)
            PROMPT="--prompt \"$2\""
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_MODE=true
            BATCH_DIR="$2"
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

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行模型
if [ "$BATCH_MODE" = true ]; then
    if [ -z "$BATCH_DIR" ]; then
        echo -e "${RED}错误: 批量模式需要指定图像目录${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}批量处理模式${NC}"
    echo -e "${YELLOW}图像目录: $BATCH_DIR${NC}"
    echo -e "${YELLOW}最多处理: $NUM_IMAGES 张图像${NC}"
    
    CMD="python3 test_mllm.py --dir \"$BATCH_DIR\" --num $NUM_IMAGES --model \"$MODEL_PATH\" --output \"$OUTPUT_DIR\" $DEVICE $LOW_RESOURCE $PROMPT --temperature $TEMPERATURE --max_tokens $MAX_TOKENS"
    
else
    if [ -z "$IMAGE_PATH" ]; then
        echo -e "${RED}错误: 必须指定图像文件路径${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}单图像处理模式${NC}"
    echo -e "${YELLOW}图像文件: $IMAGE_PATH${NC}"
    
    KEYPOINTS_ARG=""
    if [ -n "$KEYPOINTS_PATH" ]; then
        KEYPOINTS_ARG="--keypoints \"$KEYPOINTS_PATH\""
        echo -e "${YELLOW}关键点文件: $KEYPOINTS_PATH${NC}"
    fi
    
    CMD="python3 test_mllm.py --image \"$IMAGE_PATH\" $KEYPOINTS_ARG --model \"$MODEL_PATH\" --output \"$OUTPUT_DIR\" $DEVICE $LOW_RESOURCE $PROMPT --temperature $TEMPERATURE --max_tokens $MAX_TOKENS"
fi

echo -e "${GREEN}执行命令: ${NC}"
echo "$CMD"
echo -e "${BLUE}===========================================${NC}"
echo -e "${YELLOW}处理中，请稍候...${NC}"

# 执行命令
eval "$CMD"

echo -e "${BLUE}===========================================${NC}"
echo -e "${GREEN}处理完成！${NC}"
echo -e "${GREEN}报告保存在目录: $OUTPUT_DIR${NC}"
echo -e "${BLUE}===========================================${NC}" 