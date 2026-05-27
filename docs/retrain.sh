#!/bin/bash
# 使用转换后的标注数据训练模型
# 关键点定义更新为医学解剖学标准：
# 1. 左侧股骨头中心点(left_femoral_head_center)
# 2. 右侧股骨头中心点(right_femoral_head_center)
# 3. 左侧髋臼外缘点(left_acetabular_edge)
# 4. 右侧髋臼外缘点(right_acetabular_edge)
# 5. 耻骨联合点(pubic_symphysis)
# 6. 左侧髋臼荷重面内侧点(left_sourcil_medial)
# 7. 左侧髋臼荷重面外侧点(left_sourcil_lateral)
# 8. 右侧髋臼荷重面内侧点(right_sourcil_medial)
# 9. 右侧髋臼荷重面外侧点(right_sourcil_lateral)

# 设置工作目录
cd /home/xlian289-tNxksKkC/hip_project

# 设置PYTHONPATH
export PYTHONPATH=/home/xlian289-tNxksKkC/hip_project

# 激活环境
source /hpc2hdd/home/xlian289/miniconda3/bin/activate hip

# 准备训练数据目录
CONVERT_DIR="/home/xlian289-tNxksKkC/hip_project/data/convert_annotations"
TRAIN_DIR="/home/xlian289-tNxksKkC/hip_project/data/convert_train"
mkdir -p $TRAIN_DIR

# 复制所有转换后的标注文件
echo "复制转换后的标注文件..."
cp $CONVERT_DIR/*.json $TRAIN_DIR/

# 复制对应的图像文件
echo "复制对应的图像文件..."
for json_file in $TRAIN_DIR/*.json; do
    base_name=$(basename "$json_file" .json)
    
    # 查找对应的图像文件
    image_file=$(find /home/xlian289-tNxksKkC/hip_project/data/expand -name "${base_name}.jpg" -o -name "${base_name}.jpeg" -o -name "${base_name}.png")
    
    if [ -n "$image_file" ]; then
        # 复制图像到训练目录
        cp "$image_file" "$TRAIN_DIR/"
        echo "已复制: $base_name"
    else
        echo "警告：找不到与 $json_file 对应的图像文件"
    fi
done

echo "数据准备完成！共处理 $(ls -1 $TRAIN_DIR/*.json | wc -l) 个标注文件"

# 从预训练模型开始训练
echo "开始训练模型..."
PRETRAINED_MODEL="/home/xlian289-tNxksKkC/hip_project/outputs/model_best_20250506_121253.pth"

# 使用训练日志记录输出
LOG_FILE="/home/xlian289-tNxksKkC/hip_project/train_log_$(date +%Y%m%d_%H%M%S).log"
echo "训练日志将保存到: $LOG_FILE"

# 使用python执行训练脚本，注意加载预训练模型
python train.py \
    --data-dir $TRAIN_DIR \
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