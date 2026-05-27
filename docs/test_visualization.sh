#!/bin/bash
# 测试更新后的骨盆关键点可视化功能

# 确保目录存在
mkdir -p output_images

echo "测试模型预测可视化..."
if [ -f "models/model_best_20250506_163007.pth" ]; then
    python visualize_angles.py --image data/samples/$(ls data/samples | head -n 1) \
                             --model models/model_best_20250506_163007.pth \
                             --output output_images/model_prediction.jpg
    echo "模型预测可视化已保存到 output_images/model_prediction.jpg"
else
    echo "警告: 未找到模型文件，请先运行 download_model.sh 下载模型"
fi

echo "测试标注数据可视化..."
if [ -d "data/json_annotations" ] && [ "$(ls -A data/json_annotations)" ]; then
    json_file=$(ls data/json_annotations | head -n 1)
    image_file=${json_file%.json}.jpg
    
    if [ -f "data/samples/$image_file" ]; then
        python visualize_angles.py --image data/samples/$image_file \
                                 --json data/json_annotations/$json_file \
                                 --output output_images/annotation_visualization.jpg
        echo "标注数据可视化已保存到 output_images/annotation_visualization.jpg"
    else
        echo "警告: 找不到对应的图像文件 data/samples/$image_file"
    fi
else
    echo "警告: 未找到JSON标注文件，请先运行 download_model.sh 下载样本数据"
fi

echo "测试完成" 