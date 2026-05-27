#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例：如何正确使用0506最佳模型

本示例展示如何使用骨盆X光片分析的最佳模型(0506版本)进行关键点检测及角度测量
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 项目根目录
sys.path.append(project_root)

# 导入必要的模块（使用最佳模型）
from hip_model.utils.model_loader import load_best_model
from hip_model.dataset import get_transforms
from hip_analysis.mllm_medical_report import HipMLLM
from hip_model.visualize_angles import calculate_angles, visualize_keypoints_and_angles

def process_image(image_path, output_dir="outputs"):
    """
    使用最佳模型(0506)处理X光片图像
    
    参数:
        image_path: 图像路径
        output_dir: 输出目录
    """
    print(f"\n{'='*50}")
    print(f"处理图像: {image_path}")
    print(f"{'='*50}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. 加载最佳模型(0506)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_best_model(device=device)  # 这里使用我们专门创建的加载函数
    
    # 2. 读取和预处理图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 应用预处理
    transform = get_transforms(train=False)
    transformed = transform(image=image_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # 3. 进行预测
    with torch.no_grad():
        outputs = model(img_tensor)
        keypoints = outputs['keypoints'].cpu().numpy()[0]
        # 角度预测（如果模型支持）
        if 'angles' in outputs:
            predicted_angles = outputs['angles'].cpu().numpy()[0]
            print("\n预测的角度值:")
            angle_names = ['left_ce', 'right_ce', 'left_sharp', 'right_sharp', 'left_tonnis', 'right_tonnis']
            for i, name in enumerate(angle_names):
                print(f"{name}: {predicted_angles[i]:.2f}°")
    
    # 4. 计算角度
    angles = calculate_angles(keypoints)
    
    # 打印计算的角度
    print("\n计算的角度值:")
    for key, value in angles.items():
        print(f"{key}: {value:.2f}°")
    
    # 5. 可视化关键点和角度
    output_image_path = os.path.join(output_dir, f"{base_name}_keypoints.png")
    fig = visualize_keypoints_and_angles(
        image_path, 
        keypoints, 
        output_path=output_image_path,
        title="0506最佳模型预测结果"
    )
    
    print(f"\n关键点和角度可视化已保存至: {output_image_path}")
    
    # 6. 生成医学报告（如果需要）
    try:
        # 保存关键点为JSON文件，供报告生成使用
        keypoints_path = os.path.join(output_dir, f"{base_name}_keypoints.json")
        keypoints_list = keypoints.tolist()
        import json
        with open(keypoints_path, 'w') as f:
            json.dump({"keypoints": keypoints_list}, f)
        
        # 初始化MLLM模型并生成报告
        mllm = HipMLLM()
        report = mllm.generate_report(
            image_path=image_path,
            keypoints_path=keypoints_path,
            output_dir=output_dir
        )
        
        print("\n生成的医学报告:")
        print("-" * 50)
        print(report)
        print("-" * 50)
        
    except Exception as e:
        print(f"\n生成医学报告失败: {str(e)}")
    
    return keypoints, angles

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='使用0506最佳模型进行骨盆X光片分析')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default='outputs', help='输出目录')
    
    args = parser.parse_args()
    
    process_image(args.image, args.output)

if __name__ == "__main__":
    main() 