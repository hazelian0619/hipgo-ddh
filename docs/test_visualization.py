#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
骨盆关键点可视化测试脚本
"""

import os
import json
import cv2
import numpy as np
from visualize_angles import visualize_keypoints_and_angles, load_keypoints_from_json

def test_visualization():
    """测试可视化功能"""
    # 测试数据目录
    data_dir = 'data/raw_images'
    
    # 选择具有代表性的图片
    test_images = [
        'xray_012.jpg',  # 标准姿势
        'xray_023.jpg',  # 不同对比度
        'xray_044.jpg',  # 不同角度
        'xray_057.jpg'   # 不同位置
    ]
    
    # 创建输出目录
    output_dir = 'output/visualization_test'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始测试可视化功能...")
    
    for img_file in test_images:
        try:
            # 构建文件路径
            img_path = os.path.join(data_dir, img_file)
            json_path = os.path.join(data_dir, img_file.replace('.jpg', '.json'))
            
            if not os.path.exists(json_path):
                print(f"警告: 未找到对应的JSON文件 {json_path}")
                continue
                
            print(f"\n处理图像: {img_file}")
            
            # 加载关键点
            keypoints = load_keypoints_from_json(json_path)
            
            # 生成可视化
            output_path = os.path.join(output_dir, f"vis_{img_file}")
            visualize_keypoints_and_angles(img_path, keypoints, output_path)
            
            print(f"已保存可视化结果到: {output_path}")
            
        except Exception as e:
            print(f"处理 {img_file} 时出错: {str(e)}")
            continue
    
    print("\n可视化测试完成!")

if __name__ == '__main__':
    test_visualization() 