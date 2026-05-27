#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将自动标注生成的JSON文件转换为原始格式
自动标注格式: {'image_name': '...', 'keypoints': [[x1, y1], [x2, y2], ...]}
原始格式: {'shapes': [{'label': '1', 'points': [[x1, y1]], ...}]}

关键点定义：
1. 左侧股骨头中心点(left_femoral_head_center)
2. 右侧股骨头中心点(right_femoral_head_center)
3. 左侧髋臼外缘点(left_acetabular_edge)
4. 右侧髋臼外缘点(right_acetabular_edge)
5. 耻骨联合点(pubic_symphysis)
6. 左侧髋臼荷重面内侧点(left_sourcil_medial)
7. 左侧髋臼荷重面外侧点(left_sourcil_lateral)
8. 右侧髋臼荷重面内侧点(right_sourcil_medial)
9. 右侧髋臼荷重面外侧点(right_sourcil_lateral)
"""

import os
import json
import glob
import argparse
from tqdm import tqdm

def convert_annotation(src_path, dst_path):
    """转换单个标注文件"""
    # 读取自动标注文件
    with open(src_path, 'r') as f:
        auto_data = json.load(f)
    
    # 提取图像名称和关键点
    image_name = auto_data.get('image_name', os.path.basename(src_path).replace('.json', '.jpg'))
    keypoints = auto_data.get('keypoints', [])
    
    # 创建原始格式数据
    original_data = {
        'version': '5.6.1',
        'flags': {},
        'shapes': [],
        'imagePath': image_name,
        'imageData': None
    }
    
    # 关键点标签与解剖学定义
    keypoint_definitions = {
        1: "左侧股骨头中心点",
        2: "右侧股骨头中心点",
        3: "左侧髋臼外缘点",
        4: "右侧髋臼外缘点",
        5: "耻骨联合点",
        6: "左侧髋臼荷重面内侧点",
        7: "左侧髋臼荷重面外侧点",
        8: "右侧髋臼荷重面内侧点",
        9: "右侧髋臼荷重面外侧点"
    }
    
    # 转换关键点格式
    for i, point in enumerate(keypoints):
        shape = {
            'label': str(i+1),  # 标签从1开始
            'points': [[point[0], point[1]]],  # 二维数组形式
            'group_id': None,
            'description': keypoint_definitions.get(i+1, ""),
            'shape_type': 'point',
            'flags': {}
        }
        original_data['shapes'].append(shape)
    
    # 保存转换后的文件
    with open(dst_path, 'w') as f:
        json.dump(original_data, f, indent=2)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='将自动标注格式转换为原始格式')
    parser.add_argument('--src-dir', type=str, required=True, help='自动标注目录')
    parser.add_argument('--dst-dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.dst_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(args.src_dir, '*.json'))
    print(f'找到 {len(json_files)} 个标注文件')
    
    # 转换所有文件
    success_count = 0
    for json_file in tqdm(json_files):
        base_name = os.path.basename(json_file)
        dst_file = os.path.join(args.dst_dir, base_name)
        
        if convert_annotation(json_file, dst_file):
            success_count += 1
    
    print(f'成功转换 {success_count}/{len(json_files)} 个标注文件')

if __name__ == '__main__':
    main() 