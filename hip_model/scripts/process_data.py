#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
from pathlib import Path
import random

def load_annotations(annotation_file):
    """加载原始标注数据"""
    with open(annotation_file, 'r') as f:
        return json.load(f)

def process_annotations(annotations, output_dir, split='train'):
    """处理标注数据并保存到新的目录结构"""
    # 创建输出目录
    images_dir = Path(output_dir) / split / 'images'
    annotations_dir = Path(output_dir) / split / 'annotations'
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据集模板
    with open(f'{output_dir}/{split}/annotations/dataset.json', 'r') as f:
        dataset = json.load(f)
    
    # 处理每个标注
    for idx, ann in enumerate(annotations):
        image_file = ann['image_file']
        keypoints = ann['keypoints']
        angles = ann['angles']
        
        # 尝试多个可能的图像源路径
        possible_src_paths = [
            f'ce_angle_project/processed_data/images/{image_file}',
            f'ce_angle_project/data/images/{image_file}',
            f'labeled_data/train/images/{image_file}',
            f'labeled_data/val/images/{image_file}'
        ]
        
        # 复制图像文件
        dst_image = images_dir / image_file
        copied = False
        for src_image in possible_src_paths:
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
                copied = True
                break
        
        if not copied:
            print(f'警告：找不到图像文件 {image_file}')
            continue
        
        # 添加图像信息
        image_info = {
            'id': idx + 1,
            'file_name': image_file,
            'width': 2048,  # 默认值，实际应该从图像中读取
            'height': 2048
        }
        dataset['images'].append(image_info)
        
        # 添加标注信息
        annotation = {
            'id': idx + 1,
            'image_id': idx + 1,
            'category_id': 1,
            'keypoints': keypoints,
            'angles': angles
        }
        dataset['annotations'].append(annotation)
    
    # 保存处理后的数据集
    output_file = annotations_dir / 'dataset.json'
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

def main():
    """主函数"""
    # 加载原始标注
    annotations = load_annotations('ce_angle_project/processed_data/train_annotations.json')
    
    # 随机打乱数据
    random.seed(42)
    random.shuffle(annotations)
    
    # 划分训练集和验证集
    split_idx = int(len(annotations) * 0.8)
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]
    
    # 处理训练集
    process_annotations(train_annotations, 'dataset', 'train')
    
    # 处理验证集
    process_annotations(val_annotations, 'dataset', 'val')
    
    print(f'处理完成：')
    print(f'- 训练集：{len(train_annotations)}张图像')
    print(f'- 验证集：{len(val_annotations)}张图像')

if __name__ == '__main__':
    main() 