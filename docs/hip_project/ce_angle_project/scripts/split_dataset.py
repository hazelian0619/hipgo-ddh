#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将原始数据集分割为训练集和验证集
"""

import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    """
    将原始数据集分割为训练集和验证集
    
    Args:
        input_dir: 输入目录(包含图像和标注文件)
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    random.seed(seed)
    
    # 创建输出目录
    train_img_dir = os.path.join(output_dir, 'labeled_data', 'train', 'images')
    train_ann_dir = os.path.join(output_dir, 'labeled_data', 'train', 'annotations')
    val_img_dir = os.path.join(output_dir, 'labeled_data', 'val', 'images')
    val_ann_dir = os.path.join(output_dir, 'labeled_data', 'val', 'annotations')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_ann_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_ann_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)
    
    # 计算训练集大小
    train_size = int(len(image_files) * train_ratio)
    
    # 分割数据集
    train_images = image_files[:train_size]
    val_images = image_files[train_size:]
    
    # 复制训练集文件
    for img_file in train_images:
        # 复制图像文件
        src_img = os.path.join(input_dir, img_file)
        dst_img = os.path.join(train_img_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # 复制标注文件
        ann_file = img_file.replace('.jpg', '.json')
        src_ann = os.path.join(input_dir, ann_file)
        dst_ann = os.path.join(train_ann_dir, ann_file)
        if os.path.exists(src_ann):
            shutil.copy2(src_ann, dst_ann)
    
    # 复制验证集文件
    for img_file in val_images:
        # 复制图像文件
        src_img = os.path.join(input_dir, img_file)
        dst_img = os.path.join(val_img_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # 复制标注文件
        ann_file = img_file.replace('.jpg', '.json')
        src_ann = os.path.join(input_dir, ann_file)
        dst_ann = os.path.join(val_ann_dir, ann_file)
        if os.path.exists(src_ann):
            shutil.copy2(src_ann, dst_ann)
    
    print(f"已分割数据集: {len(train_images)} 个训练样本，{len(val_images)} 个验证样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将数据集分割为训练集和验证集')
    parser.add_argument('--input_dir', type=str, default='data/raw_images', 
                        help='输入目录，包含图像和标注文件')
    parser.add_argument('--output_dir', type=str, default='hip_project/ce_angle_project', 
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    args = parser.parse_args()
    split_dataset(args.input_dir, args.output_dir, args.train_ratio, args.seed) 