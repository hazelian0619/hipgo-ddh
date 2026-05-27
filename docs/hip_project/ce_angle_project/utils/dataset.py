#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键点数据集处理类
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CEAngleDataset(Dataset):
    """髋关节CE角关键点数据集"""
    
    def __init__(self, img_dir, ann_dir, transform=None, train=True):
        """
        初始化数据集
        
        Args:
            img_dir: 图像目录
            ann_dir: 标注目录
            transform: 图像变换
            train: 是否为训练集
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.train = train
        
        # 获取所有图像文件
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取对应的标注文件
        ann_name = img_name.replace('.jpg', '.json')
        ann_path = os.path.join(self.ann_dir, ann_name)
        
        # 加载标注
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
            
        # 提取关键点
        keypoints = []
        for shape in ann_data['shapes']:
            # 确保按照标签顺序排序关键点
            label = int(shape['label'])
            point = shape['points'][0]
            keypoints.append([label, point[0], point[1]])
        
        # 按标签排序
        keypoints.sort(key=lambda x: x[0])
        
        # 移除标签，只保留坐标
        keypoints = np.array([[kp[1], kp[2]] for kp in keypoints])
        
        # 记录图像尺寸
        height, width = img.shape[:2]
        
        # 应用变换
        if self.transform:
            # 将关键点转换为albumentations格式
            keypoints_albu = [(x, y) for x, y in keypoints]
            transformed = self.transform(image=img, keypoints=keypoints_albu)
            img = transformed['image']
            keypoints = np.array(transformed['keypoints'])
        
        # 如果是PyTorch模型，需要将关键点标准化到[0,1]范围
        norm_keypoints = keypoints.copy()
        norm_keypoints[:, 0] = norm_keypoints[:, 0] / width
        norm_keypoints[:, 1] = norm_keypoints[:, 1] / height
        
        # 创建可见性标志（全部可见）
        visibility = np.ones(len(keypoints))
        
        # 将关键点和可见性拼接
        keypoints_with_vis = np.column_stack((norm_keypoints, visibility))
        
        return {
            'image': img, 
            'keypoints': torch.tensor(keypoints_with_vis, dtype=torch.float32),
            'image_id': img_name,
            'original_size': (height, width)
        }

def get_transforms(train=True, img_size=512):
    """
    获取数据变换
    
    Args:
        train: 是否为训练集
        img_size: 输入图像大小
    
    Returns:
        albumentations变换
    """
    if train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    return transform 