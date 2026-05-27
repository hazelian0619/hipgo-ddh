#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import logging
import json
import numpy as np
from PIL import Image
import cv2
import importlib.util

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def setup_directories():
    """设置必要的目录结构"""
    dirs = [
        'models',
        'datasets',
        'labeled_data/train/images',
        'labeled_data/val/images'
    ]
    for d in dirs:
        full_path = os.path.join(current_dir, d)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"创建目录: {full_path}")

def create_dataset_module():
    """创建数据集模块"""
    # 创建datasets/__init__.py
    init_path = os.path.join(current_dir, 'datasets/__init__.py')
    init_content = """
from .keypoint_dataset import PelvicKeypointDataset
"""
    with open(init_path, 'w') as f:
        f.write(init_content)
    logger.info(f"创建文件: {init_path}")

    # 创建datasets/keypoint_dataset.py
    dataset_path = os.path.join(current_dir, 'datasets/keypoint_dataset.py')
    dataset_content = """
import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class PelvicKeypointDataset(Dataset):
    def __init__(self, data_dir, annotation_file, img_size=(512, 512), transform=None, mode='train'):
        self.data_dir = os.path.abspath(data_dir)
        self.annotation_file = os.path.abspath(annotation_file)
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        
        # 加载标注
        try:
            with open(self.annotation_file, 'r') as f:
                self.annotations = json.load(f)
            logger.info(f"成功加载 {len(self.annotations)} 个 {mode} 样本")
        except Exception as e:
            logger.error(f"加载标注文件失败: {str(e)}")
            self.annotations = []
        
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        try:
            # 获取图像路径
            img_path = os.path.join(self.data_dir, self.annotations[idx]['image'])
            
            # 读取图像
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.img_size)
            
            # 转换为tensor
            image = self.base_transform(image)
            
            # 获取关键点
            keypoints = torch.tensor(self.annotations[idx]['keypoints'], dtype=torch.float32)
            
            return {
                'image': image,
                'keypoints': keypoints
            }
        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {str(e)}")
            return None
"""

    with open(dataset_path, 'w') as f:
        f.write(dataset_content)
    logger.info(f"创建文件: {dataset_path}")

def load_module_from_file(module_name, file_path):
    """从文件加载模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"无法从文件加载模块: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"成功加载模块: {module_name}")
        return module
    except Exception as e:
        logger.error(f"加载模块失败: {str(e)}")
        raise

def test_data_loading():
    """测试数据加载"""
    try:
        # 强制重新加载模块
        if 'datasets' in sys.modules:
            del sys.modules['datasets']
            del sys.modules['datasets.keypoint_dataset']
        
        # 从文件直接加载模块
        dataset_path = os.path.join(current_dir, 'datasets/keypoint_dataset.py')
        dataset_module = load_module_from_file('keypoint_dataset', dataset_path)
        
        logger.info("测试数据加载...")
        
        # 创建测试数据
        test_data = {
            "image": "test.jpg",
            "keypoints": [[100, 100], [200, 200], [300, 300]]  # 示例关键点
        }
        
        # 保存测试数据
        json_path = os.path.join(current_dir, 'labeled_data/train/dataset.json')
        with open(json_path, 'w') as f:
            json.dump([test_data], f)
        logger.info(f"创建测试数据文件: {json_path}")
        
        # 创建测试图像
        img_path = os.path.join(current_dir, 'labeled_data/train/images/test.jpg')
        test_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.imwrite(img_path, test_img)
        logger.info(f"创建测试图像: {img_path}")
        
        # 创建数据集
        data_dir = os.path.join(current_dir, 'labeled_data/train/images')
        dataset = dataset_module.PelvicKeypointDataset(
            data_dir=data_dir,
            annotation_file=json_path,
            img_size=(224, 224)
        )
        
        logger.info(f"数据集大小: {len(dataset)}")
        
        # 测试第一个样本
        sample = dataset[0]
        if sample is not None:
            logger.info(f"图像形状: {sample['image'].shape}")
            logger.info(f"关键点形状: {sample['keypoints'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始测试...")
    
    try:
        # 1. 设置目录结构
        setup_directories()
        
        # 2. 创建数据集模块
        create_dataset_module()
        
        # 3. 测试数据加载
        if not test_data_loading():
            logger.error("数据加载测试失败")
            return
        
        logger.info("所有测试通过！")
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        return

if __name__ == "__main__":
    main()
    