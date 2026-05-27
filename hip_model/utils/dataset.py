import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.default import config
from utils.transforms import get_train_transforms, get_val_transforms

class CEAngleDataset(Dataset):
    """
    CE角度关键点检测数据集

    Args:
        data_dir: 数据根目录
        split: 训练/验证划分
        transform: 数据增强
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 默认使用预设的transforms
        if self.transform is None:
            if split == 'train':
                self.transform = get_train_transforms()
            else:
                self.transform = get_val_transforms()
        
        # 构建图像和标注路径
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.ann_dir = os.path.join(data_dir, split, 'annotations')
        
        # 获取所有图像文件列表
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # 核对每个图像是否有对应的标注文件
        self.valid_imgs = []
        self.valid_anns = []
        
        for img_file in self.img_files:
            # 构建对应的标注文件名
            base_name = os.path.splitext(img_file)[0]
            ann_file = f"{base_name}.json"
            ann_path = os.path.join(self.ann_dir, ann_file)
        
            # 如果标注文件存在，添加到有效列表
            if os.path.exists(ann_path):
                self.valid_imgs.append(os.path.join(self.img_dir, img_file))
                self.valid_anns.append(ann_path)
        
        print(f"Loaded {len(self.valid_imgs)} valid samples for split '{split}'")
        
    def __len__(self):
        return len(self.valid_imgs)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.valid_imgs[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # 读取标注
        ann_path = self.valid_anns[idx]
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann_data = json.load(f)
        
        # 提取9个关键点坐标，格式为 [[x1, y1], [x2, y2], ..., [x9, y9]]
        keypoints = np.array(ann_data['keypoints'])
        
        # 确保有9个关键点
        assert keypoints.shape == (config.model.num_keypoints, 2), \
            f"Expected {config.model.num_keypoints} keypoints, got {keypoints.shape[0]}"
        
        # 提取图像大小
        h, w = img.shape[:2]
        
        # 归一化关键点坐标到[0,1]范围
        keypoints = keypoints.astype(np.float32)
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h
        
        # 应用数据增强
        if self.transform:
            # 转回像素坐标用于albumentations
            kpts_pixel = keypoints.copy()
            kpts_pixel[:, 0] *= w
            kpts_pixel[:, 1] *= h
            
            # 转换为albumentations需要的关键点格式
            keypoints_list = []
            for i in range(len(kpts_pixel)):
                keypoints_list.append((kpts_pixel[i][0], kpts_pixel[i][1], 1))  # x, y, visibility
            
            transformed = self.transform(
                image=img,
                keypoints=keypoints_list
            )
            
            img = transformed['image']  # 已经是tensor了
            
            if 'keypoints' in transformed:
                transformed_keypoints = np.array(transformed['keypoints'])
                if len(transformed_keypoints) > 0:
                    # 提取x, y坐标，忽略visibility
                    transformed_keypoints = transformed_keypoints[:, :2]
                    
                    # 图像尺寸可能已经改变，需要获取transform后的尺寸
                    if isinstance(img, torch.Tensor):
                        new_h, new_w = img.shape[1], img.shape[2]
                    else:
                        new_h, new_w = img.shape[:2]
                    
                    # 归一化回[0,1]范围
                    transformed_keypoints[:, 0] /= new_w
                    transformed_keypoints[:, 1] /= new_h
                    
                    keypoints = transformed_keypoints
        
        # 确保关键点是float32类型的tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        # 如果图像不是tensor，转换为tensor
        if not isinstance(img, torch.Tensor):
            transform = ToTensorV2()
            img = transform(image=img)['image']
        
        return {
            'image': img,
            'keypoints': keypoints,
            'image_path': img_path
        }

    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        """
        数据批次整理函数
        Args:
            batch: 批次数据列表
        Returns:
            整理后的批次数据字典
        """
        images = torch.stack([item['image'] for item in batch])
        keypoints = torch.stack([item['keypoints'] for item in batch])
        img_files = [item['image_path'] for item in batch]
        
        return {
            'image': images,
            'keypoints': keypoints,
            'img_file': img_files
        }
