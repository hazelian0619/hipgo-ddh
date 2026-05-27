#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载髋关节角度9点标注数据集
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# 水平翻转时需要交换的关键点对（0-indexed）
# 点1(0)↔点2(1)，点3(2)↔点4(3)，点6(5)↔点8(7)，点7(6)↔点9(8)，点5(4)不变
FLIP_PAIRS = [(0, 1), (2, 3), (5, 7), (6, 8)]


def _swap_keypoints_after_flip(keypoints: np.ndarray) -> np.ndarray:
    """
    水平翻转后交换左右侧关键点的语义顺序。

    Args:
        keypoints: shape [9, 2]，归一化坐标

    Returns:
        重排后的 keypoints，shape [9, 2]
    """
    swapped = keypoints.copy()
    for i, j in FLIP_PAIRS:
        swapped[i], swapped[j] = keypoints[j].copy(), keypoints[i].copy()
    return swapped


class HipKeypointDataset(Dataset):
    """髋关节9点关键点数据集"""

    def __init__(self, img_dir, transform=None, train=True, split_ratio=0.8, seed=42):
        """
        Args:
            img_dir: 包含图像和JSON标注的目录
            transform: 图像变换（不含HorizontalFlip，翻转在__getitem__里单独处理）
            train: 是否为训练集
            split_ratio: 训练集比例
            seed: 随机种子
        """
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        np.random.seed(seed)
        indices = np.arange(len(self.img_files))
        np.random.shuffle(indices)

        train_size = int(len(indices) * split_ratio)
        self.indices = indices[:train_size] if train else indices[train_size:]
        self.img_files = [self.img_files[i] for i in self.indices]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_path = os.path.join(self.img_dir, img_name.replace('.jpg', '.json'))
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        # 提取关键点，按标签1~9排序
        keypoints_raw = []
        for shape in ann_data['shapes']:
            label = int(shape['label'])
            point = shape['points'][0]
            keypoints_raw.append([label, point[0], point[1]])
        keypoints_raw.sort(key=lambda x: x[0])
        keypoints = np.array([[kp[1], kp[2]] for kp in keypoints_raw], dtype=np.float32)  # [9,2] 像素坐标

        # 记录原始图尺寸（用于可视化时反算像素坐标）
        orig_height, orig_width = img.shape[:2]

        # --- 训练时手动处理水平翻转，确保关键点语义正确互换 ---
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()  # 水平翻转图像
            keypoints[:, 0] = orig_width - keypoints[:, 0]  # x坐标镜像
            keypoints = _swap_keypoints_after_flip(keypoints)  # 语义互换

        # 应用其余变换（不含HorizontalFlip）
        if self.transform:
            keypoints_albu = [(x, y) for x, y in keypoints]
            transformed = self.transform(image=img, keypoints=keypoints_albu)
            img = transformed['image']
            keypoints = np.array(transformed['keypoints'], dtype=np.float32)

        # 归一化坐标到 [0, 1]
        # 用变换后的图像尺寸做归一化（而非原始尺寸），保证和transform后的坐标空间一致
        new_h, new_w = img.shape[-2:] if isinstance(img, torch.Tensor) else img.shape[:2]
        norm_keypoints = keypoints.copy()
        norm_keypoints[:, 0] /= new_w
        norm_keypoints[:, 1] /= new_h

        visibility = np.ones(len(keypoints), dtype=np.float32)
        keypoints_with_vis = np.column_stack((norm_keypoints, visibility))

        return {
            'image': img,
            'keypoints': torch.tensor(keypoints_with_vis, dtype=torch.float32),
            'image_id': img_name,
            'original_size': (orig_height, orig_width),
            'transformed_size': (new_h, new_w),
        }

    def visualize(self, idx):
        """可视化数据集中的一个样本"""
        sample = self[idx]
        img = sample['image']
        keypoints = sample['keypoints']

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip(img * std + mean, 0, 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        height, width = sample['original_size']
        for i, kp in enumerate(keypoints):
            x = kp[0].item() * width
            y = kp[1].item() * height
            plt.plot(x, y, 'ro', markersize=8)
            plt.text(x, y, str(i + 1), color='white')

        plt.title(f"Image ID: {sample['image_id']}")
        plt.show()


def get_transforms(train=True, img_size=512):
    """
    获取数据变换。

    Resize策略：LongestMaxSize + PadIfNeeded（等比缩放后补黑边）
    原因：数据来源为社交媒体，宽高比从0.45到2.22，直接Resize会变形骨盆比例。
    等比padding保持骨骼形状，关键点坐标由albumentations自动同步修正。

    HorizontalFlip 已移至 Dataset.__getitem__ 中单独处理（保证关键点语义正确互换），
    这里不包含HorizontalFlip。

    Args:
        train: 是否为训练集
        img_size: 输入图像大小

    Returns:
        albumentations 变换
    """
    if train:
        transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=0,
            ),
            # HorizontalFlip 不在此处——已在 Dataset.__getitem__ 里正确处理
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Rotate(limit=15, border_mode=0, fill=0, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=0,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    return transform
