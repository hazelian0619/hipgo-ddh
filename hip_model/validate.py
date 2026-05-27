#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证CNN-GAT模型性能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from models.cnn_gat_model import CNN_GAT
from dataset import HipKeypointDataset, get_transforms
from torch.utils.data import DataLoader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def calculate_pck(pred_keypoints, true_keypoints, threshold=0.1):
    """
    计算PCK (Percentage of Correct Keypoints)
    
    Args:
        pred_keypoints: 预测关键点 [batch_size, num_keypoints, 2]
        true_keypoints: 真实关键点 [batch_size, num_keypoints, 2]
        threshold: 阈值，通常是图像对角线长度的百分比
        
    Returns:
        pck: PCK分数
    """
    batch_size, num_keypoints, _ = pred_keypoints.shape
    correct = 0
    total = batch_size * num_keypoints
    
    for b in range(batch_size):
        # 计算图像对角线长度
        # 假设图像尺寸是1x1（标准化后的坐标）
        diagonal = np.sqrt(1**2 + 1**2)
        
        for k in range(num_keypoints):
            # 计算预测和真实关键点之间的欧几里得距离
            dist = np.sqrt(
                (pred_keypoints[b, k, 0] - true_keypoints[b, k, 0])**2 + 
                (pred_keypoints[b, k, 1] - true_keypoints[b, k, 1])**2
            )
            
            # 如果距离小于阈值，则认为是正确的
            if dist < threshold * diagonal:
                correct += 1
    
    return correct / total

def visualize_keypoints(image, true_keypoints, pred_keypoints, image_id, save_path):
    """
    可视化关键点预测结果
    
    Args:
        image: 输入图像 [H, W, C]
        true_keypoints: 真实关键点 [num_keypoints, 2]
        pred_keypoints: 预测关键点 [num_keypoints, 2]
        image_id: 图像ID
        save_path: 保存路径
    """
    # 如果图像是PyTorch张量，转换为NumPy数组
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # 绘制真实关键点
    for i, kp in enumerate(true_keypoints):
        x, y = kp
        plt.plot(x, y, 'go', markersize=8)
        plt.text(x, y, str(i+1), color='white', fontsize=12)
    
    # 绘制预测关键点
    for i, kp in enumerate(pred_keypoints):
        x, y = kp
        plt.plot(x, y, 'ro', markersize=8)
        plt.text(x, y, str(i+1), color='yellow', fontsize=12)
    
    plt.title(f"Image ID: {image_id}")
    plt.tight_layout()
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

def validate_model(args):
    """
    验证模型性能
    
    Args:
        args: 命令行参数
    """
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    
    # 创建模型
    model = CNN_GAT(
        feature_dim=model_args.feature_dim,
        gat_hidden=model_args.gat_hidden,
        gat_output=model_args.gat_output,
        edge_features_dim=model_args.edge_features_dim,
        num_keypoints=model_args.num_keypoints,
        num_angles=model_args.num_angles,
        num_gat_layers=model_args.num_gat_layers,
        num_heads=model_args.num_heads,
        dropout=model_args.dropout,
        pretrained=False
    ).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"加载模型: {args.model_path}")
    print(f"模型参数: {model_args}")
    
    # 创建数据加载器
    val_dataset = HipKeypointDataset(
        img_dir=args.data_dir,
        transform=get_transforms(train=False, img_size=args.img_size),
        train=False,
        split_ratio=args.split_ratio
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建保存可视化结果的目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化指标
    keypoint_errors = []
    pck_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="验证")):
            # 获取数据
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            image_ids = batch['image_id']
            original_sizes = batch['original_size']
            
            # 前向传播
            keypoint_positions = keypoints[:, :, :2]  # 只取坐标，不要可见性
            
            # 获取预测结果
            predictions = model(images)
            pred_keypoints = predictions['keypoints']
            pred_angles = predictions['angles']
            
            # 计算关键点误差
            keypoint_error = torch.mean(torch.sqrt(torch.sum((pred_keypoints - keypoint_positions)**2, dim=2))).item()
            keypoint_errors.append(keypoint_error)
            
            # 计算PCK
            pck = calculate_pck(
                pred_keypoints.cpu().numpy(),
                keypoint_positions.cpu().numpy(),
                threshold=args.pck_threshold
            )
            pck_scores.append(pck)
            
            # 可视化结果
            if batch_idx < args.num_visualizations:
                for i in range(min(images.shape[0], args.num_visualizations - batch_idx * images.shape[0])):
                    # 获取原始图像尺寸
                    h, w = original_sizes[i]
                    h, w = h.item(), w.item()
                    
                    # 转换预测坐标到像素坐标
                    pred_keypoints_vis = pred_keypoints[i].cpu().numpy().copy() * np.array([w, h])
                    true_keypoints_vis = keypoint_positions[i].cpu().numpy().copy() * np.array([w, h])
                    
                    # 可视化
                    save_path = os.path.join(args.output_dir, f"vis_{batch_idx * args.batch_size + i}_{image_ids[i]}")
                    visualize_keypoints(
                        images[i].cpu(),
                        true_keypoints_vis,
                        pred_keypoints_vis,
                        image_ids[i],
                        save_path
                    )
    
    # 计算平均指标
    avg_keypoint_error = np.mean(keypoint_errors)
    avg_pck = np.mean(pck_scores)
    
    print(f"平均关键点误差: {avg_keypoint_error:.4f}")
    print(f"PCK@{args.pck_threshold}: {avg_pck:.4f}")
    
    # 保存结果
    results = {
        'keypoint_error': avg_keypoint_error,
        'pck': avg_pck
    }
    
    print("验证完成!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证CNN-GAT模型性能")
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='data/raw_images', help='包含图像和标注的目录')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations', help='保存可视化结果的目录')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--img-size', type=int, default=512, help='输入图像大小')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='训练集比例')
    
    # 验证参数
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作线程数')
    parser.add_argument('--pck-threshold', type=float, default=0.1, help='PCK阈值')
    parser.add_argument('--num-visualizations', type=int, default=10, help='可视化样本数量')
    
    args = parser.parse_args()
    validate_model(args) 