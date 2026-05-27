#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练CNN-GAT模型用于髋关节关键点检测
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
from datetime import datetime

from models.cnn_gat_model import CNN_GAT
from dataset import HipKeypointDataset, get_transforms

# 设置设备：CUDA优先，MPS暂跳过（grid_sample在MPS上不兼容）
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"使用设备: {device}")

def train_model(args):
    """
    训练模型
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建数据加载器
    train_dataset = HipKeypointDataset(
        img_dir=args.data_dir,
        transform=get_transforms(train=True, img_size=args.img_size),
        train=True,
        split_ratio=args.split_ratio
    )
    
    val_dataset = HipKeypointDataset(
        img_dir=args.data_dir,
        transform=get_transforms(train=False, img_size=args.img_size),
        train=False,
        split_ratio=args.split_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = CNN_GAT(
        feature_dim=args.feature_dim,
        gat_hidden=args.gat_hidden,
        gat_output=args.gat_output,
        edge_features_dim=args.edge_features_dim,
        num_keypoints=args.num_keypoints,
        num_angles=args.num_angles,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pretrained=args.pretrained
    ).to(device)
    
    # 定义损失函数
    # 按点难度加权：基于规范化坐标std，std越大（越难定位）权重越低
    # 数据来源：scripts/eval_per_point.py 统计的各点规范化std
    # 点5（耻骨联合）因骨盆个体差异大（长/短骨盆、男/女差异）天然难以精确定位，给予更低权重
    _point_stds = [0.017, 0.017, 0.051, 0.055, 0.073, 0.029, 0.041, 0.028, 0.048]
    _raw_weights = [1.0 / (1.0 + s) for s in _point_stds]
    _sum = sum(_raw_weights)
    _point_weights = [w / _sum * 9 for w in _raw_weights]  # 归一化，均值=1
    keypoint_weights = torch.tensor(_point_weights, dtype=torch.float32).to(device)  # [9]

    def weighted_keypoint_loss(pred, target):
        """逐点加权MSE：难以定位的点贡献更低的loss"""
        # pred, target: [B, 9, 2]
        per_point_mse = ((pred - target) ** 2).mean(dim=2)  # [B, 9]
        weighted = per_point_mse * keypoint_weights.unsqueeze(0)  # [B, 9]
        return weighted.mean()
    
    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 从预训练模型继续训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点: {args.resume}")
            try:
                # 添加argparse.Namespace到安全全局变量列表，为PyTorch 2.6兼容性
                torch.serialization.add_safe_globals([argparse.Namespace])
                checkpoint = torch.load(args.resume, weights_only=False)
                
                # 确保安全加载
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("模型权重已加载")
                    
                if 'optimizer_state_dict' in checkpoint and not args.new_optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("优化器状态已加载")
                    
                if 'epoch' in checkpoint and not args.new_optimizer:
                    start_epoch = checkpoint['epoch']
                    print(f"从第 {start_epoch} 轮继续训练")
                    
                if 'val_loss' in checkpoint and not args.new_optimizer:
                    best_val_loss = checkpoint['val_loss']
                    print(f"最佳验证损失: {best_val_loss:.6f}")
                    
                print("检查点加载完成!")
            except Exception as e:
                print(f"加载检查点时出错: {str(e)}")
                print("尝试仅加载模型权重...")
                
                # 尝试仅加载模型权重
                try:
                    checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
                    model.load_state_dict(checkpoint)
                    print("模型权重已加载")
                except Exception as e:
                    print(f"加载模型权重失败: {str(e)}")
        else:
            print(f"未找到检查点: {args.resume}")
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 创建保存模型的目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练模型
    early_stop_count = 0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'keypoint_loss': []
    }
    
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        keypoint_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # 获取数据
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            
            # 前向传播
            keypoint_positions = keypoints[:, :, :2]  # 只取坐标，不要可见性
            
            # 获取预测结果
            predictions = model(images)
            pred_keypoints = predictions['keypoints']
            
            # 计算关键点损失
            keypoint_loss = weighted_keypoint_loss(pred_keypoints, keypoint_positions)

            # 不预测角度，直接使用关键点损失作为总损失
            loss = keypoint_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item()
            keypoint_loss_sum += keypoint_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'kp_loss': f"{keypoint_loss.item():.4f}"
            })
        
        # 计算平均损失
        train_loss /= len(train_loader)
        keypoint_loss_avg = keypoint_loss_sum / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据
                images = batch['image'].to(device)
                keypoints = batch['keypoints'].to(device)
                
                # 前向传播
                keypoint_positions = keypoints[:, :, :2]  # 只取坐标，不要可见性
                
                # 获取预测结果
                predictions = model(images)
                pred_keypoints = predictions['keypoints']
                
                # 计算关键点损失
                keypoint_loss = weighted_keypoint_loss(pred_keypoints, keypoint_positions)

                # 不预测角度，直接使用关键点损失作为总损失
                loss = keypoint_loss
                
                # 更新统计信息
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"KP Loss: {keypoint_loss_avg:.4f}")
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['keypoint_loss'].append(keypoint_loss_avg)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            
            # 保存模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(args.output_dir, f"model_best_{timestamp}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, save_path)
            
            print(f"保存最佳模型到 {save_path}")
        else:
            early_stop_count += 1
            
        # 早停
        if early_stop_count >= args.patience:
            print(f"早停: {args.patience} 轮验证损失没有改善")
            break
    
    # 绘制训练历史
    plot_history(history, args.output_dir)
    
    print("训练完成!")

def plot_history(history, output_dir):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        output_dir: 输出目录
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 绘制关键点损失
    plt.subplot(2, 1, 2)
    plt.plot(history['keypoint_loss'], label='Keypoint Loss')
    plt.title('Component Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"training_history_{timestamp}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"保存训练历史到 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练CNN-GAT模型用于髋关节关键点检测")
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='data/raw_images', help='包含图像和标注的目录')
    parser.add_argument('--output-dir', type=str, default='outputs', help='保存模型和日志的目录')
    parser.add_argument('--img-size', type=int, default=512, help='输入图像大小')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='训练集比例')
    
    # 模型参数
    parser.add_argument('--feature-dim', type=int, default=256, help='特征维度')
    parser.add_argument('--gat-hidden', type=int, default=128, help='GAT隐藏层维度')
    parser.add_argument('--gat-output', type=int, default=64, help='GAT输出层维度')
    parser.add_argument('--edge-features-dim', type=int, default=32, help='边特征维度')
    parser.add_argument('--num-keypoints', type=int, default=9, help='关键点数量')
    parser.add_argument('--num-angles', type=int, default=6, help='角度数量')
    parser.add_argument('--num-gat-layers', type=int, default=2, help='GAT层数量')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比例')
    parser.add_argument('--pretrained', action='store_true', help='是否使用预训练的骨干网络')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--keypoint-weight', type=float, default=1.0, help='关键点损失权重')
    parser.add_argument('--angle-weight', type=float, default=0.1, help='角度损失权重')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 继续训练参数
    parser.add_argument('--resume', type=str, default='', help='加载检查点路径')
    parser.add_argument('--new-optimizer', action='store_true', help='使用新的优化器状态（不加载检查点中的优化器状态）')
    
    args = parser.parse_args()
    train_model(args) 