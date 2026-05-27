#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil

from models.cnn_gat_model import CNN_GAT
from utils.dataset import CEAngleDataset
from utils.metrics import calculate_keypoint_accuracy
from configs.default import config

# 使用配置文件中的设置
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)

def keypoint_loss(pred_keypoints, target_keypoints):
    """计算关键点预测损失"""
    # 欧氏距离损失
    mse_loss = nn.MSELoss()(pred_keypoints, target_keypoints)
    # L1损失
    l1_loss = nn.L1Loss()(pred_keypoints, target_keypoints)
    return mse_loss + 0.5 * l1_loss

def train_epoch(model, train_loader, optimizer, device, epoch, writer, accumulation_steps=4):
    """训练一个轮次"""
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        target_keypoints = batch['keypoints'].to(device)
        
        # 前向传播
        pred_keypoints = model(images)
        
        # 计算损失
        loss = keypoint_loss(pred_keypoints, target_keypoints)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计损失和准确率
        epoch_loss += loss.item() * accumulation_steps
        
        accuracy = calculate_keypoint_accuracy(pred_keypoints, target_keypoints)
        epoch_accuracy += accuracy
        
        if (batch_idx + 1) % 5 == 0:
            print(f"轮次 {epoch} [{batch_idx+1}/{len(train_loader)}] "
                  f"损失: {loss.item()*accumulation_steps:.4f} "
                  f"准确率: {accuracy:.4f}")
    
    # 计算平均值
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
    
    return epoch_loss, epoch_accuracy

def validate(model, val_loader, device, epoch, writer):
    """验证模型"""
    model.eval()
    val_loss = 0
    val_accuracy = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            target_keypoints = batch['keypoints'].to(device)
            
            pred_keypoints = model(images)
            
            loss = keypoint_loss(pred_keypoints, target_keypoints)
            
            val_loss += loss.item()
            
            accuracy = calculate_keypoint_accuracy(pred_keypoints, target_keypoints)
            val_accuracy += accuracy
    
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    
    print(f"验证集 - 轮次: {epoch}, 损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")
    
    return val_loss, val_accuracy

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """保存检查点"""
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CE角度检测模型训练')
    parser.add_argument('--epochs', type=int, default=config.training.num_epochs, help='训练轮次数')
    parser.add_argument('--batch-size', type=int, default=config.training.batch_size, help='批处理大小')
    parser.add_argument('--lr', type=float, default=config.training.learning_rate, help='学习率')
    parser.add_argument('--resume', default='', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--data-dir', default=config.data.data_dir, type=str, help='数据集根目录')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 创建TensorBoard日志
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # 创建数据集和数据加载器
    train_dataset = CEAngleDataset(
        data_dir=args.data_dir,
        split='train'
    )
    
    val_dataset = CEAngleDataset(
        data_dir=args.data_dir,
        split='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")
    
    # 创建模型
    model = CNN_GAT(
        backbone=config.model.backbone,
        pretrained=config.model.pretrained,
        num_keypoints=config.model.num_keypoints,
        node_feature_dim=config.model.node_feature_dim,
        gat_layers=config.model.gat_layers,
        gat_heads=config.model.gat_heads,
        gat_feature_dim=config.model.gat_feature_dim,
        dropout=config.model.dropout
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")
    
    # 定义优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=config.training.min_lr
    )
    
    # 创建早停实例
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta
    )
    
    # 恢复训练（如果指定）
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"加载检查点成功 (轮次 {checkpoint['epoch']})")
        else:
            print(f"没有找到检查点 '{args.resume}'")
    
    # 训练循环
    print(f"开始训练，共 {args.epochs} 轮...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 训练一个轮次
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            accumulation_steps=config.training.gradient_accumulation_steps
        )
        
        # 验证模型
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        # 调整学习率
        scheduler.step()
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, 
        filename=os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth.tar'))
        
        # 早停检查
        early_stopping(val_loss, model, os.path.join(config.checkpoint_dir, 'early_stop_model.pth'))
        if early_stopping.early_stop:
            print("早停触发！停止训练。")
            break
        
        epoch_time = time.time() - epoch_start_time
        print(f"轮次 {epoch} 完成，耗时: {epoch_time:.2f}秒")
        
        # 记录学习率
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('LearningRate', current_lr, epoch)
    
    writer.close()
    print("训练完成！")

if __name__ == "__main__":
    main() 