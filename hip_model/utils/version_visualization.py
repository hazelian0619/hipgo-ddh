# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class VersionProgressVisualizer:
    def __init__(self, save_dir='figures'):
        self.save_dir = save_dir
        plt.style.use('seaborn')
        
    def plot_training_progress(self, version_data):
        """绘制训练进度对比图"""
        plt.figure(figsize=(12, 6))
        
        versions = list(version_data.keys())
        train_losses = [v['train_loss'][1] for v in version_data.values()]
        val_losses = [v['val_loss'] for v in version_data.values()]
        
        x = np.arange(len(versions))
        width = 0.35
        
        plt.bar(x - width/2, train_losses, width, label='Train Loss', color='skyblue')
        plt.bar(x + width/2, val_losses, width, label='Val Loss', color='lightcoral')
        
        plt.title('Training Progress Across Versions')
        plt.xlabel('Version')
        plt.ylabel('Loss')
        plt.xticks(x, versions)
        plt.legend()
        plt.yscale('log')
        
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'))
        plt.close()
        
    def plot_error_trends(self, version_data):
        """绘制误差趋势图"""
        plt.figure(figsize=(12, 6))
        
        versions = list(version_data.keys())
        keypoint_errors = [v['keypoint_error'] for v in version_data.values()]
        angle_errors = [v['angle_error'] for v in version_data.values()]
        
        plt.plot(versions, keypoint_errors, 'o-', label='Keypoint Error (px)', color='blue')
        plt.plot(versions, angle_errors, 's-', label='Angle Error (°)', color='red')
        
        plt.title('Error Metrics Trends')
        plt.xlabel('Version')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, 'error_trends.png'))
        plt.close()
        
    def plot_speed_improvement(self, version_data):
        """绘制速度提升图"""
        plt.figure(figsize=(10, 6))
        
        versions = list(version_data.keys())
        speeds = [v['speed'] for v in version_data.values()]
        
        plt.plot(versions, speeds, 'o-', color='green', linewidth=2)
        plt.fill_between(versions, [0]*len(versions), speeds, alpha=0.2, color='green')
        
        plt.title('Training Speed Improvement')
        plt.xlabel('Version')
        plt.ylabel('Iterations per Second')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, 'speed_improvement.png'))
        plt.close()

# 使用历史数据
version_data = {
    'V1': {
        'train_loss': [278940, 23101],
        'val_loss': None,
        'keypoint_error': 1500,
        'angle_error': 50,
        'speed': 1.5,
        'features': '基础ResNet18，MSE损失，简单数据加载'
    },
    'V2': {
        'train_loss': [23101, 0.8179],
        'val_loss': None,
        'keypoint_error': 300,
        'angle_error': 10,
        'speed': 2.8,
        'features': '添加数据归一化，基础数据增强，改进数据加载'
    },
    'V3': {
        'train_loss': [0.2093, 0.1983],
        'val_loss': 0.0015,
        'keypoint_error': 131.35,
        'angle_error': None,
        'speed': 4.0,
        'features': '添加FPN结构，多尺度特征，双分支预测'
    },
    'V4': {
        'train_loss': [0.0587, 0.0161],
        'val_loss': 0.0015,
        'keypoint_error': 76.27,
        'angle_error': 50.93,
        'speed': 7.5,
        'features': 'CBAM注意力，动态权重，改进预测头'
    },
    'V5': {
        'train_loss': [0.0474, 0.0371],
        'val_loss': 0.0769,
        'keypoint_error': 27.17,
        'angle_error': 9.80,
        'speed': 8.2,
        'features': '组合损失函数，梯度累积，早停机制'
    },
    'V6': {
        'train_loss': [0.0385, 0.0054],
        'val_loss': 0.0106,
        'keypoint_error': 48.02,
        'angle_error': 1.49,
        'speed': 12.0,
        'features': '学习率调度，损失权重平衡，验证集优化'
    },
    'V7': {
        'train_loss': [0.0094, 0.0037],
        'val_loss': 0.0085,
        'keypoint_error': 51.59,
        'angle_error': 1.19,
        'speed': 14.0,
        'features': 'GPU训练支持，检查点保存，损失平滑'
    }
}

# 生成可视化
visualizer = VersionProgressVisualizer()
visualizer.plot_training_progress(version_data)
visualizer.plot_error_trends(version_data)
visualizer.plot_speed_improvement(version_data)