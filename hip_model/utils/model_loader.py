#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型加载工具
提供便捷的模型加载函数，特别是对最佳模型(0506)的支持
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
sys.path.append(project_root)

# 导入模型
from hip_model.models.cnn_gat_model import CNN_GAT

# 定义最佳模型路径
BEST_MODEL_PATH = os.path.join(project_root, "hip_model/models/model_best_20250506_163007.pth")

def load_best_model_0506(device=None, eval_mode=True):
    """
    加载0506最佳模型(model_best_20250506_163007.pth)
    
    参数:
        device: 设备(cuda, cpu, mps)，如果为None则自动检测
        eval_mode: 是否设置为评估模式
    
    返回:
        loaded_model: 加载了最佳模型权重的CNN_GAT模型实例
    """
    # 检查设备
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"找不到最佳模型文件: {BEST_MODEL_PATH}")
    
    # 添加安全全局变量（PyTorch 2.6兼容性）
    try:
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
    except:
        print("无法添加安全全局变量，可能会影响模型加载")
    
    try:
        # 加载模型权重
        print(f"加载0506最佳模型: {BEST_MODEL_PATH}")
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print("已从完整检查点加载模型权重")
        else:
            model_state = checkpoint
            print("已直接加载模型权重")
        
        # 创建模型实例
        model = CNN_GAT(
            feature_dim=256,
            gat_hidden=128,
            gat_output=64,
            edge_features_dim=32,
            num_keypoints=9,
            num_angles=6,
            num_gat_layers=2,
            num_heads=8,
            dropout=0.1,
            pretrained=True
        ).to(device)
        
        # 加载权重
        model.load_state_dict(model_state)
        
        # 设置评估模式（如果需要）
        if eval_mode:
            model.eval()
        
        print("0506最佳模型加载完成")
        return model
        
    except Exception as e:
        raise RuntimeError(f"0506最佳模型加载失败: {str(e)}")

# 简便起见的别名函数
def load_best_model(*args, **kwargs):
    """加载最佳模型的快捷方式"""
    return load_best_model_0506(*args, **kwargs) 