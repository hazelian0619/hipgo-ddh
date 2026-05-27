# -*- coding: utf-8 -*-
import numpy as np
import torch

def compute_angle(v1, v2):
    """计算两个向量间的夹角"""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norms
    return np.degrees(np.arccos(cos_angle))

def calculate_bilateral_ce_angles(points):
    """计算双侧CE角度"""
    # 垂直参考向量（向上为负）
    vertical = np.array([0, -1])
    
    # 计算左侧CE角度
    left_vector = points[2] - points[0]  # 左侧外缘点 - 左侧中心点
    left_angle = compute_angle(vertical, left_vector)
    
    # 计算右侧CE角度
    right_vector = points[3] - points[1]  # 右侧外缘点 - 右侧中心点
    right_angle = compute_angle(vertical, right_vector)
    
    return left_angle, right_angle 

def calculate_keypoint_accuracy(pred, target, threshold=0.05):
    """
    计算关键点预测准确率
    
    Args:
        pred: 预测的关键点坐标 [batch_size, num_keypoints, 2]
        target: 目标关键点坐标 [batch_size, num_keypoints, 2]
        threshold: 距离阈值，小于此阈值视为正确预测
        
    Returns:
        accuracy: 准确率 (0-1之间的值)
    """
    batch_size, num_keypoints, _ = pred.size()
    
    # 计算预测点与目标点之间的欧氏距离
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=2))
    
    # 计算正确预测的点数
    correct = (distances < threshold).float().sum().item()
    
    # 计算准确率
    return correct / (batch_size * num_keypoints) 