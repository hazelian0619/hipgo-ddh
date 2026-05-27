#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时修复CNN-GAT模型的关键点特征提取器问题
"""
import torch
import torch.nn.functional as F

def fix_keypoint_feature_extractor():
    """修复KeypointFeatureExtractor类的实现"""
    import models.cnn_gat_model as model_module
    
    # 保存原始实现的引用
    original_forward = model_module.KeypointFeatureExtractor.forward
    
    # 定义新的forward方法
    def fixed_forward(self, feature_maps, keypoint_positions):
        """
        修复后的前向传播方法，确保特征维度正确
        """
        batch_size, num_keypoints = keypoint_positions.shape[:2]
        
        # 初始化节点特征
        node_features = torch.zeros(batch_size, num_keypoints, self.node_feature_dim, device=keypoint_positions.device)
        
        # 获取多尺度特征
        p2 = feature_maps['p2']  # 高分辨率特征 (1/4)
        p3 = feature_maps['p3']  # 中分辨率特征 (1/8)
        p4 = feature_maps['p4']  # 低分辨率特征 (1/16)
        p5 = feature_maps['p5']  # 最低分辨率特征 (1/32)
        
        # 打印特征维度信息以进行调试
        print(f"p4特征形状: {p4.shape}, cnn_feature_dim: {self.cnn_feature_dim}")
        
        # 对每个关键点提取特征
        for b in range(batch_size):
            for k in range(num_keypoints):
                # 获取归一化的关键点位置 [0,1]
                pos = keypoint_positions[b, k]
                
                # 创建一个定长的特征向量
                feature_vector = torch.zeros(self.cnn_feature_dim, device=pos.device)
                
                # 从p4特征图提取特征（使用主要特征图）
                _, c, h, w = p4.shape
                
                # 计算特征图上的坐标
                x = pos[0] * w
                y = pos[1] * h
                
                # 使用双线性插值获取精确位置的特征
                grid = torch.zeros(1, 1, 1, 2, device=pos.device)
                grid[0, 0, 0, 0] = 2 * x / (w - 1) - 1  # 转换到[-1, 1]范围
                grid[0, 0, 0, 1] = 2 * y / (h - 1) - 1  # 转换到[-1, 1]范围
                
                # 使用grid_sample进行精确采样
                sampled_feature = F.grid_sample(
                    p4[b:b+1], 
                    grid, 
                    mode='bilinear', 
                    padding_mode='zeros', 
                    align_corners=True
                )
                
                # 将采样特征重塑为向量
                sampled_feature = sampled_feature.reshape(-1)
                
                # 打印采样特征维度进行调试
                if b == 0 and k == 0:
                    print(f"采样特征维度: {sampled_feature.shape}")
                
                # 自适应处理特征维度
                if sampled_feature.shape[0] == self.cnn_feature_dim:
                    # 维度匹配，直接使用
                    feature_vector = sampled_feature
                elif sampled_feature.shape[0] > self.cnn_feature_dim:
                    # 维度过大，截断或平均
                    if sampled_feature.shape[0] % self.cnn_feature_dim == 0:
                        # 如果是整数倍，做平均
                        factor = sampled_feature.shape[0] // self.cnn_feature_dim
                        sampled_feature = sampled_feature.view(self.cnn_feature_dim, factor).mean(dim=1)
                        feature_vector = sampled_feature
                    else:
                        # 否则截断
                        feature_vector = sampled_feature[:self.cnn_feature_dim]
                else:
                    # 维度不足，填充
                    feature_vector[:sampled_feature.shape[0]] = sampled_feature
                
                # 投影到节点特征空间
                node_feature = self.feature_projection(feature_vector)
                
                # 存储节点特征
                node_features[b, k] = node_feature
        
        return node_features
    
    # 用修复后的方法替换原方法
    model_module.KeypointFeatureExtractor.forward = fixed_forward
    print("成功修复KeypointFeatureExtractor的forward方法!")

if __name__ == "__main__":
    fix_keypoint_feature_extractor()
    