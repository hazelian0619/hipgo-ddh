"""训练配置文件"""

import os
from datetime import datetime

class Config:
    # 数据集配置
    data_root = '../data/expand'
    train_ratio = 0.8
    val_ratio = 0.2
    
    # 模型配置
    pretrained_model = '../outputs/model_best_20250506_121253.pth'
    backbone = 'resnet50'
    feature_channels = 256
    num_heads = 8
    
    # 训练配置
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # GAT配置
    gat_hidden_dim = 256
    gat_output_dim = 128
    dropout = 0.1
    alpha = 0.2  # LeakyReLU slope
    
    # 损失函数权重
    keypoint_loss_weight = 1.0
    edge_loss_weight = 0.5
    angle_loss_weight = 0.3
    
    # 保存配置
    save_dir = '../outputs'
    exp_name = 'hip_detection'
    
    def get_save_path(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.save_dir, f'{self.exp_name}_{timestamp}') 