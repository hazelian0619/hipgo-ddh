from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class DataConfig:
    """数据集配置"""
    # 数据目录和文件大小
    data_dir: str = 'raw_labeled_data'
    image_size: int = 512
    
    # 关键点相关
    num_keypoints: int = 9
    num_angles: int = 6
    
    # 数据加载配置
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # 数据增强配置
    color_jitter: float = 0.2
    rotation_range: int = 15
    scale_range: Tuple[float, float] = (0.8, 1.0)

@dataclass
class ModelConfig:
    """模型配置"""
    # CNN骨干网络
    backbone: str = 'resnet50'
    pretrained: bool = True
    
    # 特征维度
    feature_dim: int = 256
    node_feature_dim: int = 128
    
    # GAT配置
    gat_layers: int = 3
    gat_heads: int = 4
    gat_feature_dim: int = 64
    gat_hidden: int = 128
    gat_output: int = 64
    
    # 其他参数
    num_keypoints: int = 9
    dropout: float = 0.2

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    batch_size: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    num_epochs: int = 100
    
    # 梯度累积和裁剪
    gradient_accumulation_steps: int = 4
    max_norm: float = 1.0
    
    # 早停和模型保存
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    save_interval: int = 5
    
    # 损失权重
    angle_weight: float = 0.1
    
    # 验证配置
    val_interval: int = 1

@dataclass
class Config:
    """全局配置"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # 输出配置
    output_dir: str = 'output'
    checkpoint_dir: str = 'output/checkpoints'
    log_dir: str = 'output/logs'
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        """确保输出目录存在"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

# 默认配置实例
config = Config()
