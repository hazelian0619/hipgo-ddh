import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class FeatureExtractor(nn.Module):
    """
    基于ResNet50的特征提取器，用于从X光图像中提取关键特征
    """
    def __init__(self, pretrained=True, feature_dim=256):
        super(FeatureExtractor, self).__init__()
        # 加载预训练的ResNet50模型
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # 获取ResNet50各个阶段，而不是直接用Sequential
        # 这样可以更精确地控制特征提取过程
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        
        self.layer1 = self.backbone.layer1  # 输出 256 通道
        self.layer2 = self.backbone.layer2  # 输出 512 通道
        self.layer3 = self.backbone.layer3  # 输出 1024 通道
        self.layer4 = self.backbone.layer4  # 输出 2048 通道
        
        # 添加特征降维层
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(2048, feature_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度特征提取（FPN结构）
        self.lateral_conv1 = nn.Conv2d(1024, feature_dim, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(512, feature_dim, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(256, feature_dim, kernel_size=1)
        
        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 输出卷积层
        self.output_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 [batch_size, 3, H, W]
            
        返回:
            multi_scale_features: 多尺度特征字典
            final_features: 最终融合特征 [batch_size, feature_dim, H/16, W/16]
        """
        # 保存中间特征
        features = []
        
        # 提取ResNet各阶段特征 - 修改为直接使用各个层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)  # 256 channels, 1/4 resolution
        
        x = self.layer2(x)
        features.append(x)  # 512 channels, 1/8 resolution
        
        x = self.layer3(x)
        features.append(x)  # 1024 channels, 1/16 resolution
        
        x = self.layer4(x)
        features.append(x)  # 2048 channels, 1/32 resolution
        
        # 特征降维
        p5 = self.feature_reduction(features[3])  # 1/32 resolution
        
        # 自顶向下路径 (top-down pathway)
        p4 = self.upsample(p5) + self.lateral_conv1(features[2])  # 1/16 resolution
        p3 = self.upsample(p4) + self.lateral_conv2(features[1])  # 1/8 resolution
        p2 = self.upsample(p3) + self.lateral_conv3(features[0])  # 1/4 resolution
        
        # 最终处理
        p2 = self.output_conv(p2)
        p3 = self.output_conv(p3)
        p4 = self.output_conv(p4)
        p5 = self.output_conv(p5)
        
        # 返回多尺度特征
        multi_scale_features = {
            'p2': p2,  # 高分辨率特征，适合精细定位
            'p3': p3,
            'p4': p4,
            'p5': p5   # 低分辨率特征，适合全局信息
        }
        
        # 返回最终特征（p4层，1/16分辨率作为主要特征）
        return multi_scale_features, p4 