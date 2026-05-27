"""纯CNN关键点检测模型 — HipGo实验最佳架构"""
import torch
import torch.nn as nn
from hipgo.models.backbone import FeatureExtractor


class CNNKeypoint(nn.Module):
    """ResNet50骨干 + 自适应池化 + 3层MLP预测头，输出9个关键点坐标"""

    def __init__(self, pretrained=True, num_keypoints=9):
        super().__init__()
        self.backbone = FeatureExtractor(pretrained=pretrained, feature_dim=256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.predictor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_keypoints * 2),
        )
        self.num_keypoints = num_keypoints

    def forward(self, x):
        _, main_feat = self.backbone(x)               # [B, 256, H, W]
        gf = self.global_pool(main_feat).view(x.size(0), -1)  # [B, 256]
        pred = self.predictor(gf).view(x.size(0), self.num_keypoints, 2)
        return {'keypoints': torch.sigmoid(pred)}
