#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet 关键点检测模型（实验E: SOTA对比基线）

HRNet 通过保持高分辨率表示贯穿整个网络，在医学关键点检测中
是公认的SOTA架构（如 VertXNet, Cephalometric Landmark Detection 等均使用）。

这里用 torchvision 自带的 HRNet 作为骨干 + 关键点预测头，
与你的 CNN-GAT 在相同数据上公平对比。
"""

import torch
import torch.nn as nn
import torchvision.models as models


class HRNetKeypoint(nn.Module):
    """
    HRNet-W32 骨干 + 关键点预测头。
    对照实验E：纯CNN多尺度方案 vs 你的CNN-GAT图方案。
    """

    def __init__(self, num_keypoints=9, pretrained=True):
        super().__init__()
        # 尝试加载 HRNet。新版 torchvision 有，旧版可能没有。
        try:
            from torchvision.models import HRNet_W32_Weights
            weights = HRNet_W32_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.hrnet_w32(weights=weights)
        except (ImportError, AttributeError):
            # 回退：从 torch hub 加载
            try:
                self.backbone = torch.hub.load('pytorch/vision', 'hrnet_w32', pretrained=pretrained)
            except Exception:
                # 最终回退：使用 SimpleBaseline（ResNet + 反卷积）
                print("[HRNet不可用，回退到 SimpleBaseline (ResNet50 + Deconv)]")
                self.backbone = None
                self._init_simple_baseline(pretrained)

        # HRNet 最终输出 32 个关键点（COCO格式），我们改成 9 个
        if self.backbone is not None:
            # 替换最后的分类/回归头
            in_features = 32  # HRNet-W32 的最终通道数
            self.predictor = nn.Sequential(
                nn.Conv2d(in_features, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_keypoints * 2),
            )

    def _init_simple_baseline(self, pretrained):
        """SimpleBaseline: ResNet50 + 反卷积层，关键点检测常用基线"""
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的pool和fc
        self.predictor = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_keypoints * 2),
        )
        self._is_simple_baseline = True

    def forward(self, x):
        if self.backbone is None:
            raise RuntimeError("骨干网络未初始化")

        if hasattr(self, '_is_simple_baseline'):
            feat = self.backbone(x)
            pred = self.predictor(feat)
        else:
            # HRNet 需要特定处理
            feat = self.backbone(x)
            if isinstance(feat, dict):
                # HRNet 返回多尺度特征
                feat = list(feat.values())[-1]
            pred = self.predictor(feat)

        pred = pred.view(x.size(0), -1, 2)
        return {
            'keypoints': torch.sigmoid(pred),
            'angles': torch.zeros(x.size(0), 6, device=x.device),
        }
