import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableFusionLayer(nn.Module):
    """
    可学习的CNN-GAT融合层
    
    实现CNN和GAT的双向特征交互，通过注意力机制动态调整CNN特征与图结构特征的权重
    """
    def __init__(self, cnn_feature_dim, gat_feature_dim, fusion_dim=256):
        super(LearnableFusionLayer, self).__init__()
        self.cnn_feature_dim = cnn_feature_dim
        self.gat_feature_dim = gat_feature_dim
        self.fusion_dim = fusion_dim
        
        # CNN特征映射到融合空间
        self.cnn_transform = nn.Sequential(
            nn.Linear(cnn_feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # GAT特征映射到融合空间
        self.gat_transform = nn.Sequential(
            nn.Linear(gat_feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(fusion_dim * 2, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合特征变换
        self.fusion_transform = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # CNN特征更新门控网络
        self.cnn_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # GAT特征更新门控网络
        self.gat_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.cnn_output_proj = nn.Linear(fusion_dim, cnn_feature_dim)
        self.gat_output_proj = nn.Linear(fusion_dim, gat_feature_dim)
    
    def forward(self, cnn_features, gat_features):
        """
        前向传播
        
        参数:
            cnn_features: CNN特征 [batch_size, cnn_feature_dim]
            gat_features: GAT特征 [batch_size, gat_feature_dim]
            
        返回:
            updated_cnn_features: 更新后的CNN特征 [batch_size, cnn_feature_dim]
            updated_gat_features: 更新后的GAT特征 [batch_size, gat_feature_dim]
            fusion_features: 融合特征 [batch_size, fusion_dim]
        """
        batch_size = cnn_features.size(0)
        
        # 将特征映射到相同的融合空间
        cnn_fusion = self.cnn_transform(cnn_features)  # [batch_size, fusion_dim]
        gat_fusion = self.gat_transform(gat_features)  # [batch_size, fusion_dim]
        
        # 计算注意力融合权重
        concat_features = torch.cat([cnn_fusion, gat_fusion], dim=1)  # [batch_size, fusion_dim*2]
        attn_weights = self.attention_weights(concat_features)  # [batch_size, 2]
        
        # 加权融合特征
        cnn_weight = attn_weights[:, 0].unsqueeze(1)  # [batch_size, 1]
        gat_weight = attn_weights[:, 1].unsqueeze(1)  # [batch_size, 1]
        
        fusion_features = cnn_weight * cnn_fusion + gat_weight * gat_fusion  # [batch_size, fusion_dim]
        fusion_features = self.fusion_transform(fusion_features)  # [batch_size, fusion_dim]
        
        # 计算更新门控
        cnn_update_gate = self.cnn_gate(torch.cat([cnn_fusion, fusion_features], dim=1))  # [batch_size, fusion_dim]
        gat_update_gate = self.gat_gate(torch.cat([gat_fusion, fusion_features], dim=1))  # [batch_size, fusion_dim]
        
        # 使用门控机制更新特征
        updated_cnn_fusion = (1 - cnn_update_gate) * cnn_fusion + cnn_update_gate * fusion_features
        updated_gat_fusion = (1 - gat_update_gate) * gat_fusion + gat_update_gate * fusion_features
        
        # 投影回原始特征空间
        updated_cnn_features = self.cnn_output_proj(updated_cnn_fusion)  # [batch_size, cnn_feature_dim]
        updated_gat_features = self.gat_output_proj(updated_gat_fusion)  # [batch_size, gat_feature_dim]
        
        return updated_cnn_features, updated_gat_features, fusion_features


class SpatialGraphFusion(nn.Module):
    """
    空间-图特征融合模块
    
    实现CNN特征图和图特征之间的空间对齐和交互
    """
    def __init__(self, cnn_feature_dim, gat_feature_dim, fusion_dim=256):
        super(SpatialGraphFusion, self).__init__()
        self.cnn_feature_dim = cnn_feature_dim
        self.gat_feature_dim = gat_feature_dim
        self.fusion_dim = fusion_dim
        
        # 点级特征融合层
        self.point_fusion = LearnableFusionLayer(
            cnn_feature_dim=cnn_feature_dim,
            gat_feature_dim=gat_feature_dim,
            fusion_dim=fusion_dim
        )
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(cnn_feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cnn_feature_dim, cnn_feature_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_feature_dim // 16, cnn_feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, cnn_feature_map, graph_features, keypoint_positions):
        batch_size, _, height, width = cnn_feature_map.shape
        num_keypoints = graph_features.size(1)

        fusion_map = cnn_feature_map.clone()
        enhanced_graph_list = []

        for b in range(batch_size):
            b_graph_feats = []
            for k in range(num_keypoints):
                pos = keypoint_positions[b, k]
                x, y = int(pos[0] * (width - 1)), int(pos[1] * (height - 1))

                if 0 <= x < width and 0 <= y < height:
                    cnn_point_feature = cnn_feature_map[b, :, y, x]
                    graph_point_feature = graph_features[b, k]

                    updated_cnn, updated_graph, _ = self.point_fusion(
                        cnn_point_feature.unsqueeze(0),
                        graph_point_feature.unsqueeze(0)
                    )
                    b_graph_feats.append(updated_graph.squeeze(0))

                    # 非inplace更新特征图位置
                    mask = torch.zeros(1, self.cnn_feature_dim, height, width, device=cnn_feature_map.device)
                    mask[0, :, y, x] = 1.0
                    fusion_map = fusion_map * (1 - mask) + updated_cnn.squeeze(0).unsqueeze(-1).unsqueeze(-1) * mask
                else:
                    b_graph_feats.append(graph_features[b, k])

            enhanced_graph_list.append(torch.stack(b_graph_feats))

        enhanced_graph_features = torch.stack(enhanced_graph_list)

        spatial_attn = self.spatial_attention(fusion_map)
        channel_attn = self.channel_attention(fusion_map)
        enhanced_feature_map = fusion_map * spatial_attn * channel_attn

        return enhanced_feature_map, enhanced_graph_features 