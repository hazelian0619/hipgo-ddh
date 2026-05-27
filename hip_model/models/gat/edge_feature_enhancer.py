import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeFeatureEnhancer(nn.Module):
    """
    边缘特征增强模块
    基于GATwithEFA (Graph Attention Network with Edge Feature Attention)
    
    该模块用于增强骨盆关键点之间的空间关系，构建包含距离、角度、相对位置等多种关系的边特征矩阵
    """
    def __init__(self, in_features, edge_features_dim=32):
        super(EdgeFeatureEnhancer, self).__init__()
        self.in_features = in_features
        self.edge_features_dim = edge_features_dim
        
        # 边特征生成网络
        self.edge_feature_generator = nn.Sequential(
            nn.Linear(in_features * 2 + 3, edge_features_dim), # 节点特征拼接 + 距离和角度特征
            nn.ReLU(inplace=True),
            nn.Linear(edge_features_dim, edge_features_dim),
            nn.ReLU(inplace=True)
        )
        
        # 边重要性注意力
        self.edge_attention = nn.Sequential(
            nn.Linear(edge_features_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_edge_spatial_features(self, node_positions):
        """
        计算边的空间特征（距离、方向等）
        
        参数:
            node_positions: 节点位置张量 [num_nodes, 2]
        
        返回:
            edge_spatial_features: 边空间特征 [num_nodes, num_nodes, 3]
        """
        num_nodes = node_positions.size(0)
        
        # 初始化边特征张量 [num_nodes, num_nodes, 3]
        edge_spatial_features = torch.zeros(num_nodes, num_nodes, 3, device=node_positions.device)
        
        # 计算节点之间的距离和方向
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # 不计算自环的特征
                    # 提取两个节点的坐标
                    pos_i = node_positions[i]
                    pos_j = node_positions[j]
                    
                    # 计算欧氏距离
                    distance = torch.norm(pos_i - pos_j)
                    
                    # 计算方向向量 (normalized)
                    direction = (pos_j - pos_i) / (distance + 1e-6)
                    
                    # 存储边特征: [距离, x方向, y方向]
                    edge_spatial_features[i, j, 0] = distance
                    edge_spatial_features[i, j, 1:] = direction
        
        return edge_spatial_features
    
    def build_edge_features(self, node_features, node_positions):
        """
        构建边特征矩阵
        
        参数:
            node_features: 节点特征矩阵 [num_nodes, in_features]
            node_positions: 节点位置坐标 [num_nodes, 2]
            
        返回:
            edge_features: 边特征矩阵 [num_nodes, num_nodes, edge_features_dim]
            edge_weights: 边权重矩阵 [num_nodes, num_nodes, 1]
        """
        num_nodes = node_features.size(0)
        
        # 计算边的空间特征
        edge_spatial_features = self.compute_edge_spatial_features(node_positions)
        
        # 初始化边特征矩阵
        edge_features = torch.zeros(num_nodes, num_nodes, self.edge_features_dim, device=node_features.device)
        
        # 为每条边生成特征
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # 不处理自环
                    # 拼接源节点和目标节点的特征
                    node_pair_features = torch.cat([
                        node_features[i], 
                        node_features[j], 
                        edge_spatial_features[i, j]
                    ], dim=0)
                    
                    # 生成边特征
                    edge_features[i, j] = self.edge_feature_generator(node_pair_features)
        
        # 计算边权重
        edge_weights = torch.zeros(num_nodes, num_nodes, 1, device=node_features.device)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_weights[i, j] = self.edge_attention(edge_features[i, j])
        
        return edge_features, edge_weights
    
    def forward(self, node_features, node_positions, adjacency=None):
        """
        前向传播
        
        参数:
            node_features: 节点特征矩阵 [num_nodes, in_features]
            node_positions: 节点位置坐标 [num_nodes, 2]
            adjacency: 邻接矩阵 [num_nodes, num_nodes]
            
        返回:
            edge_features: 边特征矩阵 [num_nodes, num_nodes, edge_features_dim]
            edge_weights: 边权重矩阵 [num_nodes, num_nodes, 1]
        """
        # 构建边特征
        edge_features, edge_weights = self.build_edge_features(node_features, node_positions)
        
        # 如果提供了邻接矩阵，将非邻接的边权重设为0
        if adjacency is not None:
            mask = (adjacency == 0).unsqueeze(-1)
            edge_weights = edge_weights.masked_fill(mask, 0)
        
        return edge_features, edge_weights 