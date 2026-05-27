import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.feature_extractor import FeatureExtractor
from models.gat.edge_feature_enhancer import EdgeFeatureEnhancer
from models.gat.graph_attention_layer import MultiHeadGraphAttention
from models.fusion.learnable_fusion_layer import LearnableFusionLayer, SpatialGraphFusion

class KeypointFeatureExtractor(nn.Module):
    """
    关键点特征提取器
    从CNN特征图中提取关键点特征
    """
    def __init__(self, cnn_feature_dim, node_feature_dim):
        super(KeypointFeatureExtractor, self).__init__()
        self.cnn_feature_dim = cnn_feature_dim
        self.node_feature_dim = node_feature_dim
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, node_feature_dim),
            nn.LayerNorm(node_feature_dim),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, feature_maps, keypoint_positions):
        """
        前向传播
        
        参数:
            feature_maps: 多尺度CNN特征图字典，每个尺度包含 [batch_size, cnn_feature_dim, H, W]
            keypoint_positions: 关键点位置 [batch_size, num_keypoints, 2]，取值范围为[0,1]
            
        返回:
            node_features: 关键点节点特征 [batch_size, num_keypoints, node_feature_dim]
        """
        batch_size, num_keypoints = keypoint_positions.shape[:2]
        
        # 初始化节点特征
        node_features = torch.zeros(batch_size, num_keypoints, self.node_feature_dim, device=keypoint_positions.device)
        
        # 获取多尺度特征
        p2 = feature_maps['p2']  # 高分辨率特征 (1/4)
        p3 = feature_maps['p3']  # 中分辨率特征 (1/8)
        p4 = feature_maps['p4']  # 低分辨率特征 (1/16)
        p5 = feature_maps['p5']  # 最低分辨率特征 (1/32)
        
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
                
                # 自适应处理特征维度
                if sampled_feature.shape[0] == self.cnn_feature_dim:
                    # 维度匹配，直接使用（不要直接赋值引用）
                    feature_vector[:] = sampled_feature
                elif sampled_feature.shape[0] > self.cnn_feature_dim:
                    # 维度过大，截断或平均
                    if sampled_feature.shape[0] % self.cnn_feature_dim == 0:
                        # 如果是整数倍，做平均
                        factor = sampled_feature.shape[0] // self.cnn_feature_dim
                        sampled_feature = sampled_feature.view(self.cnn_feature_dim, factor).mean(dim=1)
                        feature_vector[:] = sampled_feature
                    else:
                        # 否则截断
                        feature_vector[:] = sampled_feature[:self.cnn_feature_dim]
                else:
                    # 维度不足，填充
                    feature_vector[:sampled_feature.shape[0]] = sampled_feature
                
                # 投影到节点特征空间
                node_feature = self.feature_projection(feature_vector)
                
                # 存储节点特征
                node_features[b, k] = node_feature
        
        return node_features

class GraphBuilder(nn.Module):
    """
    图构建模块
    构建关键点之间的图结构
    """
    def __init__(self, threshold_distance=0.3, include_self_loops=True):
        super(GraphBuilder, self).__init__()
        self.threshold_distance = threshold_distance
        self.include_self_loops = include_self_loops
    
    def forward(self, keypoint_positions):
        """
        前向传播，构建邻接矩阵
        
        参数:
            keypoint_positions: 关键点位置 [batch_size, num_keypoints, 2]，取值范围为[0,1]
            
        返回:
            adjacency: 邻接矩阵 [batch_size, num_keypoints, num_keypoints]
        """
        batch_size, num_keypoints, _ = keypoint_positions.shape
        
        # 初始化邻接矩阵
        adjacency = torch.zeros(batch_size, num_keypoints, num_keypoints, device=keypoint_positions.device)
        
        for b in range(batch_size):
            # 计算每对关键点之间的欧几里得距离
            points = keypoint_positions[b]  # [num_keypoints, 2]
            
            # 扩展维度以计算成对距离
            p1 = points.unsqueeze(1)  # [num_keypoints, 1, 2]
            p2 = points.unsqueeze(0)  # [1, num_keypoints, 2]
            
            # 计算欧几里得距离
            distances = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1))  # [num_keypoints, num_keypoints]
            
            # 根据距离阈值创建邻接矩阵
            adjacency[b] = (distances < self.threshold_distance).float()
            
            # 处理自环
            if self.include_self_loops:
                # 添加自环
                adjacency[b].fill_diagonal_(1.0)
            else:
                # 移除自环
                adjacency[b].fill_diagonal_(0.0)
                
        return adjacency


class GraphAttentionNetwork(nn.Module):
    """
    图注意力网络
    对关键点节点特征进行注意力加权的消息传递
    """
    def __init__(
        self, 
        in_features, 
        hidden_features,
        out_features,
        edge_features_dim=32,
        num_layers=2,
        num_heads=8,
        dropout=0.1
    ):
        super(GraphAttentionNetwork, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_features_dim = edge_features_dim
        self.num_layers = num_layers
        
        # 边特征增强器
        self.edge_enhancer = EdgeFeatureEnhancer(
            in_features=in_features,
            edge_features_dim=edge_features_dim
        )
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        
        # 第一层
        self.gat_layers.append(
            MultiHeadGraphAttention(
                in_features=in_features,
                out_features=hidden_features // num_heads,
                edge_features_dim=edge_features_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True,
                use_edge_features=True
            )
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                MultiHeadGraphAttention(
                    in_features=hidden_features,
                    out_features=hidden_features // num_heads,
                    edge_features_dim=edge_features_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    use_edge_features=True
                )
            )
        
        # 最后一层
        if num_layers > 1:
            self.gat_layers.append(
                MultiHeadGraphAttention(
                    in_features=hidden_features,
                    out_features=out_features,
                    edge_features_dim=edge_features_dim,
                    heads=1,
                    dropout=dropout,
                    concat=False,
                    use_edge_features=True
                )
            )
    
    def forward(self, node_features, keypoint_positions, adjacency):
        """
        前向传播
        
        参数:
            node_features: 节点特征 [batch_size, num_keypoints, in_features]
            keypoint_positions: 关键点位置 [batch_size, num_keypoints, 2]
            adjacency: 邻接矩阵 [batch_size, num_keypoints, num_keypoints]
            
        返回:
            output_features: 输出节点特征 [batch_size, num_keypoints, out_features]
        """
        batch_size, num_keypoints, _ = node_features.shape
        outputs = []
        
        # 逐批次处理
        for b in range(batch_size):
            # 提取当前批次的数据
            curr_node_features = node_features[b]  # [num_keypoints, in_features]
            curr_positions = keypoint_positions[b]  # [num_keypoints, 2]
            curr_adjacency = adjacency[b]  # [num_keypoints, num_keypoints]
            
            # 构建边特征
            edge_features, _ = self.edge_enhancer(curr_node_features, curr_positions, curr_adjacency)
            
            # 进行图注意力传播
            x = curr_node_features
            for i, gat_layer in enumerate(self.gat_layers):
                x = gat_layer(x, curr_adjacency, edge_features)
                
            # 存储批次输出
            outputs.append(x)
        
        # 合并批次输出
        output_features = torch.stack(outputs)
        
        return output_features

class KeypointPredictor(nn.Module):
    """
    关键点预测器
    基于融合特征预测关键点坐标
    """
    def __init__(self, in_features, hidden_features=128, num_keypoints=9):
        super(KeypointPredictor, self).__init__()
        
        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, num_keypoints * 2)
        )
        
        self.num_keypoints = num_keypoints
        
    def forward(self, features):
        """
        前向传播
        
        参数:
            features: 输入特征 [batch_size, in_features]
            
        返回:
            keypoints: 预测的关键点坐标 [batch_size, num_keypoints, 2]
        """
        batch_size = features.size(0)
        
        # 预测关键点
        pred = self.predictor(features)
        pred = pred.view(batch_size, self.num_keypoints, 2)
        
        # 使用Sigmoid确保坐标在[0,1]范围内
        keypoints = torch.sigmoid(pred)
        
        return keypoints


class AnglePredictor(nn.Module):
    """
    角度预测器
    基于关键点特征预测角度值
    """
    def __init__(self, in_features, hidden_features=128, num_angles=6):
        super(AnglePredictor, self).__init__()
        
        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, num_angles)
        )
        
    def forward(self, features):
        """
        前向传播
        
        参数:
            features: 输入特征 [batch_size, in_features]
            
        返回:
            angles: 预测的角度值 [batch_size, num_angles]
        """
        # 预测角度
        angles = self.predictor(features)
        
        return angles


class CNN_GAT(nn.Module):
    """
    深度融合CNN-GAT模型
    用于骨盆X光片关键点检测和空间关系建模
    """
    def __init__(
        self, 
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
    ):
        super(CNN_GAT, self).__init__()
        self.feature_dim = feature_dim
        self.gat_hidden = gat_hidden
        self.gat_output = gat_output
        self.edge_features_dim = edge_features_dim
        self.num_keypoints = num_keypoints
        self.num_angles = num_angles
        self.num_gat_layers = num_gat_layers
        
        # CNN特征提取器
        self.feature_extractor = FeatureExtractor(
            pretrained=pretrained,
            feature_dim=feature_dim
        )
        
        # 关键点特征提取器
        self.keypoint_feature_extractor = KeypointFeatureExtractor(
            cnn_feature_dim=feature_dim,
            node_feature_dim=gat_hidden
        )
        
        # 图构建器
        self.graph_builder = GraphBuilder(
            threshold_distance=0.3,
            include_self_loops=True
        )
        
        # 图注意力网络
        self.gat = GraphAttentionNetwork(
            in_features=gat_hidden,
            hidden_features=gat_hidden,
            out_features=gat_output,
            edge_features_dim=edge_features_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # CNN-GAT融合层
        self.fusion_layer = LearnableFusionLayer(
            cnn_feature_dim=feature_dim,
            gat_feature_dim=gat_output,
            fusion_dim=feature_dim
        )
        
        # 空间-图融合模块
        self.spatial_graph_fusion = SpatialGraphFusion(
            cnn_feature_dim=feature_dim,
            gat_feature_dim=gat_output,
            fusion_dim=feature_dim
        )
        
        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + gat_output, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 关键点预测器
        self.keypoint_predictor = KeypointPredictor(
            in_features=feature_dim,
            hidden_features=feature_dim // 2,
            num_keypoints=num_keypoints
        )
        
        # 角度预测器
        self.angle_predictor = AnglePredictor(
            in_features=feature_dim + gat_output * num_keypoints,
            hidden_features=feature_dim,
            num_angles=num_angles
        )
        
    def forward(self, x, keypoint_positions=None, adjacency=None, return_features: bool = False):
        """
        前向传播
        
        参数:
            x: 输入图像 [batch_size, 3, H, W]
            keypoint_positions: 关键点位置 [batch_size, num_keypoints, 2]，如果提供则只预测角度
            adjacency: 邻接矩阵 [batch_size, num_keypoints, num_keypoints]，如果为None则自动构建
            
        返回:
            dict: 包含预测的关键点和角度
        """
        batch_size = x.size(0)
        
        # 提取CNN特征
        multi_scale_features, main_features = self.feature_extractor(x)
        
        # 提取全局CNN特征
        cnn_global_features = self.global_avg_pool(main_features).view(batch_size, -1)
        
        # 训练和推理阶段处理
        if keypoint_positions is not None:
            # 使用提供的关键点位置（验证或推理阶段）
            
            # 提取关键点特征
            node_features = self.keypoint_feature_extractor(multi_scale_features, keypoint_positions)
            
            # 如果没有提供邻接矩阵，自动构建
            if adjacency is None:
                adjacency = self.graph_builder(keypoint_positions)
            
            # 应用图注意力网络
            graph_features = self.gat(node_features, keypoint_positions, adjacency)
            
            # CNN-GAT融合
            enhanced_feature_map, enhanced_graph_features = self.spatial_graph_fusion(
                main_features, graph_features, keypoint_positions
            )
            
            # 全局池化增强的CNN特征
            enhanced_cnn_global = self.global_avg_pool(enhanced_feature_map).view(batch_size, -1)
            
            # 平均池化图特征
            graph_global = torch.mean(enhanced_graph_features, dim=1)
            
            # 特征融合
            global_features = torch.cat([enhanced_cnn_global, graph_global], dim=1)
            fused_features = self.fc(global_features)
            
            # 预测角度
            # 关键点特征和全局特征拼接
            angle_input = torch.cat([
                fused_features,
                enhanced_graph_features.view(batch_size, -1)
            ], dim=1)
            angles = self.angle_predictor(angle_input)
            
            out = {'keypoints': keypoint_positions, 'angles': angles}
            if return_features:
                out.update(
                    {
                        'cnn_global_features': cnn_global_features,
                        'fusion_features': fused_features,
                        'enhanced_cnn_global': enhanced_cnn_global,
                        'graph_global': graph_global,
                        'enhanced_graph_features': enhanced_graph_features,
                    }
                )
            return out
            
        else:
            # 训练阶段，预测关键点
            
            # 首先预测关键点位置
            keypoints = self.keypoint_predictor(cnn_global_features)
            
            # 构建图
            adjacency = self.graph_builder(keypoints)
            
            # 提取关键点特征
            node_features = self.keypoint_feature_extractor(multi_scale_features, keypoints)
            
            # 应用图注意力网络
            graph_features = self.gat(node_features, keypoints, adjacency)
            
            # CNN-GAT融合
            enhanced_feature_map, enhanced_graph_features = self.spatial_graph_fusion(
                main_features, graph_features, keypoints
            )
            
            # 全局池化增强的CNN特征
            enhanced_cnn_global = self.global_avg_pool(enhanced_feature_map).view(batch_size, -1)
            
            # 平均池化图特征
            graph_global = torch.mean(enhanced_graph_features, dim=1)
            
            # 特征融合
            global_features = torch.cat([enhanced_cnn_global, graph_global], dim=1)
            fused_features = self.fc(global_features)
            
            # 预测角度
            # 关键点特征和全局特征拼接
            angle_input = torch.cat([
                fused_features,
                enhanced_graph_features.view(batch_size, -1)
            ], dim=1)
            angles = self.angle_predictor(angle_input)
            
            out = {'keypoints': keypoints, 'angles': angles}
            if return_features:
                out.update(
                    {
                        'cnn_global_features': cnn_global_features,
                        'fusion_features': fused_features,
                        'enhanced_cnn_global': enhanced_cnn_global,
                        'graph_global': graph_global,
                        'enhanced_graph_features': enhanced_graph_features,
                    }
                )
            return out

# 为了兼容性添加CNN_GAT_Model作为CNN_GAT的别名
CNN_GAT_Model = CNN_GAT 
