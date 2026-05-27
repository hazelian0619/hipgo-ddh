import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    图注意力层
    参考自论文 "Graph Attention Networks" (Veličković et al.)
    并整合了边特征注意力机制，参考自 "GAT with Edge Feature Attention"
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        edge_features_dim=32,
        dropout=0.1, 
        alpha=0.2, 
        concat=True,
        use_edge_features=True
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features_dim = edge_features_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.use_edge_features = use_edge_features
        
        # 节点特征变换
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # 自注意力机制
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # 边特征注意力机制
        if self.use_edge_features:
            self.edge_attention = nn.Linear(edge_features_dim, 1, bias=False)
            
            # 边特征整合机制
            self.edge_feature_transform = nn.Linear(edge_features_dim, out_features, bias=False)
        
        # 参数初始化
        self.reset_parameters()
        
        # 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.activation = nn.ELU()
        
    def reset_parameters(self):
        """初始化参数"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a.weight, gain=gain)
        if self.use_edge_features:
            nn.init.xavier_normal_(self.edge_attention.weight, gain=gain)
            nn.init.xavier_normal_(self.edge_feature_transform.weight, gain=gain)
            
    def forward(self, input, adj, edge_features=None):
        """
        前向传播
        
        参数:
            input: 输入节点特征 [N, in_features]
            adj: 邻接矩阵 [N, N]
            edge_features: 边特征 [N, N, edge_features_dim]
            
        返回:
            output: 输出节点特征 [N, out_features]
        """
        # 线性变换节点特征
        h = self.W(input)  # [N, out_features]
        N = h.size(0)
        
        # 计算注意力系数 (普通GAT注意力)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, N, 2 * self.out_features)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # [N, N]
        
        # 整合边特征注意力 (如果使用边特征)
        if self.use_edge_features and edge_features is not None:
            # 计算边特征注意力
            edge_attn = self.leakyrelu(self.edge_attention(edge_features).squeeze(-1))  # [N, N]
            
            # 组合节点注意力和边特征注意力
            e = e + edge_attn
        
        # 掩码和归一化注意力系数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用注意力系数
        h_prime = torch.matmul(attention, h)  # [N, out_features]
        
        # 整合边特征 (如果使用边特征)
        if self.use_edge_features and edge_features is not None:
            # 变换边特征
            transformed_edge_features = self.edge_feature_transform(edge_features)  # [N, N, out_features]
            
            # 计算边特征的加权和
            # 使用广播机制，将attention从[N, N]扩展到[N, N, 1]
            expanded_attention = attention.unsqueeze(-1)
            edge_contribution = (transformed_edge_features * expanded_attention).sum(dim=1)  # [N, out_features]
            
            # 组合节点特征和边特征
            h_prime = h_prime + edge_contribution
        
        # 应用非线性激活函数
        if self.concat:
            return self.activation(h_prime)
        else:
            return h_prime
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadGraphAttention(nn.Module):
    """
    多头图注意力层
    包含多个并行的图注意力层，并将它们的输出拼接或平均
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        edge_features_dim=32,
        heads=8, 
        dropout=0.1, 
        alpha=0.2, 
        concat=True,
        use_edge_features=True
    ):
        super(MultiHeadGraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # 创建多个图注意力层
        self.attentions = nn.ModuleList()
        for _ in range(heads):
            self.attentions.append(
                GraphAttentionLayer(
                    in_features=in_features, 
                    out_features=out_features,
                    edge_features_dim=edge_features_dim,
                    dropout=dropout, 
                    alpha=alpha, 
                    concat=concat,
                    use_edge_features=use_edge_features
                )
            )
            
        # 输出层Dropout
        self.out_dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj, edge_features=None):
        """
        前向传播
        
        参数:
            x: 输入节点特征 [N, in_features]
            adj: 邻接矩阵 [N, N]
            edge_features: 边特征 [N, N, edge_features_dim]
            
        返回:
            output: 输出节点特征 [N, out_features * heads] 或 [N, out_features]
        """
        # 合并所有注意力头的输出
        if self.concat:
            # 拼接每个注意力头的输出
            output = torch.cat([att(x, adj, edge_features) for att in self.attentions], dim=1)
        else:
            # 平均每个注意力头的输出
            output = torch.mean(torch.stack([att(x, adj, edge_features) for att in self.attentions]), dim=0)
            
        # 应用Dropout
        output = self.out_dropout(output)
        return output 