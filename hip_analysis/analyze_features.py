import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import sys

# Make hip_model importable when running from hip_analysis/.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from hip_model.models.cnn_gat_model import CNN_GAT
from hip_model.dataset import HipKeypointDataset, get_transforms
from torch.utils.data import DataLoader

def extract_features(model, dataloader, device):
    """提取模型特征"""
    model.eval()
    fusion_features = []
    pred_features = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            
            # 获取模型预测
            predictions = model(images, return_features=True)
            
            # 提取融合特征
            fusion_feat = predictions['fusion_features'].cpu().numpy()
            fusion_features.append(fusion_feat)
            
            # 提取预测特征
            pred_feat = predictions['keypoints'].cpu().numpy()
            pred_features.append(pred_feat)
            
            # 保存标签
            labels.append(keypoints.cpu().numpy())
    
    return (np.concatenate(fusion_features, axis=0),
            np.concatenate(pred_features, axis=0),
            np.concatenate(labels, axis=0))

def visualize_tsne(features, labels, title, save_path):
    """t-SNE可视化"""
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 应用t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features_scaled)
    
    # 创建可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = CNN_GAT(
        feature_dim=256,
        gat_hidden=128,
        gat_output=64,
        edge_features_dim=32,
        num_keypoints=9,
        num_angles=6
    ).to(device)
    
    # 加载模型权重
    model_path = 'outputs/models/best_model.pth'  # 修改为正确的模型路径
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建数据加载器
    dataset = HipKeypointDataset(
        img_dir='data/train',
        transform=get_transforms(train=False)
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 提取特征
    fusion_features, pred_features, labels = extract_features(model, dataloader, device)
    
    # 创建输出目录
    os.makedirs('analysis_results', exist_ok=True)
    
    # 可视化融合特征
    visualize_tsne(
        fusion_features,
        labels[:, 0, 0],  # 使用第一个关键点的x坐标作为标签
        't-SNE Visualization of Fusion Features',
        'analysis_results/fusion_features_tsne.png'
    )
    
    # 可视化预测特征
    visualize_tsne(
        pred_features.reshape(pred_features.shape[0], -1),
        labels[:, 0, 0],
        't-SNE Visualization of Prediction Features',
        'analysis_results/prediction_features_tsne.png'
    )

if __name__ == '__main__':
    main() 
