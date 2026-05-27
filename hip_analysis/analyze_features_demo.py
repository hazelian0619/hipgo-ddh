import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

def generate_synthetic_data(n_samples=999):  # 修改为999以确保可以被3整除
    """生成模拟数据"""
    # 生成融合特征
    n_per_class = n_samples // 3
    
    # 生成三种不同模式的特征
    pattern1 = np.random.randn(n_per_class, 256) + np.array([2] * 256)
    pattern2 = np.random.randn(n_per_class, 256) - np.array([2] * 256)
    pattern3 = np.random.randn(n_per_class, 256)
    
    fusion_features = np.vstack([pattern1, pattern2, pattern3])
    
    # 生成预测特征 (9个关键点，每个2维坐标)
    pred_features = np.random.rand(n_samples, 9, 2)
    
    # 生成标签
    labels = np.zeros(n_samples)
    labels[:n_per_class] = 0  # 正常样本
    labels[n_per_class:2*n_per_class] = 1  # 轻度异常
    labels[2*n_per_class:] = 2  # 严重异常
    
    return fusion_features, pred_features, labels

def visualize_tsne(features, labels, title, save_path, perplexity=30):
    """t-SNE可视化"""
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 应用t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=42
    )
    features_2d = tsne.fit_transform(features_scaled)
    
    # 创建可视化
    plt.figure(figsize=(12, 8))
    
    # 使用更好的配色方案
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    # 为每个类别分别绘制散点图
    for i, label in enumerate(['Normal', 'Mild Anomaly', 'Severe Anomaly']):
        mask = labels == i
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[i],
            label=label,
            alpha=0.6,
            s=50
        )
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=10)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 美化
    plt.tight_layout()
    
    # 保存高质量图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_clusters(features_2d, labels):
    """分析聚类结果"""
    results = {}
    
    # 计算每个类别的中心点
    for i in range(3):
        mask = labels == i
        center = features_2d[mask].mean(axis=0)
        results[f'Class {i} Center'] = center
    
    # 计算类内距离
    for i in range(3):
        mask = labels == i
        points = features_2d[mask]
        center = results[f'Class {i} Center']
        distances = np.sqrt(((points - center) ** 2).sum(axis=1))
        results[f'Class {i} Intra-cluster Distance'] = distances.mean()
    
    # 计算类间距离
    for i in range(3):
        for j in range(i + 1, 3):
            center_i = results[f'Class {i} Center']
            center_j = results[f'Class {j} Center']
            distance = np.sqrt(((center_i - center_j) ** 2).sum())
            results[f'Distance between Class {i} and {j}'] = distance
    
    return results

def main():
    print("生成模拟数据...")
    fusion_features, pred_features, labels = generate_synthetic_data()
    
    # 创建输出目录
    os.makedirs('analysis_results', exist_ok=True)
    
    print("\n分析融合特征...")
    # 可视化融合特征
    visualize_tsne(
        fusion_features,
        labels,
        't-SNE Visualization of Fusion Features\n(256-dimensional to 2D projection)',
        'analysis_results/fusion_features_tsne.png'
    )
    
    print("\n分析预测特征...")
    # 可视化预测特征
    visualize_tsne(
        pred_features.reshape(pred_features.shape[0], -1),
        labels,
        't-SNE Visualization of Prediction Features\n(18-dimensional to 2D projection)',
        'analysis_results/prediction_features_tsne.png',
        perplexity=20
    )
    
    # 分析聚类结果
    print("\n分析聚类结果...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    fusion_features_2d = tsne.fit_transform(StandardScaler().fit_transform(fusion_features))
    cluster_analysis = analyze_clusters(fusion_features_2d, labels)
    
    print("\n聚类分析结果:")
    for metric, value in cluster_analysis.items():
        if 'Center' in metric:
            print(f"{metric}: ({value[0]:.2f}, {value[1]:.2f})")
        else:
            print(f"{metric}: {value:.2f}")

if __name__ == '__main__':
    main() 