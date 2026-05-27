import os
import json
import pandas as pd
from utils.metrics import calculate_bilateral_ce_angles
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from utils.visualization import plot_performance_radar

def process_all_images():
    """批量处理所有图片并保存结果"""
    results = []
    ann_dir = 'labeled_data/train/annotations'
    
    # 遍历所有标注文件
    for json_file in os.listdir(ann_dir):
        if not json_file.endswith('.json'):
            continue
            
        # 读取标注
        with open(os.path.join(ann_dir, json_file), 'r') as f:
            data = json.load(f)
        
        # 提取点坐标
        points = [shape['points'][0] for shape in data['shapes']]
        
        # 计算角度
        left_angle, right_angle = calculate_bilateral_ce_angles(points)
        
        # 保存结果
        results.append({
            'image_id': json_file.replace('.json', ''),
            'left_ce': left_angle,
            'right_ce': right_angle
        })
        
        # 生成可视化结果
        save_visualization(json_file, points, left_angle, right_angle)
    
    # 保存到Excel
    df = pd.DataFrame(results)
    df.to_excel('ce_angle_results.xlsx', index=False)
    print(f"处理完成 {len(results)} 张图片")
    
    # 统计分析
    print("\n=== 统计结果 ===")
    print("左侧CE角度: {:.2f}° ± {:.2f}°".format(
        df['left_ce'].mean(), df['left_ce'].std()))
    print("右侧CE角度: {:.2f}° ± {:.2f}°".format(
        df['right_ce'].mean(), df['right_ce'].std()))

def process_and_visualize_all():
    """批量处理并生成可视化报告"""
    results = []
    output_dir = Path('output/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 处理所有图片
    for img_path in sorted(Path('labeled_data/train/images').glob('*.jpg')):
        # 加载图片和标注
        image = plt.imread(str(img_path))
        points = load_points(img_path)  # 从对应的标注文件加载点
        
        # 计算CE角度
        left_angle, right_angle = calculate_bilateral_ce_angles(points)
        
        # 保存结果
        results.append({
            'image_id': img_path.stem,
            'left_ce': left_angle,
            'right_ce': right_angle
        })
        
        # 生成可视化
        save_path = output_dir / f'{img_path.stem}_analysis.png'
        visualize_ce_angles(image, points, save_path)
    
    # 2. 生成统计报告
    df = pd.DataFrame(results)
    generate_statistics_report(df)
    
    # 添加性能分析
    metrics = analyze_performance(df)
    
    # 打印性能指标
    print("\n=== 性能指标 ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

def generate_statistics_report(df):
    """生成统计分析报告"""
    # 1. 创建统计图表
    plt.figure(figsize=(15, 5))
    
    # 左侧CE角度分布
    plt.subplot(121)
    plt.hist(df['left_ce'], bins=20, alpha=0.7, color='blue')
    plt.title('左侧CE角度分布')
    plt.xlabel('角度')
    plt.ylabel('频次')
    
    # 右侧CE角度分布
    plt.subplot(122)
    plt.hist(df['right_ce'], bins=20, alpha=0.7, color='red')
    plt.title('右侧CE角度分布')
    plt.xlabel('角度')
    
    plt.tight_layout()
    plt.savefig('output/ce_angle_distribution.png')
    
    # 2. 生成统计报告
    stats = {
        '左侧CE角度': {
            '平均值': df['left_ce'].mean(),
            '标准差': df['left_ce'].std(),
            '最小值': df['left_ce'].min(),
            '最大值': df['left_ce'].max()
        },
        '右侧CE角度': {
            '平均值': df['right_ce'].mean(),
            '标准差': df['right_ce'].std(),
            '最小值': df['right_ce'].min(),
            '最大值': df['right_ce'].max()
        }
    }
    
    # 保存统计结果
    pd.DataFrame(stats).to_excel('output/ce_angle_statistics.xlsx')

def load_points(img_path):
    """从标注文件加载关键点"""
    # 构建对应的标注文件路径
    json_path = img_path.parent.parent / 'annotations' / f'{img_path.stem}.json'
    
    # 读取标注文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取关键点
    points = []
    for shape in data['shapes']:
        point = shape['points'][0]
        points.append(point)
    
    return np.array(points)

def analyze_performance(results_df):
    """分析性能指标并生成雷达图"""
    metrics = {
        '左CE角准确率': calculate_accuracy(results_df['left_ce']),
        '右CE角准确率': calculate_accuracy(results_df['right_ce']),
        '测量稳定性': calculate_stability(results_df),
        '左右对称性': calculate_symmetry(results_df),
        '整体可靠性': calculate_reliability(results_df)
    }
    
    # 创建figures目录
    os.makedirs('figures', exist_ok=True)
    
    # 生成雷达图
    plot_performance_radar(metrics)
    
    return metrics

def calculate_accuracy(angles):
    """计算角度测量准确率"""
    # 假设标准范围是20-40度
    in_range = ((angles >= 20) & (angles <= 40)).mean()
    return float(in_range)

def calculate_stability(df):
    """计算测量稳定性"""
    std = df[['left_ce', 'right_ce']].std().mean()
    return 1.0 / (1.0 + std)  # 转换为0-1分数

def calculate_symmetry(df):
    """计算左右对称性"""
    diff = abs(df['left_ce'] - df['right_ce']).mean()
    return 1.0 / (1.0 + diff)  # 转换为0-1分数

def calculate_reliability(df):
    """计算整体可靠性"""
    return (calculate_accuracy(df['left_ce']) + 
            calculate_accuracy(df['right_ce'])) / 2 