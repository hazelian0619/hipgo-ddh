#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
骨盆X光片模型性能图表生成脚本
生成模型性能的bar graph和training steps折线图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import pandas as pd
from datetime import datetime
import seaborn as sns

# 设置输出目录
OUTPUT_DIR = "outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义颜色
DEEP_BLUE = "#003366"  # 深蓝色（主色调）
ORANGE_RED = "#FF5733"  # 桔红色（突出表现）

# 设置全局字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 优先使用Arial，然后是中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.style.use('ggplot')  # 使用ggplot风格，更加专业

def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_bar_chart(data, x_col, y_col, title, xlabel, ylabel, figsize=(12, 8), color_palette='viridis', save_path=None):
    """创建柱状图"""
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=x_col, y=y_col, data=data, palette=color_palette)
    
    # 设置标题和标签
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # 旋转x轴标签（如果需要）
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加数值标签
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=10, rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_grouped_bar_chart(data, x_col, y_cols, title, xlabel, ylabel, figsize=(14, 8), color_palette='viridis', save_path=None):
    """创建分组柱状图"""
    plt.figure(figsize=figsize)
    
    # 提取数据
    x = np.arange(len(data))
    width = 0.8 / len(y_cols)  # 柱状图宽度
    
    # 设置颜色
    colors = plt.cm.get_cmap(color_palette, len(y_cols))
    
    # 绘制柱状图
    for i, col in enumerate(y_cols):
        offset = width * i - width * (len(y_cols) - 1) / 2
        bars = plt.bar(x + offset, data[col], width, label=col, color=colors(i/len(y_cols)))
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 设置标题和标签
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # 设置x轴刻度
    plt.xticks(x, data[x_col], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_line_chart(data, x_col, y_cols, title, xlabel, ylabel, figsize=(12, 8), color_palette='viridis', save_path=None):
    """创建折线图"""
    plt.figure(figsize=figsize)
    
    # 设置颜色
    colors = plt.cm.get_cmap(color_palette, len(y_cols))
    
    # 绘制折线图
    for i, col in enumerate(y_cols):
        plt.plot(data[x_col], data[col], marker='o', linewidth=2.5, 
                markersize=8, label=col, color=colors(i/len(y_cols)))
    
    # 设置标题和标签
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # 设置x轴刻度
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_radar_chart(data, categories, title, figsize=(10, 10), save_path=None):
    """创建雷达图"""
    plt.figure(figsize=figsize)
    
    # 获取方法名称和指标
    methods = data['方法'].tolist()
    
    # 将数据转换为雷达图所需格式
    values = data[categories].values
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    # 绘制雷达图
    ax = plt.subplot(111, polar=True)
    
    for i, method in enumerate(methods):
        # 归一化数据以适应雷达图
        method_values = values[i].tolist()
        
        # 闭合图形
        method_values += method_values[:1]
        
        # 绘制折线
        ax.plot(angles, method_values, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, method_values, alpha=0.1, color=colors[i])
    
    # 设置雷达图属性
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    """主函数"""
    # 设置中文字体
    set_chinese_font()
    
    # 确保输出目录存在
    os.makedirs('outputs/charts', exist_ok=True)
    
    print("读取表格数据...")
    # 读取表格数据
    model_df = pd.read_csv('outputs/tables/model_comparison.csv')
    keypoint_df = pd.read_csv('outputs/tables/keypoint_accuracy.csv')
    ablation_df = pd.read_csv('outputs/tables/ablation_studies.csv')
    dataset_df = pd.read_csv('outputs/tables/dataset_comparison.csv')
    
    print("\n生成模型性能对比图...")
    # 1. 模型性能MAE柱状图
    create_bar_chart(
        model_df, 
        '方法', 
        'MAE (mm)', 
        '不同方法的平均绝对误差(MAE)对比', 
        '方法', 
        'MAE (mm)', 
        figsize=(14, 8), 
        save_path='outputs/charts/model_mae_comparison.png'
    )
    
    # 2. 模型性能PCK折线图
    create_line_chart(
        model_df, 
        '方法', 
        ['PCK@5mm (%)', 'PCK@10mm (%)'], 
        '不同方法的PCK指标对比', 
        '方法', 
        'PCK (%)', 
        figsize=(14, 8), 
        save_path='outputs/charts/model_pck_comparison.png'
    )
    
    # 3. 关键点精度对比图
    create_grouped_bar_chart(
        keypoint_df, 
        '关键点', 
        ['ResNet50-FPN (mm)', 'HRNet (mm)', 'Ours (mm)'], 
        '各关键点定位误差对比', 
        '关键点', 
        '平均误差 (mm)', 
        figsize=(16, 8), 
        save_path='outputs/charts/keypoint_accuracy_comparison.png'
    )
    
    # 4. 消融实验结果图
    create_line_chart(
        ablation_df, 
        '模型配置', 
        ['MAE (mm)', 'RMSE (mm)'], 
        '消融实验-误差指标变化', 
        '模型配置', 
        '误差 (mm)', 
        figsize=(14, 8), 
        save_path='outputs/charts/ablation_error_metrics.png'
    )
    
    create_line_chart(
        ablation_df, 
        '模型配置', 
        ['PCK@5mm (%)', 'PCK@10mm (%)'], 
        '消融实验-PCK指标变化', 
        '模型配置', 
        'PCK (%)', 
        figsize=(14, 8), 
        save_path='outputs/charts/ablation_pck_metrics.png'
    )
    
    # 5. 数据集对比图
    create_grouped_bar_chart(
        dataset_df, 
        '数据集', 
        ['MAE (mm)', 'RMSE (mm)'], 
        '不同数据集的误差指标对比', 
        '数据集', 
        '误差 (mm)', 
        figsize=(12, 8), 
        save_path='outputs/charts/dataset_error_comparison.png'
    )
    
    create_bar_chart(
        dataset_df, 
        '数据集', 
        'PCK@10mm (%)', 
        '不同数据集的PCK@10mm指标对比', 
        '数据集', 
        'PCK@10mm (%)', 
        figsize=(12, 8), 
        save_path='outputs/charts/dataset_pck_comparison.png'
    )
    
    # 6. 雷达图 - 模型综合性能对比
    # 归一化数据以适应雷达图 - 反转误差指标（越小越好）
    radar_df = model_df.copy()
    for col in ['MAE (mm)', 'RMSE (mm)']:
        max_val = radar_df[col].max()
        radar_df[col] = 1 - (radar_df[col] / max_val)  # 归一化并反转
    
    for col in ['PCK@5mm (%)', 'PCK@10mm (%)']:
        radar_df[col] = radar_df[col] / 100  # 归一化到[0,1]
    
    # 选择几个代表性方法进行对比
    selected_methods = ['ResNet50-FPN', 'HRNet', 'ResNet50-FPN + GCN', 'ResNet50-FPN + GAT (ours)']
    radar_categories = ['MAE (mm)', 'RMSE (mm)', 'PCK@5mm (%)', 'PCK@10mm (%)']
    
    filtered_radar_df = radar_df[radar_df['方法'].isin(selected_methods)]
    
    create_radar_chart(
        filtered_radar_df,
        radar_categories,
        '模型综合性能对比',
        figsize=(10, 10),
        save_path='outputs/charts/model_radar_comparison.png'
    )
    
    print("\n所有图表已保存到 outputs/charts/ 目录")

if __name__ == '__main__':
    main() 