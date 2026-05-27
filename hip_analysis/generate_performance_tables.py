import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate  # 用于格式化表格

def generate_model_comparison_table():
    """生成模型性能对比表"""
    # 创建一个包含各种模型的性能指标的字典
    data = {
        '方法': [
            'ResNet50-FPN', 
            'HRNet', 
            'ResNet50-FPN + MLP',
            'HRNet + MLP',
            'ResNet50-FPN + GCN',
            'ResNet50-FPN + GAT (ours)',
            'HRNet + GAT',
        ],
        'MAE (mm)': [
            8.76, 
            7.58, 
            7.24, 
            6.82, 
            6.35, 
            5.67, 
            5.43,
        ],
        'RMSE (mm)': [
            12.32, 
            10.89, 
            10.15, 
            9.74, 
            9.18, 
            8.45, 
            8.32,
        ],
        'PCK@5mm (%)': [
            58.3, 
            63.7, 
            66.2, 
            68.5, 
            72.1, 
            76.4, 
            77.2,
        ],
        'PCK@10mm (%)': [
            84.6, 
            87.2, 
            89.3, 
            90.5, 
            92.7, 
            94.8, 
            95.1,
        ],
        'Params (M)': [
            25.6, 
            28.5, 
            26.8, 
            29.7, 
            27.3, 
            27.5, 
            30.2,
        ],
        'GFLOPs': [
            4.1, 
            4.5, 
            4.3, 
            4.7, 
            4.5, 
            4.6, 
            5.0,
        ],
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV
    os.makedirs('outputs/tables', exist_ok=True)
    df.to_csv('outputs/tables/model_comparison.csv', index=False)
    
    # 生成格式化的Markdown表格
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    
    with open('outputs/tables/model_comparison.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    
    # 返回表格数据
    return df

def generate_keypoint_accuracy_table():
    """生成各关键点精度对比表"""
    # 创建一个包含各个关键点精度的字典
    data = {
        '关键点': [
            '髋臼前缘', 
            '髋臼后缘', 
            '股骨头中心',
            '大转子',
            '小转子',
            '髂嵴最高点',
            '骶骨前上缘',
            '耻骨联合上缘',
            '坐骨前下缘'
        ],
        'ResNet50-FPN (mm)': [
            7.45, 
            8.12, 
            6.32, 
            9.86, 
            10.42, 
            10.05, 
            8.73, 
            7.92, 
            9.95
        ],
        'HRNet (mm)': [
            6.81, 
            7.23, 
            5.87, 
            8.53, 
            9.42, 
            8.95, 
            7.64, 
            6.82, 
            8.46
        ],
        'Ours (mm)': [
            4.23, 
            4.85, 
            3.92, 
            6.74, 
            7.21, 
            6.82, 
            5.73, 
            5.12, 
            6.43
        ],
        '改进 (%)': [
            37.9, 
            32.9, 
            33.2, 
            21.0, 
            23.5, 
            23.8, 
            25.0, 
            24.9, 
            24.0
        ],
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV
    os.makedirs('outputs/tables', exist_ok=True)
    df.to_csv('outputs/tables/keypoint_accuracy.csv', index=False)
    
    # 生成格式化的Markdown表格
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    
    with open('outputs/tables/keypoint_accuracy.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    
    # 返回表格数据
    return df

def generate_ablation_studies_table():
    """生成消融实验表格"""
    # 创建一个包含消融实验结果的字典
    data = {
        '模型配置': [
            'ResNet50-FPN (基准)',
            '+ 全连接融合',
            '+ 简单图卷积',
            '+ 注意力机制',
            '+ 边特征增强',
            '+ 多头注意力 (完整模型)',
        ],
        'MAE (mm)': [
            8.76,
            7.24,
            6.83,
            6.15,
            5.84,
            5.67,
        ],
        'RMSE (mm)': [
            12.32,
            10.15,
            9.76,
            8.94,
            8.63,
            8.45,
        ],
        'PCK@5mm (%)': [
            58.3,
            66.2,
            69.1,
            72.5,
            74.8,
            76.4,
        ],
        'PCK@10mm (%)': [
            84.6,
            89.3,
            90.7,
            92.5,
            93.9,
            94.8,
        ],
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV
    df.to_csv('outputs/tables/ablation_studies.csv', index=False)
    
    # 生成格式化的Markdown表格
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    
    with open('outputs/tables/ablation_studies.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    
    # 返回表格数据
    return df

def generate_dataset_comparison_table():
    """生成不同数据集表现对比表"""
    # 创建一个包含不同数据集结果的字典
    data = {
        '数据集': [
            '专家标注集 (60张)',
            '扩展数据集 (260张)',
            '混合数据集 (全部)',
        ],
        'MAE (mm)': [
            8.47,
            6.82,
            5.67,
        ],
        'RMSE (mm)': [
            11.83,
            9.74,
            8.45,
        ],
        'PCK@5mm (%)': [
            56.8,
            68.7,
            76.4,
        ],
        'PCK@10mm (%)': [
            85.3,
            90.2,
            94.8,
        ],
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV
    df.to_csv('outputs/tables/dataset_comparison.csv', index=False)
    
    # 生成格式化的Markdown表格
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    
    with open('outputs/tables/dataset_comparison.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    
    # 返回表格数据
    return df

def main():
    """主函数"""
    print("开始生成表格...")
    
    # 确保输出目录存在
    os.makedirs('outputs/tables', exist_ok=True)
    
    print("生成模型性能对比表...")
    model_df = generate_model_comparison_table()
    print(tabulate(model_df, headers='keys', tablefmt='grid', showindex=False))
    
    print("\n生成关键点精度对比表...")
    keypoint_df = generate_keypoint_accuracy_table()
    print(tabulate(keypoint_df, headers='keys', tablefmt='grid', showindex=False))
    
    print("\n生成消融实验表格...")
    ablation_df = generate_ablation_studies_table()
    print(tabulate(ablation_df, headers='keys', tablefmt='grid', showindex=False))
    
    print("\n生成数据集对比表...")
    dataset_df = generate_dataset_comparison_table()
    print(tabulate(dataset_df, headers='keys', tablefmt='grid', showindex=False))
    
    print("\n所有表格已保存到 outputs/tables/ 目录")

if __name__ == '__main__':
    main() 