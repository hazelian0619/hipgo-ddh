# -*- coding: utf-8 -*-
from utils.version_visualization import VersionProgressVisualizer, version_data
from utils.performance_visualization import PerformanceVisualizer
import numpy as np
import os

def main():
    # 确保输出目录存在
    os.makedirs('figures', exist_ok=True)
    
    # 1. 版本进度可视化
    version_visualizer = VersionProgressVisualizer()
    version_visualizer.plot_training_progress(version_data)
    version_visualizer.plot_error_trends(version_data)
    version_visualizer.plot_speed_improvement(version_data)
    
    # 2. 性能评估可视化
    performance_data = {
        'keypoint_errors': np.array([51.59, 48.02, 49.15, 50.23, 47.89]),
        'predicted_angles': np.array([25.3, 26.1, 24.8, 25.5, 25.9]),
        'ground_truth_angles': np.array([25.0, 26.0, 25.0, 25.0, 26.0]),
        'keypoint_accuracy': 0.92,
        'angle_accuracy': 0.95,
        'stability': 0.90,
        'speed': 0.88,
        'reliability': 0.93,
        'inference_times': np.array([0.19, 0.18, 0.20, 0.17, 0.21])
    }
    
    performance_visualizer = PerformanceVisualizer()
    performance_visualizer.plot_keypoint_error_distribution(performance_data['keypoint_errors'])
    performance_visualizer.plot_angle_bland_altman(
        performance_data['predicted_angles'],
        performance_data['ground_truth_angles']
    )
    performance_visualizer.plot_performance_metrics({
        'Keypoint Accuracy': performance_data['keypoint_accuracy'],
        'Angle Accuracy': performance_data['angle_accuracy'],
        'Stability': performance_data['stability'],
        'Speed': performance_data['speed'],
        'Reliability': performance_data['reliability']
    })
    performance_visualizer.plot_inference_time(performance_data['inference_times'])
    
    print("\n可视化图表已生成在 figures/ 目录下：")
    print("版本进度图表：")
    print("1. training_progress.png - 训练进度对比")
    print("2. error_trends.png - 误差趋势")
    print("3. speed_improvement.png - 速度提升")
    print("\n性能评估图表：")
    print("4. keypoint_error_dist.png - 关键点定位误差分布")
    print("5. bland_altman.png - CE角度测量Bland-Altman分析")
    print("6. performance_radar.png - 模型整体性能雷达图")
    print("7. inference_time.png - 推理时间分布")

if __name__ == "__main__":
    main()