# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class PerformanceVisualizer:
    def __init__(self, save_dir='figures'):
        self.save_dir = save_dir
        plt.style.use('seaborn')
        
    def plot_keypoint_error_distribution(self, errors):
        """绘制关键点定位误差分布"""
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.axvline(errors.mean(), color='r', linestyle='--', 
                   label='Mean: {:.2f}px'.format(errors.mean()))
        plt.axvline(errors.mean() + 2*errors.std(), color='g', linestyle='--',
                   label='95% CI')
        plt.axvline(errors.mean() - 2*errors.std(), color='g', linestyle='--')
        
        plt.title('Keypoint Localization Error Distribution')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'keypoint_error_dist.png'))
        plt.close()
        
    def plot_angle_bland_altman(self, predictions, ground_truth):
        """绘制Bland-Altman图"""
        mean = (predictions + ground_truth) / 2
        diff = predictions - ground_truth
        
        plt.figure(figsize=(10, 6))
        plt.scatter(mean, diff, alpha=0.5)
        plt.axhline(diff.mean(), color='r', linestyle='--')
        plt.axhline(diff.mean() + 1.96*diff.std(), color='g', linestyle='--')
        plt.axhline(diff.mean() - 1.96*diff.std(), color='g', linestyle='--')
        
        plt.title('Bland-Altman Plot for CE Angle Measurement')
        plt.xlabel('Mean of Measurements (degrees)')
        plt.ylabel('Difference (degrees)')
        plt.savefig(os.path.join(self.save_dir, 'bland_altman.png'))
        plt.close()
        
    def plot_performance_metrics(self, metrics):
        """绘制性能指标雷达图"""
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        plt.title('Model Performance Metrics')
        plt.savefig(os.path.join(self.save_dir, 'performance_radar.png'))
        plt.close()
        
    def plot_inference_time(self, times):
        """绘制推理时间分析"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=times)
        plt.title('Inference Time Distribution')
        plt.ylabel('Time (seconds)')
        plt.savefig(os.path.join(self.save_dir, 'inference_time.png'))
        plt.close()

def generate_performance_report(results):
    """生成完整的性能评估报告"""
    visualizer = PerformanceVisualizer()
    
    # 1. 关键点定位误差分布
    visualizer.plot_keypoint_error_distribution(results['keypoint_errors'])
    
    # 2. CE角度测量Bland-Altman分析
    visualizer.plot_angle_bland_altman(
        results['predicted_angles'],
        results['ground_truth_angles']
    )
    
    # 3. 性能指标雷达图
    metrics = {
        'Keypoint Accuracy': results['keypoint_accuracy'],
        'Angle Accuracy': results['angle_accuracy'],
        'Stability': results['stability'],
        'Speed': results['speed'],
        'Reliability': results['reliability']
    }
    visualizer.plot_performance_metrics(metrics)
    
    # 4. 推理时间分析
    visualizer.plot_inference_time(results['inference_times']) 