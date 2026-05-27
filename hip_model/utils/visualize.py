import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_predictions(image, boxes, angles, save_path=None):
    """可视化预测结果"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # 绘制每个预测框和角度
    for box, angle in zip(boxes, angles):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x, y, f'{angle:.1f}°', color='r')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_curves(losses, metrics, save_path=None):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    # 绘制评估指标曲线
    plt.subplot(1, 2, 2)
    for name, values in metrics.items():
        plt.plot(values, label=name)
    plt.title('Evaluation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 