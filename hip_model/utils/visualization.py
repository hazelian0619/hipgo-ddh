import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 创建目录
os.makedirs('figures', exist_ok=True)

# 设置样式
plt.style.use('seaborn')
sns.set_theme()

# 训练历史数据
train_losses = [0.2497, 0.1983, 0.0587, 0.0161, 0.0054, 0.0038]
val_losses = [0.1291, 0.0769, 0.0106, 0.0085, 0.0015]
epochs = range(1, len(train_losses) + 1)

# 1. 绘制训练历史
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs[:-1], val_losses, 'r-', label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('figures/training_history.png')
plt.close()

# 2. 绘制误差分布
point_errors = [51.59, 48.02, 27.17, 76.27, 131.35]
angle_errors = [1.19, 1.49, 9.80, 50.93]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(point_errors, ax=ax1, bins=10)
ax1.set_title('Point Error Distribution')
ax1.set_xlabel('Error (pixels)')

sns.histplot(angle_errors, ax=ax2, bins=10)
ax2.set_title('Angle Error Distribution')
ax2.set_xlabel('Error (degrees)')
plt.savefig('figures/error_distribution.png')
plt.close()

# 3. 绘制性能雷达图
metrics = {
    '点位精度': 51.59,
    '角度精度': 1.19,
    '推理速度': 0.19,
    'GPU利用率': 85,
    '模型大小': 366
}

categories = list(metrics.keys())
values = list(metrics.values())

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
values = np.concatenate((values, [values[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Model Performance Metrics')
plt.savefig('figures/performance_radar.png')
plt.close()

print("可视化图表已生成在 figures/ 目录下")
