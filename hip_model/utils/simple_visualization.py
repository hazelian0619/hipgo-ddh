# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import matplotlib.pyplot as plt
import numpy as np

# 1. 版本迭代数据
versions = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
point_errors = [1500, 300, 131.35, 76.27, 27.17, 48.02, 51.59]
angle_errors = [50, 10, 0, 50.93, 9.80, 1.49, 1.19]
speeds = [1.5, 2.8, 4.0, 7.5, 8.2, 12.0, 14.0]

# 创建图表
plt.figure(figsize=(15, 10))

# 2. 绘制误差趋势
plt.subplot(2, 2, 1)
plt.plot(versions, point_errors, 'b-o', label=u'Keypoint Error (px)')
plt.plot(versions, angle_errors, 'r-o', label=u'Angle Error (deg)')
plt.title(u'Error Trends')
plt.legend()
plt.grid(True)

# 3. 绘制速度提升
plt.subplot(2, 2, 2)
plt.plot(versions, speeds, 'g-o')
plt.title(u'Training Speed Improvement')
plt.ylabel(u'Iterations/s')
plt.grid(True)

# 4. 绘制性能雷达图
metrics = {
    'Point Accuracy': 0.92,
    'Angle Accuracy': 0.95,
    'Inference Speed': 0.88,
    'Stability': 0.90,
    'Reliability': 0.93
}

# 雷达图设置
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
values = list(metrics.values())
values += values[:1]
angles = np.concatenate((angles, [angles[0]]))

ax = plt.subplot(2, 2, 3, projection='polar')
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(list(metrics.keys()))
plt.title(u'Model Performance Metrics')

# 5. 保存图表
plt.tight_layout()
plt.savefig('performance_visualization.png')
print("Visualization saved as: performance_visualization.png")