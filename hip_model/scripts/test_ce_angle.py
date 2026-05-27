import json
import matplotlib.pyplot as plt
import numpy as np
from utils.metrics import calculate_bilateral_ce_angles

def draw_ce_angle(ax, center_point, edge_point, is_left=True):
    """优化CE角度可视化，更专业的医学影像风格"""
    # 1. 绘制垂直参考线
    height = 250  # 更长的参考线
    vert_x = center_point[0]
    vert_y = np.array([center_point[1]-height, center_point[1]+height])
    ax.plot([vert_x, vert_x], vert_y, '--', color='white', alpha=0.8, 
            linewidth=1.5, dashes=(5, 5))  # 更精细的虚线
    
    # 2. 绘制CE连线
    ax.plot([center_point[0], edge_point[0]], 
            [center_point[1], edge_point[1]], 
            '-', color='#00FF00', alpha=1.0, linewidth=2)  # 医学常用的绿色
    
    # 3. 绘制角度弧线
    radius = 100  # 更大的弧线
    dx = edge_point[0] - center_point[0]
    dy = edge_point[1] - center_point[1]
    angle = np.degrees(np.arctan2(-dy, dx))
    
    # 绘制双层弧线
    theta1 = 90
    theta2 = angle
    if theta2 < theta1:
        theta1, theta2 = theta2, theta1
    
    theta = np.linspace(theta1, theta2, 100)
    # 内层弧线
    x = center_point[0] + radius * np.cos(np.radians(theta))
    y = center_point[1] - radius * np.sin(np.radians(theta))
    ax.plot(x, y, color='#FF3366', alpha=1.0, linewidth=2)  # 醒目的红色
    
    # 外层装饰弧线
    x_outer = center_point[0] + (radius+5) * np.cos(np.radians(theta))
    y_outer = center_point[1] - (radius+5) * np.sin(np.radians(theta))
    ax.plot(x_outer, y_outer, color='#FF3366', alpha=0.3, linewidth=1)
    
    # 4. 标注角度值
    mid_theta = (theta1 + theta2) / 2
    text_x = center_point[0] + radius * 1.8 * np.cos(np.radians(mid_theta))
    text_y = center_point[1] - radius * 1.8 * np.sin(np.radians(mid_theta))
    side = "Right" if is_left else "Left"  # 镜像效果，调换左右
    
    # 专业的标注框
    ax.text(text_x, text_y, f'{side} CE: {abs(theta2-theta1):.1f}°', 
            color='cyan', fontsize=12, fontweight='bold', 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3))
    
    # 5. 添加刻度线
    tick_length = 10
    for theta_tick in np.linspace(theta1, theta2, 5):
        x_start = center_point[0] + radius * np.cos(np.radians(theta_tick))
        y_start = center_point[1] - radius * np.sin(np.radians(theta_tick))
        x_end = center_point[0] + (radius+tick_length) * np.cos(np.radians(theta_tick))
        y_end = center_point[1] - (radius+tick_length) * np.sin(np.radians(theta_tick))
        ax.plot([x_start, x_end], [y_start, y_end], 
                color='#FF3366', alpha=0.8, linewidth=1)

def visualize_ce_angles(image, points, save_path=None):
    """优化的CE角度可视化函数"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. 显示X光图像
    ax.imshow(image, cmap='gray')
    
    # 2. 绘制关键点
    colors = ['red', 'blue', 'green', 'yellow']
    labels = ['左中心', '右中心', '左外缘', '右外缘']
    for i, (point, color, label) in enumerate(zip(points, colors, labels)):
        ax.plot(point[0], point[1], 'o', color=color, markersize=8, label=label)
    
    # 3. 计算并绘制CE角度
    left_angle, right_angle = calculate_bilateral_ce_angles(points)
    
    # 4. 添加标注
    plt.title('CE角度测量结果', fontsize=14)
    plt.text(10, 30, f'左CE角: {left_angle:.1f}°', color='white', fontsize=12)
    plt.text(10, 60, f'右CE角: {right_angle:.1f}°', color='white', fontsize=12)
    
    # 5. 美化设置
    plt.legend(loc='upper right')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def test_angle_calculation():
    """测试CE角度计算和可视化"""
    try:
        print("开始测试...")
        # 1. 加载标注数据
        with open('labeled_data/train/annotations/xray_001.json', 'r') as f:
            data = json.load(f)
        
        # 2. 提取4个关键点
        points = []
        for shape in data['shapes']:
            point = shape['points'][0]
            points.append(point)
        
        # 3. 计算双侧CE角度
        left_angle, right_angle = calculate_bilateral_ce_angles(points)
        
        print("\n=== CE角度计算结果 ===")
        print(f"左侧CE角度: {left_angle:.2f}°")
        print(f"右侧CE角度: {right_angle:.2f}°")
        
        # 4. 可视化结果
        img = plt.imread('labeled_data/train/images/xray_001.jpg')
        visualize_ce_angles(img, points)
        
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    test_angle_calculation()