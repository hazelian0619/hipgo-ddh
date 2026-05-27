#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
骨盆关键点检测与角度可视化工具
实现关键点标注、骨盆结构绘制以及医学角度的计算与可视化
"""

import os
import sys
import json
import argparse
import numpy as np
from math import degrees, atan2, acos, sqrt, pi

# Optional plotting deps: keep pure angle math usable without matplotlib.
try:  # pragma: no cover
    import matplotlib  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as patches  # type: ignore
    from matplotlib.path import Path  # type: ignore

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None
    patches = None
    Path = None

# Optional deps: keep angle math usable even if these aren't installed.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    import albumentations as A  # type: ignore
    from albumentations.pytorch import ToTensorV2  # type: ignore
except Exception:  # pragma: no cover
    A = None
    ToTensorV2 = None

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 预测时使用的标准预处理
try:
    from utils.transforms import get_prediction_transforms
except Exception:
    # 可视化/角度计算仍可用；仅模型预测会不可用
    get_prediction_transforms = None

# 尝试导入模型类
try:
    from models.cnn_gat_model import CNN_GAT
except ImportError:
    print("警告: 无法导入CNN_GAT模型，模型预测功能将不可用")
    print("确保models/cnn_gat_model.py在正确位置")

# 髋关节9点关键点定义 - 更新为正确的解剖学定义
KEYPOINT_NAMES = {
    1: "左侧股骨头中心点",      # left_femoral_head_center
    2: "右侧股骨头中心点",      # right_femoral_head_center
    3: "左侧髋臼外缘点",        # left_acetabular_edge
    4: "右侧髋臼外缘点",        # right_acetabular_edge
    5: "耻骨联合点",           # pubic_symphysis
    6: "左侧髋臼荷重面内侧点",  # left_sourcil_medial（用于Tönnis角）
    7: "左侧髋臼荷重面外侧点",  # left_sourcil_lateral（用于Tönnis角）
    8: "右侧髋臼荷重面内侧点",  # right_sourcil_medial（用于Tönnis角）
    9: "右侧髋臼荷重面外侧点"   # right_sourcil_lateral（用于Tönnis角）
}

# 统一配色方案
COLORS = {
    'structure': '#0D47A1',  # 深蓝色 - 骨盆结构实线和辅助线
    'angle': '#E65100',      # 橘色 - 角度弧线和标注文字
}

ANGLE_NAMES = {
    'ce': 'CE',     # 中心边缘角
    'sharp': 'Sharp',   # Sharp角
    'tonnis': 'Tönnis'  # Tönnis角
}

# 简化关键点颜色配置
KEYPOINT_COLORS = {
    'femoral': COLORS['structure'],     # 股骨头中心
    'acetabular': COLORS['structure'], # 髋臼外缘
    'pubic': COLORS['structure'],      # 耻骨联合
    'sourcil': COLORS['structure']     # 荷重面
}

# 角度计算函数
def calculate_angles(keypoints):
    """
    基于解剖学标记点计算骨盆角度
    参数:
        keypoints: 9个关键点的列表，按以下顺序：
            1. 左侧股骨头中心点
            2. 右侧股骨头中心点
            3. 左侧髋臼外缘点
            4. 右侧髋臼外缘点
            5. 耻骨联合点
            6. 左侧髋臼荷重面内侧点
            7. 左侧髋臼荷重面外侧点
            8. 右侧髋臼荷重面内侧点
            9. 右侧髋臼荷重面外侧点
    返回:
        包含左右两侧角度的字典
    """
    # 提取关键点（严格遵循上面 docstring 的 1-9 顺序；0-based 下标为 0-8）
    left_femoral_head = np.array(keypoints[0])
    right_femoral_head = np.array(keypoints[1])
    left_acetabular_edge = np.array(keypoints[2])
    right_acetabular_edge = np.array(keypoints[3])
    pubic_symphysis = np.array(keypoints[4])
    left_sourcil_medial = np.array(keypoints[5])
    left_sourcil_lateral = np.array(keypoints[6])
    right_sourcil_medial = np.array(keypoints[7])
    right_sourcil_lateral = np.array(keypoints[8])
    
    
    # 计算水平参考向量（左 -> 右）
    horizontal_ref = right_femoral_head - left_femoral_head
    # 归一化处理
    horizontal_ref = horizontal_ref / np.linalg.norm(horizontal_ref)
    
    # 计算垂直参考向量（将水平向量逆时针旋转90度，得到“向上”的方向；适配图像坐标 y 向下的情况）
    vertical_ref = np.array([horizontal_ref[1], -horizontal_ref[0]])
    # 归一化处理
    vertical_ref = vertical_ref / np.linalg.norm(vertical_ref)
    
    angles = {}
    
    # 计算CE角 (Center-Edge Angle)
    # 从股骨头中心到髋臼外缘的向量
    # 左侧CE角
    left_ce_vector = left_acetabular_edge - left_femoral_head
    left_ce_angle = angle_between_vectors(vertical_ref, left_ce_vector)
    if left_ce_angle > 90:
        left_ce_angle = 180 - left_ce_angle  # 取互补角确保角度在正确范围
    angles['left_ce_angle'] = left_ce_angle
    
    # 右侧CE角
    right_ce_vector = right_acetabular_edge - right_femoral_head
    right_ce_angle = angle_between_vectors(vertical_ref, right_ce_vector)
    if right_ce_angle > 90:
        right_ce_angle = 180 - right_ce_angle
    angles['right_ce_angle'] = right_ce_angle  # 恢复到原来的计算方式
    
    # 计算Sharp角 (Acetabular Angle)
    # 左侧Sharp角：测量髋臼内侧点到外缘的连线与水平线的夹角
    left_sharp_vector = left_acetabular_edge - left_sourcil_medial
    left_sharp_angle = angle_between_vectors(horizontal_ref, left_sharp_vector)
    if left_sharp_angle > 90:
        left_sharp_angle = 180 - left_sharp_angle  # 确保是锐角
    angles['left_sharp_angle'] = left_sharp_angle
    
    # 右侧Sharp角
    right_sharp_vector = right_acetabular_edge - right_sourcil_medial
    right_sharp_angle = angle_between_vectors(horizontal_ref, right_sharp_vector)
    right_sharp_angle = abs(right_sharp_angle)
    if right_sharp_angle > 90:
        right_sharp_angle = 180 - right_sharp_angle  # 强制补角，确保为锐角
    angles['right_sharp_angle'] = right_sharp_angle
    
    # 计算Tönnis角 (Acetabular Inclination)
    # Tönnis角：髋臼髂部斜面所引的斜形线与水平线形成的夹角
    
    # 左侧Tönnis角
    # 连接第7点（左侧髋臼荷重面外侧点）和第6点（左侧髋臼荷重面内侧点）的线与水平线的夹角
    left_sourcil_line = left_sourcil_lateral - left_sourcil_medial
    left_tonnis_angle = angle_between_vectors(horizontal_ref, left_sourcil_line)
    if left_tonnis_angle > 90:
        left_tonnis_angle = 180 - left_tonnis_angle  # 确保是锐角
    angles['left_tonnis_angle'] = left_tonnis_angle
    
    # 右侧Tönnis角
    # 连接第9点（右侧髋臼荷重面外侧点）和第8点（右侧髋臼荷重面内侧点）的线与水平线的夹角
    right_sourcil_line = right_sourcil_lateral - right_sourcil_medial
    right_tonnis_angle = angle_between_vectors(horizontal_ref, right_sourcil_line)
    if right_tonnis_angle > 90:
        right_tonnis_angle = 180 - right_tonnis_angle  # 确保是锐角
    angles['right_tonnis_angle'] = right_tonnis_angle
    
    return angles

def angle_between_vectors(v1, v2):
    """计算两个向量之间的角度（度数）"""
    # 建议添加方向判断
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)  # 用于判断方向
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    # 根据叉积判断方向
    if cross_product < 0:
        angle_rad = -angle_rad
        
    return np.degrees(angle_rad)

# 骨盆结构线绘制函数（深蓝色实线）
def draw_pelvic_structure(ax, pts):
    """优化骨盆结构线为统一深蓝色并加粗"""
    # 左侧髋臼轮廓
    ax.plot([pts[5][0], pts[6][0]], [pts[5][1], pts[6][1]], '-', color=COLORS['structure'], linewidth=3)
    ax.plot([pts[6][0], pts[2][0]], [pts[6][1], pts[2][1]], '-', color=COLORS['structure'], linewidth=3)
    # 右侧髋臼轮廓
    ax.plot([pts[7][0], pts[8][0]], [pts[7][1], pts[8][1]], '-', color=COLORS['structure'], linewidth=3)
    ax.plot([pts[8][0], pts[3][0]], [pts[8][1], pts[3][1]], '-', color=COLORS['structure'], linewidth=3)
    # 骨盆下部轮廓
    ax.plot([pts[5][0], pts[4][0], pts[7][0]], [pts[5][1], pts[4][1], pts[7][1]], '-', color=COLORS['structure'], linewidth=3)
    # 股骨头中心连线
    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], '-', color=COLORS['structure'], linewidth=2, alpha=0.8)

# 关键点绘制函数（深蓝色实心圆）
def draw_keypoints(ax, pts):
    """绘制关键点（深蓝色实心圆）"""
    for p in pts:
        ax.plot(p[0], p[1], 'o', color=COLORS['structure'], 
               markersize=8, markeredgewidth=0)

# 角度弧线与标注函数
def draw_angle_arc(ax, center, p1, p2, radius, angle_val, angle_name, side, angle_id):
    """只绘制辅助线和角度标注，不绘制橘色弧线"""
    # 计算角度
    a1 = np.arctan2(p1[1]-center[1], center[0]-p1[0])
    a2 = np.arctan2(p2[1]-center[1], p2[0]-center[0])
    
    # 角度ID到字母的映射
    angle_letter_map = {
        ('left', 1): 'a',  # L1 -> a
        ('right', 1): 'b', # R1 -> b
        ('left', 2): 'c',  # L2 -> c
        ('right', 2): 'd', # R2 -> d
        ('left', 3): 'e',  # L3 -> e
        ('right', 3): 'f'  # R3 -> f
    }
    angle_letter = angle_letter_map.get((side, angle_id), '')
    
    # 只绘制辅助线
    if angle_id == 1:  # CE角
        line1_length = radius * 2.5
        line2_length = radius * 2.5
    elif angle_id == 2:  # Sharp角
        line1_length = radius * 3.0
        line2_length = radius * 2.0
    else:  # Tönnis角
        line1_length = radius * 2.0
        line2_length = radius * 3.0
    x1 = center[0] + np.cos(a1) * line1_length
    y1 = center[1] + np.sin(a1) * line1_length
    ax.plot([center[0], x1], [center[1], y1], '--', color=COLORS['structure'], linewidth=1.8, alpha=0.9)
    x2 = center[0] + np.cos(a2) * line2_length
    y2 = center[1] + np.sin(a2) * line2_length
    ax.plot([center[0], x2], [center[1], y2], '--', color=COLORS['structure'], linewidth=1.8, alpha=0.9)
    # 标注位置
    if angle_id == 1:
        text_x = x1
        text_y = y1
    elif angle_id == 2:
        text_x = x1 + (10 if side == 'left' else -10)
        text_y = y1 + (10 if side == 'left' else -10)
    else:
        text_x = x2 + (10 if side == 'left' else -10)
        text_y = y2 + (10 if side == 'left' else -10)
    angle_text = f"{angle_letter}: {abs(angle_val):.1f}°"
    ax.text(text_x, text_y, angle_text, color=COLORS['angle'], fontsize=10, ha='center', va='center', fontweight='bold', family='sans-serif', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

# 主可视化函数
def visualize_keypoints_and_angles(image_path, keypoints, output_path=None, title=None):
    """精细的骨盆关键点与角度可视化"""
    if plt is None:
        raise ImportError("缺少依赖: matplotlib。请先安装后再运行可视化。")
    if cv2 is None:
        raise ImportError("缺少依赖: opencv-python(cv2)。请先安装后再运行可视化。")
    # 设置绘图参数
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Lucida Grande', 'Verdana']
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # 转换关键点坐标
    keypoints_pixel = []
    if np.all(np.array(keypoints) <= 1.0):
        for kp in keypoints:
            x = int(kp[0] * w)
            y = int(kp[1] * h)
            keypoints_pixel.append([x, y])
    else:
        keypoints_pixel = keypoints.copy()
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    # 绘制骨盆结构
    draw_pelvic_structure(ax, keypoints_pixel)
    
    # 绘制关键点
    draw_keypoints(ax, keypoints_pixel)
    
    # 计算角度
    angles = calculate_angles(keypoints_pixel)
    
    # 计算合适的弧线半径（基于图像尺寸比例）
    radius_ce = min(h, w) * 0.022
    radius_sharp = min(h, w) * 0.022
    radius_tonnis = min(h, w) * 0.022
    
    # 绘制左右两侧的角度（注意：X光片中左右是镜像的）
    for side in ['left', 'right']:
        # 确定关键点索引（注意：X光片是镜像的，所以图像左侧是实际右侧，图像右侧是实际左侧）
        if side == 'right':  # 图像左侧（实际是右侧）
            femoral_idx, acetabular_idx = 1, 3      # 右侧
            sourcil_medial_idx, sourcil_lateral_idx = 7, 8
        else:  # 图像右侧（实际是左侧）
            femoral_idx, acetabular_idx = 0, 2      # 左侧
            sourcil_medial_idx, sourcil_lateral_idx = 5, 6
        
        # 1. CE角（中心边缘角）- 从股骨头中心到髋臼外缘的连线与垂直线的夹角
        vertical_point = [keypoints_pixel[femoral_idx][0], 
                         keypoints_pixel[femoral_idx][1] - h*0.2]
        draw_angle_arc(
            ax, 
            keypoints_pixel[femoral_idx],  # 中心点
            vertical_point,                # 起始点（垂直向上）
            keypoints_pixel[acetabular_idx], # 终止点（髋臼外缘）
            radius_ce,                     # 弧线半径
            angles[f'{side}_ce_angle'],    # 角度值
            "CE Angle",                    # 角度名称
            side,                          # 左/右侧
            1                              # 角度编号
        )
        
        # 2. Sharp角（髋臼角）- 髋臼内侧点到外缘的连线与水平线的夹角
        horizontal_point = [keypoints_pixel[sourcil_medial_idx][0] + w*0.2, 
                           keypoints_pixel[sourcil_medial_idx][1]]
        # 添加辅助线：水平线
        ax.plot([keypoints_pixel[sourcil_medial_idx][0], horizontal_point[0]],
                [keypoints_pixel[sourcil_medial_idx][1], horizontal_point[1]],
                '--', color=COLORS['structure'], linewidth=1, alpha=0.8)
        
        draw_angle_arc(
            ax,
            keypoints_pixel[sourcil_medial_idx],  # 中心点
            horizontal_point,                     # 起始点（水平向右）
            keypoints_pixel[acetabular_idx],      # 终止点（髋臼外缘）
            radius_sharp,                        # 弧线半径
            angles[f'{side}_sharp_angle'],       # 角度值
            "Sharp Angle",                       # 角度名称
            side,                                # 左/右侧
            2                                    # 角度编号
        )
        
        # 3. Tönnis角 - 髋臼荷重面连线与水平线的夹角
        horizontal_point2 = [keypoints_pixel[sourcil_medial_idx][0] + w*0.2, 
                           keypoints_pixel[sourcil_medial_idx][1]]
        
        # 添加辅助线：水平线（如果还没绘制）
        if side == 'right' or not side == 'left':
            ax.plot([keypoints_pixel[sourcil_medial_idx][0], horizontal_point2[0]],
                    [keypoints_pixel[sourcil_medial_idx][1], horizontal_point2[1]],
                    '--', color=COLORS['structure'], linewidth=1, alpha=0.8)
        
        draw_angle_arc(
            ax,
            keypoints_pixel[sourcil_medial_idx],  # 中心点
            horizontal_point2,                   # 起始点（水平向右）
            keypoints_pixel[sourcil_lateral_idx], # 终止点（髋臼荷重面外侧点）
            radius_tonnis,                       # 弧线半径
            angles[f'{side}_tonnis_angle'],      # 角度值
            "Tönnis Angle",                      # 角度名称
            side,                                # 左/右侧
            3                                    # 角度编号
        )
    
    # 添加极简图例说明到图像右侧
    legend_text = """
a,b: CE Angle
c,d: Sharp Angle
e,f: Tönnis Angle
"""
    ax.text(0.98, 0.5, legend_text, 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='center', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, pad=0.5))
    
    # 隐藏坐标轴
    ax.axis('off')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    
    plt.tight_layout()
    return fig

def load_keypoints_from_json(json_path):
    """从JSON文件加载关键点"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'shapes' in data:
        # 原始标注格式
        shapes = sorted(data['shapes'], key=lambda x: int(x['label']))
        keypoints = [shape['points'][0] for shape in shapes]
    elif 'keypoints' in data:
        # 自动标注格式
        keypoints = data['keypoints']
    else:
        raise ValueError(f"不支持的JSON格式: {json_path}")
    
    return keypoints

def load_model_and_predict(model_path, image_path):
    """加载模型并预测关键点"""
    try:
        if torch is None or cv2 is None:
            raise ImportError("缺少依赖: torch 或 opencv-python(cv2)，无法进行模型推理。")
        if get_prediction_transforms is None:
            raise ImportError("无法导入 get_prediction_transforms，无法进行模型推理。")

        # 检查CUDA可用性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 添加安全全局变量（PyTorch 2.6兼容性）
        try:
            torch.serialization.add_safe_globals([argparse.Namespace])
        except:
            print("无法添加安全全局变量，可能会影响模型加载")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 加载完整检查点
            model_state = checkpoint['model_state_dict']
            print("已从完整检查点加载模型权重")
        else:
            # 直接加载权重
            model_state = checkpoint
            print("已直接加载模型权重")
        
        # 创建模型实例
        model = CNN_GAT(
            feature_dim=256,
            gat_hidden=128,
            gat_output=64,
            edge_features_dim=32,
            num_keypoints=9,
            num_angles=6,
            num_gat_layers=2,
            num_heads=8,
            dropout=0.1,
            pretrained=True
        ).to(device)
        
        # 加载权重
        model.load_state_dict(model_state)
        model.eval()
        print("模型加载完成")
        
        # 读取和预处理图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用预处理
        transform = get_prediction_transforms()
        transformed = transform(image=image)
        img_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(img_tensor)
            keypoints = outputs['keypoints'].cpu().numpy()[0]
            
        return keypoints
    
    except Exception as e:
        print(f"模型预测失败: {str(e)}")
        print("使用示例关键点进行演示")
        
        # 使用示例关键点
        sample_keypoints = [
            [0.3, 0.35],  # 左侧股骨头中心点
            [0.7, 0.35],  # 右侧股骨头中心点
            [0.35, 0.4],  # 左侧髋臼外缘点
            [0.75, 0.4],  # 右侧髋臼外缘点
            [0.5, 0.6],   # 耻骨联合点
            [0.25, 0.35], # 左侧髋臼荷重面内侧点
            [0.65, 0.35], # 左侧髋臼荷重面外侧点
            [0.65, 0.35], # 右侧髋臼荷重面内侧点
            [0.75, 0.45]   # 右侧髋臼荷重面外侧点
        ]
        return np.array(sample_keypoints)

def normalize_angle(angle):
    """将角度标准化到[0, 90]范围"""
    angle = abs(angle)  # 取绝对值
    if angle > 90:
        angle = 180 - angle
    return angle

def calculate_vector(start_point, end_point):
    """计算从起点到终点的向量"""
    return np.array(end_point) - np.array(start_point)

def calculate_angle(vector1, vector2, reference_vector):
    """计算与参考向量的夹角"""
    angle = angle_between_vectors(vector1, reference_vector)
    return normalize_angle(angle)

def main():
    parser = argparse.ArgumentParser(description='髋关节关键点和角度可视化工具')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--json', type=str, default=None, help='关键点JSON文件路径')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径')
    parser.add_argument('--title', type=str, default=None, help='图像标题')
    
    args = parser.parse_args()
    
    # 检查输入
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图像文件不存在: {args.image}")
    
    # 如果提供JSON文件，从文件加载关键点
    if args.json and os.path.exists(args.json):
        keypoints = load_keypoints_from_json(args.json)
        visualize_keypoints_and_angles(args.image, keypoints, args.output, args.title)
    
    # 如果提供模型，使用模型预测关键点
    elif args.model:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"模型文件不存在: {args.model}")
        
        # 加载模型并预测
        keypoints = load_model_and_predict(args.model, args.image)
        
        # 可视化预测结果
        visualize_keypoints_and_angles(
            args.image, 
            keypoints, 
            args.output, 
            args.title or "模型预测关键点与角度测量"
        )
    
    else:
        raise ValueError("必须提供JSON文件或模型路径")

if __name__ == '__main__':
    main() 
