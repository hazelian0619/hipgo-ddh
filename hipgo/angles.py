"""髋关节角度计算 — 基于9个解剖关键点的几何推导

关键点定义（0-indexed）:
  0: 左侧股骨头圆心      1: 右侧股骨头圆心
  2: 左侧髋臼外缘         3: 右侧髋臼外缘
  4: 耻骨联合上缘中点
  5: 左侧髋臼荷重面内侧   6: 左侧髋臼荷重面外侧
  7: 右侧髋臼荷重面内侧   8: 右侧髋臼荷重面外侧

参考系:
  - 水平参考 = 股骨头连线 (point[1] - point[0])，不受骨盆倾斜影响
  - 垂直参考 = 水平参考逆时针旋转90°

DDH临床标准:
  CE角:    >25°正常, 20-25°边界发育不良, <20°明确DDH
  Sharp角:  <45°正常, 45-50°边界,         >50°异常
  Tönnis角: <10°正常, 10-15°边界,         >15°异常
"""
import numpy as np


def calculate_angles(keypoints):
    """从9个关键点坐标计算6个DDH角度

    Args:
        keypoints: shape [9, 2], 归一化坐标 (x, y ∈ [0, 1])

    Returns:
        dict: 6个角度值（度）
    """
    kps = np.array(keypoints)

    # 解剖参考系
    h_ref = kps[1] - kps[0]                      # 股骨头连线 = 水平参考
    h_ref = h_ref / np.linalg.norm(h_ref)
    v_ref = np.array([h_ref[1], -h_ref[0]])       # 垂直参考（旋转90°）
    v_ref = v_ref / np.linalg.norm(v_ref)

    def angle(v1, v2):
        d = np.dot(v1, v2)
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return round(float(np.degrees(np.arccos(np.clip(d / n, -1, 1)))), 1)

    def acute_angle(v1, v2):
        raw = angle(v1, v2)
        return min(raw, 180 - raw)

    return {
        # CE角（Center-Edge）：股骨头中心 → 髋臼外缘 vs 垂直线
        'left_ce':     angle(v_ref, kps[2] - kps[0]),
        'right_ce':    angle(v_ref, kps[3] - kps[1]),

        # Sharp角：荷重面内侧 → 髋臼外缘 vs 水平线（取锐角）
        'left_sharp':  acute_angle(h_ref, kps[2] - kps[5]),
        'right_sharp': acute_angle(h_ref, kps[3] - kps[7]),

        # Tönnis角（髋臼顶倾斜角）：荷重面内侧 → 外侧 vs 垂直线
        'left_tonnis':  angle(v_ref, kps[6] - kps[5]),
        'right_tonnis': angle(v_ref, kps[8] - kps[7]),
    }


def diagnose(angles, ce_threshold=25, sharp_threshold=45, tonnis_threshold=10):
    """基于角度值的DDH诊断规则

    Args:
        angles: calculate_angles() 的输出
        ce_threshold: CE角阈值（临床标准 25°）
        sharp_threshold: Sharp角阈值（临床标准 45°）
        tonnis_threshold: Tönnis角阈值（临床标准 10°）

    Returns:
        str: '双1'(双侧DDH), '单1'(单侧DDH), '双0'(正常)
    """
    issues = sum([
        angles['left_ce'] < ce_threshold,   angles['right_ce'] < ce_threshold,
        angles['left_sharp'] > sharp_threshold, angles['right_sharp'] > sharp_threshold,
        angles['left_tonnis'] > tonnis_threshold, angles['right_tonnis'] > tonnis_threshold,
    ])
    if issues >= 3:
        return '双1'
    elif issues >= 1:
        return '单1'
    else:
        return '双0'
