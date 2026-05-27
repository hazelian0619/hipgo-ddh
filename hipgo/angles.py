"""DDH角度计算 — 基于9个解剖关键点的几何推导

关键点（0-indexed）:
  0: 左股骨头圆心      1: 右股骨头圆心
  2: 左髋臼外缘         3: 右髋臼外缘
  4: 耻骨联合上缘中点
  5: 左髋臼荷重面内侧   6: 左髋臼荷重面外侧
  7: 右髋臼荷重面内侧   8: 右髋臼荷重面外侧

参考系（解剖坐标系）:
  水平参考 = 股骨头连线 (点1 - 点0)
  垂直参考 = 水平参考逆时针旋转90°

DDH临床参考范围:
  CE角    正常>25°  边界20-25°  DDH<20°
  Sharp角  正常<45°  边界45-50°  DDH>50°
  Tonnis角 正常<10°  边界10-15°  DDH>15°
"""
import numpy as np


def calculate_angles(keypoints):
    """从9个归一化关键点计算6个DDH角度

    Args:
        keypoints: shape (9, 2), 归一化坐标 (x, y ∈ [0, 1])

    Returns:
        dict: {'left_ce': float, 'right_ce': float,
               'left_sharp': float, 'right_sharp': float,
               'left_tonnis': float, 'right_tonnis': float}  单位: 度
    """
    kps = np.array(keypoints)
    h_ref = kps[1] - kps[0]
    h_ref = h_ref / np.linalg.norm(h_ref)
    v_ref = np.array([h_ref[1], -h_ref[0]])
    v_ref = v_ref / np.linalg.norm(v_ref)

    def _angle(v1, v2):
        d = np.dot(v1, v2)
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return round(float(np.degrees(np.arccos(np.clip(d / n, -1, 1)))), 1)

    def _acute(v1, v2):
        raw = _angle(v1, v2)
        return min(raw, 180 - raw)

    return {
        'left_ce':     _angle(v_ref, kps[2] - kps[0]),
        'right_ce':    _angle(v_ref, kps[3] - kps[1]),
        'left_sharp':  _acute(h_ref, kps[2] - kps[5]),
        'right_sharp': _acute(h_ref, kps[3] - kps[7]),
        'left_tonnis':  _angle(v_ref, kps[6] - kps[5]),
        'right_tonnis': _angle(v_ref, kps[8] - kps[7]),
    }


def diagnose(angles, ce_thr=25, sharp_thr=45, tonnis_thr=10):
    """基于角度值的DDH诊断

    Args:
        angles: calculate_angles() 的返回值
        ce_thr: CE角阈值 (临床: 25°)
        sharp_thr: Sharp角阈值 (临床: 45°)
        tonnis_thr: Tonnis角阈值 (临床: 10°)

    Returns:
        '双1'(双侧DDH) / '单1'(单侧DDH) / '双0'(正常)
    """
    issues = sum([
        angles['left_ce'] < ce_thr,    angles['right_ce'] < ce_thr,
        angles['left_sharp'] > sharp_thr, angles['right_sharp'] > sharp_thr,
        angles['left_tonnis'] > tonnis_thr, angles['right_tonnis'] > tonnis_thr,
    ])
    if issues >= 3:
        return '双1'
    if issues >= 1:
        return '单1'
    return '双0'
