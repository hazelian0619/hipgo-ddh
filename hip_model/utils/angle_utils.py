import numpy as np

def compute_angle(v1, v2):
    """计算两个向量之间的夹角
    
    Args:
        v1: 第一个向量，形状为 [x, y]
        v2: 第二个向量，形状为 [x, y]
        
    Returns:
        angle: 两个向量间的夹角（以度为单位，0-180范围内）
    """
    # 1. 检查向量有效性
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        print(f"Warning: 无效的输入向量: v1={v1}, v2={v2}")
        return 0.0
        
    # 2. 向量标准化
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    # 避免除以零
    if v1_norm < 1e-10 or v2_norm < 1e-10:
        print("Warning: 向量长度接近零")
        return 0.0
        
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm
    
    # 3. 计算角度
    dot_product = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    
    # 4. 结果验证
    if np.isnan(angle):
        print("Warning: 角度计算结果为NaN")
        return 0.0
        
    return angle

def calculate_bilateral_ce_angles(points):
    """计算双侧CE角度
    
    Args:
        points: 包含左右股骨头中心和髋臼外缘点的坐标
               [left_center, right_center, left_edge, right_edge]
    
    Returns:
        left_angle: 左侧CE角度
        right_angle: 右侧CE角度
    """
    # 垂直参考向量（向上为负）
    vertical = np.array([0, -1])
    
    # 左侧CE角度
    left_vector = np.array([
        points[2][0] - points[0][0],  # 左髋臼外缘x - 左股骨头中心x
        points[2][1] - points[0][1]   # 左髋臼外缘y - 左股骨头中心y
    ])
    left_angle = compute_angle(vertical, left_vector)
    
    # 右侧CE角度
    right_vector = np.array([
        points[3][0] - points[1][0],  # 右髋臼外缘x - 右股骨头中心x
        points[3][1] - points[1][1]   # 右髋臼外缘y - 右股骨头中心y
    ])
    right_angle = compute_angle(vertical, right_vector)
    
    return left_angle, right_angle

def calculate_bilateral_sharp_angles(points):
    """计算双侧Sharp角度
    
    Args:
        points: 包含耻骨联合点、左右髋臼外缘点的坐标
               [pubic_point, left_edge, right_edge]
    
    Returns:
        left_angle: 左侧Sharp角度
        right_angle: 右侧Sharp角度
    """
    # 水平参考向量
    horizontal = np.array([1, 0])
    
    # 左侧Sharp角度
    left_vector = np.array([
        points[1][0] - points[0][0],  # 左髋臼外缘x - 耻骨联合x
        points[1][1] - points[0][1]   # 左髋臼外缘y - 耻骨联合y
    ])
    left_angle = compute_angle(horizontal, left_vector)
    
    # 右侧Sharp角度
    right_vector = np.array([
        points[2][0] - points[0][0],  # 右髋臼外缘x - 耻骨联合x
        points[2][1] - points[0][1]   # 右髋臼外缘y - 耻骨联合y
    ])
    right_angle = compute_angle(horizontal, right_vector)
    
    return left_angle, right_angle

def calculate_bilateral_tonnis_angles(points):
    """计算双侧Tönnis角度
    
    Args:
        points: 包含髋臼荷重面内外侧点的坐标
               [left_medial, left_lateral, right_medial, right_lateral]
    
    Returns:
        left_angle: 左侧Tönnis角度
        right_angle: 右侧Tönnis角度
    """
    # 水平参考向量
    horizontal = np.array([1, 0])
    
    # 左侧Tönnis角度
    left_vector = np.array([
        points[1][0] - points[0][0],  # 左荷重面外侧x - 左荷重面内侧x
        points[1][1] - points[0][1]   # 左荷重面外侧y - 左荷重面内侧y
    ])
    left_angle = compute_angle(horizontal, left_vector)
    
    # 右侧Tönnis角度
    right_vector = np.array([
        points[3][0] - points[2][0],  # 右荷重面外侧x - 右荷重面内侧x
        points[3][1] - points[2][1]   # 右荷重面外侧y - 右荷重面内侧y
    ])
    right_angle = compute_angle(horizontal, right_vector)
    
    return left_angle, right_angle 