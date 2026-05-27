#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
髋关节X光片分析一致性检测
确保使用0506最佳模型进行分析
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 项目根目录
sys.path.append(project_root)

# 正确导入0506最佳模型及其他需要的模块
try:
    # 使用model_loader加载0506最佳模型，而不是直接导入CNN_GAT类
    from hip_model.utils.model_loader import load_best_model
    from hip_model.dataset import get_transforms
    from hip_analysis.mllm_medical_report import HipMLLM
    from hip_model.visualize_angles import calculate_angles
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已正确安装所有依赖项")

def check_consistency(image_path, doctor_keypoints_path=None, output_dir="consistency_results"):
    """
    检查模型预测与医生标注的一致性
    
    参数:
        image_path: X光片图像路径
        doctor_keypoints_path: 医生标注的关键点JSON文件路径（如果有）
        output_dir: 输出目录
    
    返回:
        consistency_result: 一致性分析结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n{'='*50}")
    print(f"分析图像: {image_path}")
    print(f"{'='*50}")
    
    # 1. 加载0506最佳模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_best_model(device=device)  # 使用loader加载0506模型
    
    # 2. 读取和预处理图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 应用预处理
    transform = get_transforms(train=False)
    transformed = transform(image=image_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # 3. 进行预测
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_keypoints = outputs['keypoints'].cpu().numpy()[0]
    
    # 4. 计算模型预测的角度
    predicted_angles = calculate_angles(predicted_keypoints)
    print("\n模型预测的角度值:")
    for key, value in predicted_angles.items():
        print(f"{key}: {value:.2f}°")
    
    # 5. 如果有医生标注，加载并计算一致性
    if doctor_keypoints_path and os.path.exists(doctor_keypoints_path):
        with open(doctor_keypoints_path, 'r') as f:
            doctor_data = json.load(f)
            
        # 确定医生标注的格式并提取关键点
        if 'keypoints' in doctor_data:
            doctor_keypoints = np.array(doctor_data['keypoints'])
        elif 'shapes' in doctor_data:
            shapes = sorted(doctor_data['shapes'], key=lambda x: int(x['label']))
            doctor_keypoints = np.array([shape['points'][0] for shape in shapes])
        else:
            raise ValueError(f"不支持的JSON格式: {doctor_keypoints_path}")
        
        # 计算医生标注的角度
        doctor_angles = calculate_angles(doctor_keypoints)
        print("\n医生标注的角度值:")
        for key, value in doctor_angles.items():
            print(f"{key}: {value:.2f}°")
        
        # 计算角度差异
        print("\n角度差异(医生 - 模型):")
        angle_diffs = {}
        for key in predicted_angles.keys():
            diff = abs(doctor_angles[key] - predicted_angles[key])
            angle_diffs[key] = diff
            print(f"{key}: {diff:.2f}°")
        
        # 计算关键点位置差异
        keypoint_diffs = np.sqrt(np.sum((doctor_keypoints - predicted_keypoints)**2, axis=1))
        print("\n关键点位置差异(欧几里得距离):")
        for i, diff in enumerate(keypoint_diffs):
            print(f"关键点 {i+1}: {diff:.4f}")
        
        # 生成一致性分析报告
        report_path = os.path.join(output_dir, f"{base_name}_consistency_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"图像: {image_path}\n")
            f.write(f"医生标注: {doctor_keypoints_path}\n\n")
            
            f.write("模型预测的角度值:\n")
            for key, value in predicted_angles.items():
                f.write(f"{key}: {value:.2f}°\n")
            
            f.write("\n医生标注的角度值:\n")
            for key, value in doctor_angles.items():
                f.write(f"{key}: {value:.2f}°\n")
            
            f.write("\n角度差异(医生 - 模型):\n")
            for key, diff in angle_diffs.items():
                f.write(f"{key}: {diff:.2f}°\n")
            
            f.write("\n关键点位置差异(欧几里得距离):\n")
            for i, diff in enumerate(keypoint_diffs):
                f.write(f"关键点 {i+1}: {diff:.4f}\n")
            
            # 一致性结论
            avg_angle_diff = sum(angle_diffs.values()) / len(angle_diffs)
            avg_keypoint_diff = sum(keypoint_diffs) / len(keypoint_diffs)
            
            f.write("\n一致性结论:\n")
            f.write(f"平均角度差异: {avg_angle_diff:.2f}°\n")
            f.write(f"平均关键点差异: {avg_keypoint_diff:.4f}\n")
            
            if avg_angle_diff < 3.0 and avg_keypoint_diff < 0.05:
                f.write("结论: 高度一致\n")
            elif avg_angle_diff < 5.0 and avg_keypoint_diff < 0.1:
                f.write("结论: 中度一致\n")
            else:
                f.write("结论: 低度一致\n")
        
        print(f"\n一致性分析报告已保存至: {report_path}")
        
        return {
            "model_angles": predicted_angles,
            "doctor_angles": doctor_angles,
            "angle_diffs": angle_diffs,
            "keypoint_diffs": keypoint_diffs.tolist(),
            "avg_angle_diff": avg_angle_diff,
            "avg_keypoint_diff": avg_keypoint_diff
        }
    
    else:
        # 如果没有医生标注，只保存模型预测结果
        predicted_keypoints_path = os.path.join(output_dir, f"{base_name}_predicted_keypoints.json")
        with open(predicted_keypoints_path, 'w') as f:
            json.dump({"keypoints": predicted_keypoints.tolist()}, f, indent=4)
            
        print(f"\n模型预测的关键点已保存至: {predicted_keypoints_path}")
        
        return {"model_angles": predicted_angles}

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='髋关节X光片分析一致性检测(使用0506最佳模型)')
    parser.add_argument('--image', type=str, required=True, help='X光片图像路径')
    parser.add_argument('--doctor', type=str, help='医生标注的关键点JSON文件路径')
    parser.add_argument('--output', type=str, default='consistency_results', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        result = check_consistency(args.image, args.doctor, args.output)
        print("\n一致性检测完成!")
    except Exception as e:
        print(f"一致性检测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 