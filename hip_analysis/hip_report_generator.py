#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
骨盆X光片医学报告生成器
结合MONAI视觉编码和结构化参数，通过多模态LLM生成医学报告
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 尝试导入自定义模块
try:
    from hip_model.visualize_angles import calculate_angles, load_keypoints_from_json, angle_between_vectors
except ImportError:
    print("警告: 无法导入角度计算模块，请确保visualize_angles.py在正确位置")

# MONAI相关导入
import monai
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
    ToTensor
)

# 用于多模态融合
import torch.nn as nn
import torch.nn.functional as F

# 可选: 用于LLM报告生成
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: OpenAI API不可用，将使用基于规则的报告生成")

def generate_report(image_path, json_path=None, output_dir="reports"):
    """主函数：生成医学报告"""
    print(f"正在处理图像: {image_path}")
    
    # 1. 提取结构化参数
    structured_data = extract_structured_data(image_path, json_path)
    
    # 2. 视觉编码
    visual_features = encode_image(image_path)
    
    # 3. 多模态融合
    multimodal_features = fuse_features(visual_features, structured_data)
    
    # 4. 生成报告
    report = generate_medical_report(multimodal_features, structured_data)
    
    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + "_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"报告已生成并保存至: {report_path}")
    return report

def extract_structured_data(image_path, json_path=None):
    """提取结构化参数数据"""
    print("提取结构化参数...")
    
    # 如果提供了JSON文件，直接从中加载关键点和角度
    if json_path and os.path.exists(json_path):
        keypoints = load_keypoints_from_json(json_path)
        # 读取图像获取尺寸
        image = cv2.imread(image_path)
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
        
        # 计算角度
        angles = calculate_angles(keypoints_pixel)
    else:
        # 目前暂不支持直接从图像检测关键点
        print("未提供JSON文件，使用默认值")
        angles = {
            "left_ce_angle": 25.0,
            "right_ce_angle": 28.0,
            "left_sharp_angle": 37.0,
            "right_sharp_angle": 35.0,
            "left_tonnis_angle": 8.0,
            "right_tonnis_angle": 7.0
        }
    
    # 构建结构化数据
    structured_data = {
        "angles": angles,
        "clinical_measures": {
            "CE": {"left": angles["left_ce_angle"], "right": angles["right_ce_angle"]},
            "Sharp": {"left": angles["left_sharp_angle"], "right": angles["right_sharp_angle"]},
            "Tonnis": {"left": angles["left_tonnis_angle"], "right": angles["right_tonnis_angle"]},
        },
        "normal_ranges": {
            "CE": {"min": 25, "max": 40, "description": "Center-Edge角，测量髋臼对股骨头的覆盖程度"},
            "Sharp": {"min": 33, "max": 38, "description": "股骨头中心到髋臼外缘的连线与水平线的夹角"},
            "Tonnis": {"min": 0, "max": 10, "description": "髋臼荷重面与水平线的夹角"}
        },
        "abnormalities": detect_abnormalities(angles)
    }
    
    print("结构化参数提取完成")
    return structured_data

def detect_abnormalities(angles):
    """检测异常情况"""
    abnormalities = []
    
    # CE角异常
    if angles["left_ce_angle"] < 20:
        abnormalities.append({
            "type": "CE角异常",
            "side": "左侧",
            "value": angles["left_ce_angle"],
            "description": "左侧CE角<20°，提示髋臼发育不良"
        })
    if angles["right_ce_angle"] < 20:
        abnormalities.append({
            "type": "CE角异常",
            "side": "右侧",
            "value": angles["right_ce_angle"],
            "description": "右侧CE角<20°，提示髋臼发育不良"
        })
    
    # Sharp角异常
    if angles["left_sharp_angle"] > 45:
        abnormalities.append({
            "type": "Sharp角异常",
            "side": "左侧",
            "value": angles["left_sharp_angle"],
            "description": "左侧Sharp角>45°，提示髋臼覆盖不足"
        })
    if angles["right_sharp_angle"] > 45:
        abnormalities.append({
            "type": "Sharp角异常",
            "side": "右侧",
            "value": angles["right_sharp_angle"],
            "description": "右侧Sharp角>45°，提示髋臼覆盖不足"
        })
    
    # Tönnis角异常
    if angles["left_tonnis_angle"] > 10:
        abnormalities.append({
            "type": "Tönnis角异常",
            "side": "左侧",
            "value": angles["left_tonnis_angle"],
            "description": "左侧Tönnis角>10°，提示髋臼覆盖不足"
        })
    if angles["right_tonnis_angle"] > 10:
        abnormalities.append({
            "type": "Tönnis角异常",
            "side": "右侧",
            "value": angles["right_tonnis_angle"],
            "description": "右侧Tönnis角>10°，提示髋臼覆盖不足"
        })
    
    return abnormalities

def encode_image(image_path):
    """使用MONAI视觉编码器提取图像特征"""
    print("使用MONAI视觉编码器编码图像...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载预训练的DenseNet121模型
    model = DenseNet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=1000,
        pretrained=True
    ).to(device)
    model.eval()
    
    # 使用替代方法直接读取和处理图像
    print("使用替代方法读取图像...")
    try:
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        
        # 转换为torch张量，添加批次维度
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
        print(f"输入张量形状: {input_tensor.shape}")
        
        # 提取特征
        with torch.no_grad():
            # 获取最后一个全连接层之前的特征
            features = model.features(input_tensor)
            # 全局平均池化
            gap = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            # 展平为一维特征向量
            feature_vector = gap.view(gap.size(0), -1).cpu().numpy()
        
        print(f"视觉特征维度: {feature_vector.shape}")
        
        # 将特征向量转换为字典，便于后续处理
        visual_features = {
            "feature_vector": feature_vector,
            "shape": feature_vector.shape,
            "model": "DenseNet121"
        }
        
        return visual_features
        
    except Exception as e:
        print(f"视觉编码失败: {str(e)}")
        print("使用空特征向量...")
        # 返回空特征向量，避免完全失败
        return {
            "feature_vector": np.zeros((1, 1024)),
            "shape": (1, 1024),
            "model": "DenseNet121_fallback"
        }

def fuse_features(visual_features, structured_data):
    """融合视觉特征和结构化参数"""
    print("融合多模态特征...")
    
    # 提取视觉特征向量
    vision_vector = visual_features["feature_vector"].flatten()
    
    # 提取结构化参数并转换为向量
    angles = structured_data["angles"]
    angle_values = np.array([
        angles["left_ce_angle"], 
        angles["right_ce_angle"],
        angles["left_sharp_angle"], 
        angles["right_sharp_angle"],
        angles["left_tonnis_angle"], 
        angles["right_tonnis_angle"]
    ])
    
    # 归一化角度数据
    normal_ranges = structured_data["normal_ranges"]
    normalized_angles = np.zeros_like(angle_values)
    
    # CE角归一化 (25-40°为正常范围)
    normalized_angles[0] = (angle_values[0] - 25) / 15  # 左侧CE角
    normalized_angles[1] = (angle_values[1] - 25) / 15  # 右侧CE角
    
    # Sharp角归一化 (33-38°为正常范围)
    normalized_angles[2] = (angle_values[2] - 33) / 5  # 左侧Sharp角
    normalized_angles[3] = (angle_values[3] - 33) / 5  # 右侧Sharp角
    
    # Tönnis角归一化 (0-10°为正常范围)
    normalized_angles[4] = angle_values[4] / 10  # 左侧Tönnis角
    normalized_angles[5] = angle_values[5] / 10  # 右侧Tönnis角
    
    # 异常指标：检测每个角度是否异常 (0=正常, 1=异常)
    abnormal_indicators = np.zeros(6)
    
    # CE角异常 (<20°为异常)
    abnormal_indicators[0] = 1 if angle_values[0] < 20 else 0
    abnormal_indicators[1] = 1 if angle_values[1] < 20 else 0
    
    # Sharp角异常 (>45°为异常)
    abnormal_indicators[2] = 1 if angle_values[2] > 45 else 0
    abnormal_indicators[3] = 1 if angle_values[3] > 45 else 0
    
    # Tönnis角异常 (>10°为异常)
    abnormal_indicators[4] = 1 if angle_values[4] > 10 else 0
    abnormal_indicators[5] = 1 if angle_values[5] > 10 else 0
    
    # 计算左右侧总体异常分数 (0-3分，每个角度异常加1分)
    left_abnormal_score = abnormal_indicators[0] + abnormal_indicators[2] + abnormal_indicators[4]
    right_abnormal_score = abnormal_indicators[1] + abnormal_indicators[3] + abnormal_indicators[5]
    
    # 创建结构化特征向量，包含原始角度值、归一化角度值、异常指标和总体异常分数
    structured_vector = np.concatenate([
        angle_values,              # 原始角度值 (6)
        normalized_angles,         # 归一化角度值 (6)
        abnormal_indicators,       # 异常指标 (6)
        [left_abnormal_score, right_abnormal_score]  # 左右侧总体异常分数 (2)
    ])
    
    # 创建简单的融合特征（拼接）
    # 注意：在生产环境中，可能需要更复杂的融合方法，如注意力机制
    # 这里简单拼接作为示例
    fused_features = {
        "vision_vector": vision_vector,
        "structured_vector": structured_vector,
        "abnormalities": structured_data["abnormalities"],
        "clinical_measures": structured_data["clinical_measures"],
        "normal_ranges": structured_data["normal_ranges"],
        "abnormal_scores": {
            "left": float(left_abnormal_score),
            "right": float(right_abnormal_score)
        }
    }
    
    print("多模态特征融合完成")
    return fused_features

def generate_medical_report(multimodal_features, structured_data):
    """生成医学报告"""
    print("生成医学报告...")
    
    # 提取结构化数据
    angles = structured_data["angles"]
    abnormalities = structured_data["abnormalities"]
    abnormal_scores = multimodal_features["abnormal_scores"]
    
    # 基本报告信息
    report = "## 骨盆X光片分析报告\n\n"
    report += "### 测量结果\n\n"
    report += f"- 左侧CE角: {angles['left_ce_angle']:.1f}° (正常范围: 25-40°)\n"
    report += f"- 右侧CE角: {angles['right_ce_angle']:.1f}° (正常范围: 25-40°)\n"
    report += f"- 左侧Sharp角: {angles['left_sharp_angle']:.1f}° (正常范围: 33-38°)\n"
    report += f"- 右侧Sharp角: {angles['right_sharp_angle']:.1f}° (正常范围: 33-38°)\n"
    report += f"- 左侧Tönnis角: {angles['left_tonnis_angle']:.1f}° (正常范围: 0-10°)\n"
    report += f"- 右侧Tönnis角: {angles['right_tonnis_angle']:.1f}° (正常范围: 0-10°)\n\n"
    
    if abnormalities:
        report += "### 异常发现\n\n"
        for abnormality in abnormalities:
            report += f"- {abnormality['description']}\n"
        report += "\n"
    else:
        report += "### 异常发现\n\n无明显异常\n\n"
    
    # 尝试使用LLM生成诊断建议（如果可用）
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            # 设置API密钥
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            
            # 构建提示词
            prompt = generate_llm_prompt(angles, abnormalities, abnormal_scores)
            
            # 调用OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一位专业的骨科放射科医生，专注于髋关节发育不良(DDH)的诊断。提供准确、简洁的医学报告和建议。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            # 提取生成的文本
            llm_report = response.choices[0].message.content.strip()
            
            # 添加到报告中
            report += "### 诊断评估与建议\n\n"
            report += llm_report + "\n"
            
        except Exception as e:
            print(f"LLM报告生成失败: {str(e)}")
            # 使用基于规则的报告作为后备
            rule_based_report = generate_rule_based_report(angles, abnormalities, abnormal_scores)
            report += "### 诊断评估与建议\n\n"
            report += rule_based_report + "\n"
    else:
        # 使用基于规则的报告生成
        rule_based_report = generate_rule_based_report(angles, abnormalities, abnormal_scores)
        report += "### 诊断评估与建议\n\n"
        report += rule_based_report + "\n"
    
    return report

def generate_llm_prompt(angles, abnormalities, abnormal_scores):
    """生成LLM提示词"""
    prompt = f"""根据以下骨盆X光片测量结果，生成专业的发育性髋关节发育不良(DDH)诊断评估、严重程度分级和治疗建议：

测量结果：
- 左侧CE角: {angles['left_ce_angle']:.1f}° (正常范围: 25-40°)
- 右侧CE角: {angles['right_ce_angle']:.1f}° (正常范围: 25-40°)
- 左侧Sharp角: {angles['left_sharp_angle']:.1f}° (正常范围: 33-38°)
- 右侧Sharp角: {angles['right_sharp_angle']:.1f}° (正常范围: 33-38°)
- 左侧Tönnis角: {angles['left_tonnis_angle']:.1f}° (正常范围: 0-10°)
- 右侧Tönnis角: {angles['right_tonnis_angle']:.1f}° (正常范围: 0-10°)

异常评分：
- 左侧异常评分: {abnormal_scores['left']}/3
- 右侧异常评分: {abnormal_scores['right']}/3

"""
    
    if abnormalities:
        prompt += "发现的异常：\n"
        for abnormality in abnormalities:
            prompt += f"- {abnormality['description']}\n"
    else:
        prompt += "无明显异常发现。\n"
    
    prompt += """
请分析以上数据，提供：
1. DDH诊断评估（是否存在，左右侧情况）
2. 严重程度分级（轻度/中度/重度）
3. 治疗建议
4. 可能的预后

请使用专业但易懂的语言，简明扼要。
"""
    
    return prompt

def generate_rule_based_report(angles, abnormalities, abnormal_scores):
    """生成基于规则的诊断建议"""
    left_score = abnormal_scores["left"]
    right_score = abnormal_scores["right"]
    
    # 初始化诊断报告
    report = ""
    
    # 判断是否存在DDH
    if left_score == 0 and right_score == 0:
        report += "诊断评估：骨盆X光片检查未见明显发育性髋关节发育不良（DDH）的影像学证据。所有测量角度均在正常范围内。\n\n"
        report += "严重程度：无明显异常。\n\n"
        report += "治疗建议：无需特殊干预，建议正常随访检查。\n\n"
        report += "预后评估：预期良好，继续保持正常活动。\n"
        
    else:
        # 诊断评估
        report += "诊断评估：骨盆X光片检查显示"
        
        if left_score > 0 and right_score > 0:
            report += "双侧"
        elif left_score > 0:
            report += "左侧"
        else:
            report += "右侧"
        
        report += "发育性髋关节发育不良（DDH）的影像学证据。"
        
        # 详细说明异常角度
        if abnormalities:
            report += " 具体表现为："
            for abnormality in abnormalities:
                report += f"{abnormality['description']}；"
            report = report.rstrip("；") + "。"
            
        report += "\n\n"
        
        # 严重程度评估
        report += "严重程度：\n"
        
        if left_score > 0:
            if left_score == 1:
                left_severity = "轻度"
            elif left_score == 2:
                left_severity = "中度"
            else:
                left_severity = "重度"
            report += f"- 左侧：{left_severity}（异常评分：{left_score}/3）\n"
            
        if right_score > 0:
            if right_score == 1:
                right_severity = "轻度"
            elif right_score == 2:
                right_severity = "中度"
            else:
                right_severity = "重度"
            report += f"- 右侧：{right_severity}（异常评分：{right_score}/3）\n"
            
        report += "\n"
        
        # 治疗建议
        max_score = max(left_score, right_score)
        report += "治疗建议：\n"
        
        if max_score == 1:
            report += "- 定期随访观察（每3-6个月复查X光片）\n"
            report += "- 考虑物理疗法和运动康复\n"
            report += "- 避免过度负重活动\n"
        elif max_score == 2:
            report += "- 建议骨科专科会诊\n"
            report += "- 考虑矫形器治疗\n"
            report += "- 制定个性化康复计划\n"
            report += "- 定期随访（每2-3个月复查）\n"
        else:
            report += "- 建议立即转诊至专科骨科医生\n"
            report += "- 可能需要考虑手术干预\n"
            report += "- 术前详细评估和全面影像学检查\n"
            report += "- 密切随访（每1-2个月复查）\n"
            
        report += "\n"
        
        # 预后评估
        report += "预后评估：\n"
        
        if max_score == 1:
            report += "预期良好。早期发现并适当干预的轻度DDH通常有良好预后。建议保持定期随访，观察病情进展。"
        elif max_score == 2:
            report += "预期谨慎乐观。中度DDH通过适当治疗可以获得显著改善，但需要坚持治疗方案并定期评估疗效。"
        else:
            report += "预期复杂。重度DDH可能需要多学科协作治疗，并可能存在长期并发症风险，包括早发性骨关节炎。及时干预对改善预后至关重要。"
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="骨盆X光片医学报告生成器")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--json", type=str, default=None, help="关键点JSON文件路径")
    parser.add_argument("--output", type=str, default="reports", help="输出报告目录")
    
    args = parser.parse_args()
    generate_report(args.image, args.json, args.output) 
