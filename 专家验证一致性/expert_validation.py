#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专家验证一致性评估脚本
用于评估模型诊断结果与专家标签的一致性
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "hip_model"))
sys.path.append(os.path.join(project_root, "hip_analysis"))

# 导入项目模块
try:
    # 使用model_loader加载0506最佳模型，而不是直接导入CNN_GAT类
    from hip_model.utils.model_loader import load_best_model
    from hip_model.dataset import get_transforms
    from hip_analysis.mllm_medical_report import HipMLLM
    from hip_model.visualize_angles import calculate_angles  # 添加这一行，导入优化后的角度计算函数

except ImportError as e:
    print(f"错误: 无法导入必要模块: {str(e)}")
    print("请确保hip_model和hip_analysis模块可导入")
    sys.exit(1)

def detect_keypoints(image_path, model, device):
    """
    检测图像中的关键点
    
    Args:
        image_path: 图像路径
        model: 加载好的CNN-GAT模型
        device: 设备(CPU或GPU)
    
    Returns:
        预测的关键点
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(train=False, img_size=512)
    
    # 转换图像
    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # 预测关键点
    with torch.no_grad():
        predictions = model(image_tensor)
        pred_keypoints = predictions['keypoints'][0].cpu().numpy()
    
    return pred_keypoints

def extract_diagnosis_from_report(report_text):
    """
    从报告文本中提取诊断结果
    
    Args:
        report_text: 报告文本
    
    Returns:
        诊断结果字典，包含问题类型、侧别等信息
    """
    # 初始化诊断字典
    diagnosis = {
        "has_issue": False,        # 是否有问题
        "side": "none",            # 问题侧别: "left", "right", "bilateral", "none"
        "condition": "normal",     # 问题类型: "DDH", "normal", "other"
        "confidence": "high"       # 确信度: "high", "medium", "low"
    }
    
    # 尝试从结构化输出中提取诊断结果
    if report_text and "===结构化诊断结果===" in report_text:
        try:
            # 提取结构化部分
            start_idx = report_text.find("===结构化诊断结果===")
            end_idx = report_text.find("====================", start_idx)
            if end_idx == -1:  # 如果找不到结束标记，则取到文本结尾
                structured_text = report_text[start_idx:]
            else:
                structured_text = report_text[start_idx:end_idx + 20]
            
            # 解析诊断标签
            if "诊断标签:" in structured_text:
                label_line = [line for line in structured_text.split('\n') if "诊断标签:" in line][0]
                label_match = label_line.split(":", 1)[1].strip()
                
                if "双0" in label_match:
                    diagnosis["label"] = "双0"
                    diagnosis["has_issue"] = False
                    diagnosis["side"] = "bilateral"
                    diagnosis["condition"] = "normal"
                elif "双1" in label_match:
                    diagnosis["label"] = "双1"
                    diagnosis["has_issue"] = True
                    diagnosis["side"] = "bilateral"
                    diagnosis["condition"] = "DDH"  # 默认为DDH，可能会在后续逻辑修正
                elif "双2" in label_match:
                    diagnosis["label"] = "双2"
                    diagnosis["has_issue"] = None
                    diagnosis["side"] = "bilateral"
                    diagnosis["condition"] = "uncertain"
                    diagnosis["confidence"] = "low"
                elif "单1" in label_match:
                    diagnosis["label"] = "单1"
                    diagnosis["has_issue"] = True
                    # 侧别需要从问题行解析
                
                # 解析侧别
                if "侧别:" in structured_text:
                    side_line = [line for line in structured_text.split('\n') if "侧别:" in line][0]
                    side_match = side_line.split(":", 1)[1].strip()
                    
                    if "左侧" in side_match:
                        diagnosis["side"] = "left"
                    elif "右侧" in side_match:
                        diagnosis["side"] = "right"
                    elif "双侧" in side_match:
                        diagnosis["side"] = "bilateral"
                
                # 解析问题类型
                if "类型:" in structured_text:
                    type_line = [line for line in structured_text.split('\n') if "类型:" in line][0]
                    type_match = type_line.split(":", 1)[1].strip()
                    
                    if "DDH" in type_match:
                        diagnosis["condition"] = "DDH"
                    elif "正常" in type_match:
                        diagnosis["condition"] = "normal"
                    elif "其他" in type_match:
                        diagnosis["condition"] = "other"
                
                # 已成功解析结构化数据，直接返回
                return diagnosis
            
        except Exception as e:
            print(f"解析结构化诊断结果失败: {str(e)}，回退到传统解析方法")
            # 解析失败，回退到传统方法
            pass
    
    # 传统解析方法（作为备选）
    # 转换为小写以便查找
    report_lower = report_text.lower()
    
    # 检查是否有问题
    problem_indicators = ["异常", "不足", "发育不良", "问题", "ddh", "髋臼发育不良"]
    normal_indicators = ["正常", "无异常", "无明显异常", "未见异常"]
    
    # 确定是否有问题
    has_problem = any(indicator in report_lower for indicator in problem_indicators)
    is_normal = any(indicator in report_lower for indicator in normal_indicators)
    
    if has_problem and not is_normal:
        diagnosis["has_issue"] = True
    elif is_normal and not has_problem:
        diagnosis["has_issue"] = False
    else:
        # 可能有矛盾信息或不确定，设为不确定
        if "建议进一步检查" in report_lower or "需要随访" in report_lower:
            diagnosis["has_issue"] = None  # 不确定是否有问题
            diagnosis["confidence"] = "low"  # 低确信度
    
    # 确定侧别
    if "双侧" in report_lower or "两侧" in report_lower:
        diagnosis["side"] = "bilateral"
    elif "左侧" in report_lower and "异常" in report_lower[report_lower.find("左侧")-10:report_lower.find("左侧")+10]:
        diagnosis["side"] = "left"
    elif "右侧" in report_lower and "异常" in report_lower[report_lower.find("右侧")-10:report_lower.find("右侧")+10]:
        diagnosis["side"] = "right"
    
    # 确定问题类型
    if "髋臼发育不良" in report_lower or "ddh" in report_lower:
        diagnosis["condition"] = "DDH"
    elif diagnosis["has_issue"]:
        diagnosis["condition"] = "other"
    
    # 映射到医生标签体系
    if diagnosis["has_issue"] is None or diagnosis["confidence"] == "low":
        diagnosis["label"] = "双2"  # 不确定
    elif diagnosis["has_issue"]:
        if diagnosis["side"] == "bilateral":
            diagnosis["label"] = "双1"  # 双侧有问题
        elif diagnosis["side"] in ["left", "right"]:
            diagnosis["label"] = "单1"  # 单侧有问题
        else:
            diagnosis["label"] = "双2"  # 侧别不明确，视为不确定
    else:
        diagnosis["label"] = "双0"  # 双侧无问题
    
    return diagnosis

def parse_doctor_labels(expert_dir):
    """
    解析医生标签
    
    Args:
        expert_dir: 医生标签目录
    
    Returns:
        医生标签字典，图像ID为键
    """
    doctor_labels = {}
    
    # 解析双侧无问题(双0)
    bilateral_normal_dir = os.path.join(expert_dir, "双0")
    if os.path.exists(bilateral_normal_dir):
        for img_file in os.listdir(bilateral_normal_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_id = os.path.splitext(img_file)[0]
                doctor_labels[img_id] = {
                    "has_issue": False,
                    "side": "bilateral",
                    "condition": "normal",
                    "label": "双0"
                }
    
    # 解析双侧有问题(双1)
    bilateral_issue_dir = os.path.join(expert_dir, "双1")
    if os.path.exists(bilateral_issue_dir):
        # 可能有DDH等子目录
        for subdir in os.listdir(bilateral_issue_dir):
            subdir_path = os.path.join(bilateral_issue_dir, subdir)
            if os.path.isdir(subdir_path):
                condition = subdir
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_id = os.path.splitext(img_file)[0]
                        doctor_labels[img_id] = {
                            "has_issue": True,
                            "side": "bilateral",
                            "condition": condition,
                            "label": "双1"
                        }
    
    # 解析双侧不确定(双2)
    bilateral_uncertain_dir = os.path.join(expert_dir, "双2")
    if os.path.exists(bilateral_uncertain_dir):
        for img_file in os.listdir(bilateral_uncertain_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_id = os.path.splitext(img_file)[0]
                doctor_labels[img_id] = {
                    "has_issue": None,  # 不确定
                    "side": "bilateral",
                    "condition": "uncertain",
                    "label": "双2"
                }
    
    # 解析单侧有问题(单1)
    unilateral_issue_dir = os.path.join(expert_dir, "单1")
    if os.path.exists(unilateral_issue_dir):
        for img_file in os.listdir(unilateral_issue_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_id = os.path.splitext(img_file)[0]
                doctor_labels[img_id] = {
                    "has_issue": True,
                    "side": "unilateral",  # 具体左右侧需进一步分析
                    "condition": "other",
                    "label": "单1"
                }
    
    print(f"成功解析{len(doctor_labels)}个医生标签")
    return doctor_labels

def calculate_agreement_metrics(predictions, doctor_labels):
    """计算一致性指标"""
    total = 0
    matched = 0
    label_counts = {"双0": 0, "双1": 0, "双2": 0, "单1": 0}  # 添加标签计数
    
    # 匹配的样本
    matches = []
    mismatches = []
    
    for img_id, pred in predictions.items():
        if img_id in doctor_labels:
            total += 1
            doctor_label = doctor_labels[img_id]
            
            # 统计各类标签数量
            if doctor_label["label"] in label_counts:
                label_counts[doctor_label["label"]] += 1
            
            # 直接使用标签进行匹配
            if "label" in pred and "label" in doctor_label:
                label_match = pred["label"] == doctor_label["label"]
                if label_match:
                    matched += 1
                    matches.append({
                        "image_id": img_id,
                        "prediction": pred,
                        "doctor_label": doctor_label
                    })
                else:
                    mismatches.append({
                        "image_id": img_id,
                        "prediction": pred,
                        "doctor_label": doctor_label
                    })
            else:
                # 如果没有label字段，使用原有的匹配逻辑
                # （保留原有逻辑作为备选，确保代码健壮性）
                issue_match = pred["has_issue"] == doctor_label["has_issue"]
                side_match = pred["side"] == doctor_label["side"]
                overall_match = issue_match and side_match
                
                if overall_match:
                    matched += 1
                    matches.append({
                        "image_id": img_id,
                        "prediction": pred,
                        "doctor_label": doctor_label
                    })
                else:
                    mismatches.append({
                        "image_id": img_id,
                        "prediction": pred,
                        "doctor_label": doctor_label
                    })
    
    # 计算整体一致率
    overall_agreement = matched / total if total > 0 else 0
    
    # 按标签类型计算一致率
    label_agreements = {}
    for label in label_counts:
        label_total = label_counts[label]
        if label_total > 0:
            label_matches = sum(1 for m in matches if m["doctor_label"]["label"] == label)
            label_agreements[label] = label_matches / label_total
        else:
            label_agreements[label] = 0
    
    metrics = {
        "total_samples": total,
        "matched_samples": matched,
        "overall_agreement": overall_agreement,
        "label_counts": label_counts,
        "label_agreements": label_agreements,
        "matches": matches,
        "mismatches": mismatches
    }
    
    return metrics

def run_expert_validation(args):
    """运行专家验证"""
    print(f"{'='*80}")
    print(f"开始专家验证一致性评估")
    print(f"验证数据集: {args.validation_data}")
    print(f"医生标签目录: {args.expert_labels}")
    print(f"模型路径: {args.model_path}")
    print(f"{'='*80}")
    
    # 验证是否使用最佳模型
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 检查是否为0506最佳模型
    model_filename = os.path.basename(args.model_path)
    if "0506" not in model_filename and "best" not in model_filename:
        print(f"警告: 可能未使用0506最佳模型，当前模型: {model_filename}")
        proceed = input("是否继续? [y/n]: ")
        if proceed.lower() != 'y':
            print("已取消验证")
            return
    else:
        print(f"确认使用最佳模型: {model_filename}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载CNN-GAT模型
    print("加载CNN-GAT模型...")
    try:
        # 直接使用load_best_model函数加载0506最佳模型，无需额外参数
        model = load_best_model(device=device)
        print("CNN-GAT最佳模型(0506)加载成功")
    except Exception as e:
        print(f"CNN-GAT模型加载失败: {str(e)}")
        return
    
    # 加载医学大语言模型
    print("加载医学大语言模型...")
    try:
        mllm = HipMLLM(
            model_path=args.llava_path,
            device=device
        )
        print("医学大语言模型加载成功")
    except Exception as e:
        print(f"医学大语言模型加载失败: {str(e)}")
        return
    
    # 解析医生标签
    print("解析医生标签...")
    doctor_labels = parse_doctor_labels(args.expert_labels)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    keypoints_dir = os.path.join(args.output_dir, "keypoints")
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(keypoints_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 获取验证数据集中的图像
    image_files = [f for f in os.listdir(args.validation_data) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到{len(image_files)}张图像进行验证")
    
    # 批量处理图像（修复缩进）
    predictions = {}
    
    for img_file in tqdm(image_files, desc="处理图像"):
        img_path = os.path.join(args.validation_data, img_file)
        img_id = os.path.splitext(img_file)[0]
        
        try:
            # 1. 检测关键点
            keypoints = detect_keypoints(img_path, model, device)
            
            # 保存关键点
            keypoints_path = os.path.join(keypoints_dir, f"{img_id}_keypoints.json")
            with open(keypoints_path, 'w') as f:
                json.dump(keypoints.tolist(), f, indent=2)
            
            # 计算角度
            angles = calculate_angles(keypoints)
            angles_path = os.path.join(keypoints_dir, f"{img_id}_angles.json")
            with open(angles_path, 'w') as f:
                json.dump(angles, f, indent=2)
            
            # 2. 生成报告
            report = None
            if mllm is not None:
                try:
                    # 使用角度数据生成报告
                    report = mllm.generate_report(
                        image_path=img_path,
                        keypoints_path=keypoints_path,
                        angles_data=angles,
                        output_dir=reports_dir
                    )
                except TypeError:
                    # 如果不接受angles_data参数，退回到原始调用
                    report = mllm.generate_report(
                        image_path=img_path,
                        keypoints_path=keypoints_path,
                        output_dir=reports_dir
                    )
                
                # 保存报告
                report_path = os.path.join(reports_dir, f"{img_id}_report.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
            
            # 3. 提取诊断结果
            diagnosis = extract_diagnosis_from_report(report)
            diagnosis["angles"] = angles  # 添加角度信息到诊断结果
            
            # 4. 保存预测结果
            predictions[img_id] = diagnosis
            
        except Exception as e:
            print(f"处理图像{img_file}时出错: {str(e)}")
            continue
    
    # 计算一致性指标
    print("计算一致性指标...")
    metrics = calculate_agreement_metrics(predictions, doctor_labels)
    
    # 输出结果
    print(f"\n{'='*80}")
    print(f"专家验证一致性评估结果")
    print(f"{'='*80}")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"匹配样本数: {metrics['matched_samples']}")
    print(f"整体一致率: {metrics['overall_agreement']:.4f}")
    
    # 保存结果
    results_path = os.path.join(args.output_dir, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "metrics": {k: v for k, v in metrics.items() if k not in ['matches', 'mismatches']},
            "predictions": predictions
        }, f, indent=2)
    
    print(f"结果已保存至: {results_path}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="专家验证一致性评估")
    parser.add_argument('--validation_data', type=str, required=True, help='验证数据集目录')
    parser.add_argument('--expert_labels', type=str, required=True, help='医生标签目录')
    parser.add_argument('--model_path', type=str, required=True, help='CNN-GAT模型路径')
    parser.add_argument('--llava_path', type=str, default="hip_model/models/llava_med", help='LLaVA-Med模型路径')
    parser.add_argument('--output_dir', type=str, default="validation_results", help='输出目录')
    
    args = parser.parse_args()
    run_expert_validation(args)

if __name__ == "__main__":
    main()

