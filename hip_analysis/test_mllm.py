#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态医学大语言模型测试脚本
用于验证骨盆X光片分析与医学报告生成功能
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入多模态医学大语言模型
try:
    from mllm_medical_report import HipMLLM
except ImportError:
    print("错误: 无法导入HipMLLM类，请确保mllm_medical_report.py文件存在")
    sys.exit(1)

def test_single_image(image_path, keypoints_path=None, model_path="models/llava_med", output_dir="reports", device=None, low_resource=False, prompt=None, temperature=0.7, max_new_tokens=1024):
    """测试单张图像的报告生成"""
    print(f"\n{'='*50}")
    print(f"测试图像: {image_path}")
    print(f"关键点文件: {keypoints_path if keypoints_path else '无'}")
    print(f"模型路径: {model_path}")
    print(f"使用自定义提示词: {'是' if prompt else '否'}")
    print(f"{'='*50}\n")
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return False
    
    # 如果提供了关键点文件，检查是否存在
    if keypoints_path and not os.path.exists(keypoints_path):
        print(f"警告: 关键点文件不存在: {keypoints_path}")
        keypoints_path = None
    
    try:
        # 初始化模型
        start_time = time.time()
        print("初始化多模态医学大语言模型...")
        mllm = HipMLLM(
            model_path=model_path,
            device=device,
            low_resource=low_resource
        )
        init_time = time.time() - start_time
        print(f"模型初始化完成，耗时: {init_time:.2f}秒")
        
        # 生成报告
        print("\n开始生成医学报告...")
        report_start_time = time.time()
        report = mllm.generate_report(
            image_path=image_path,
            keypoints_path=keypoints_path,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            output_dir=output_dir
        )
        report_time = time.time() - report_start_time
        
        # 显示结果
        print(f"\n医学报告生成完成，耗时: {report_time:.2f}秒")
        print(f"\n{'='*50}")
        print("生成的医学报告:")
        print(f"{'='*50}")
        print(report)
        print(f"{'='*50}")
        
        report_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + "_mllm_report.txt")
        print(f"\n报告已保存至: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"错误: 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_batch(image_dir, num_images=3, model_path="models/llava_med", output_dir="reports", device=None, low_resource=False, prompt=None, temperature=0.7, max_new_tokens=1024):
    """批量测试多张图像"""
    print(f"\n{'='*50}")
    print(f"批量测试 - 目录: {image_dir}, 图像数量: {num_images}")
    print(f"使用自定义提示词: {'是' if prompt else '否'}")
    print(f"{'='*50}\n")
    
    # 获取图像列表
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    if not image_files:
        print(f"错误: 未在目录中找到图像文件: {image_dir}")
        return False
    
    # 限制测试图像数量
    image_files = image_files[:num_images]
    print(f"将测试以下{len(image_files)}张图像:")
    for i, img in enumerate(image_files, 1):
        print(f"{i}. {img}")
    
    # 初始化模型(只初始化一次)
    start_time = time.time()
    print("\n初始化多模态医学大语言模型...")
    try:
        mllm = HipMLLM(
            model_path=model_path,
            device=device,
            low_resource=low_resource
        )
        init_time = time.time() - start_time
        print(f"模型初始化完成，耗时: {init_time:.2f}秒")
    except Exception as e:
        print(f"错误: 模型初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 逐个处理图像
    results = []
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_file)
        print(f"\n[{i}/{len(image_files)}] 处理图像: {img_file}")
        
        # 检查是否有对应的JSON文件
        json_file = img_file.replace('.jpg', '.json').replace('.png', '.json')
        json_path = os.path.join(image_dir, json_file)
        
        if os.path.exists(json_path):
            print(f"找到关键点文件: {json_file}")
        else:
            json_path = None
            print("未找到关键点文件，将直接分析图像")
        
        # 生成报告
        try:
            print("\n开始生成医学报告...")
            report_start_time = time.time()
            report = mllm.generate_report(
                image_path=img_path,
                keypoints_path=json_path,
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                output_dir=output_dir
            )
            report_time = time.time() - report_start_time
            
            print(f"医学报告生成完成，耗时: {report_time:.2f}秒")
            report_path = os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + "_mllm_report.txt")
            
            # 保存结果
            results.append({
                "image": img_file,
                "success": True,
                "time": report_time,
                "report_path": report_path
            })
            
        except Exception as e:
            print(f"错误: 处理图像失败: {str(e)}")
            results.append({
                "image": img_file,
                "success": False,
                "error": str(e)
            })
    
    # 输出统计结果
    print(f"\n{'='*50}")
    print(f"批量测试结果统计:")
    print(f"{'='*50}")
    success_count = sum(1 for r in results if r["success"])
    print(f"总共处理: {len(results)}张图像")
    print(f"成功生成: {success_count}张报告")
    print(f"失败数量: {len(results) - success_count}张图像")
    
    if success_count > 0:
        avg_time = sum(r["time"] for r in results if r["success"]) / success_count
        print(f"平均处理时间: {avg_time:.2f}秒/图像")
    
    print(f"{'='*50}")
    return success_count > 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态医学大语言模型测试脚本')
    parser.add_argument('--image', type=str, help='单张X光片图像路径')
    parser.add_argument('--keypoints', type=str, help='关键点JSON文件路径')
    parser.add_argument('--dir', type=str, help='图像目录(用于批量测试)')
    parser.add_argument('--num', type=int, default=3, help='批量测试的图像数量')
    parser.add_argument('--model', type=str, default="models/llava_med", help='模型路径')
    parser.add_argument('--output', type=str, default="reports", help='报告输出目录')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='设备类型')
    parser.add_argument('--low_resource', action='store_true', help='低资源模式')
    parser.add_argument('--prompt', type=str, help='自定义提示词，如果不指定则使用默认提示词')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度，控制随机性（0表示无随机性）')
    parser.add_argument('--max_tokens', type=int, default=1024, help='生成的最大token数量')
    
    args = parser.parse_args()
    
    # 检查必要的输入
    if not args.image and not args.dir:
        parser.print_help()
        print("\n错误: 必须提供--image或--dir参数")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 运行测试
    if args.image:
        # 单张图像测试
        test_single_image(
            image_path=args.image,
            keypoints_path=args.keypoints,
            model_path=args.model,
            output_dir=args.output,
            device=args.device,
            low_resource=args.low_resource,
            prompt=args.prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
    else:
        # 批量测试
        test_batch(
            image_dir=args.dir,
            num_images=args.num,
            model_path=args.model,
            output_dir=args.output,
            device=args.device,
            low_resource=args.low_resource,
            prompt=args.prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 