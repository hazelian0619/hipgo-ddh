#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载LLaVA-Med多模态医学大语言模型
用于骨盆X光片分析与医学报告生成
"""

import os
import argparse
import torch
import logging
from transformers import AutoProcessor, AutoModelForCausalLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(model_name, output_dir, use_cache=True):
    """
    下载并保存LLaVA-Med模型
    
    参数:
        model_name: HuggingFace模型名称
        output_dir: 输出目录
        use_cache: 是否使用缓存
    """
    logger.info(f"开始下载模型: {model_name}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 下载并保存处理器
        logger.info("下载处理器...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir="./cache" if use_cache else None
        )
        processor.save_pretrained(output_dir)
        logger.info(f"处理器已保存至: {output_dir}")
        
        # 下载并保存模型
        logger.info("下载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            cache_dir="./cache" if use_cache else None
        )
        model.save_pretrained(output_dir)
        logger.info(f"模型已保存至: {output_dir}")
        
        logger.info("模型下载完成！")
        return True
        
    except Exception as e:
        logger.error(f"模型下载失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='下载LLaVA-Med多模态医学大语言模型')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default="liuhaotian/llava-med-v1.0-7b",
        help='HuggingFace模型名称'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="models/llava_med",
        help='模型输出目录'
    )
    
    parser.add_argument(
        '--no_cache', 
        action='store_true',
        help='禁用HuggingFace缓存'
    )
    
    args = parser.parse_args()
    
    # 下载模型
    success = download_model(
        model_name=args.model,
        output_dir=args.output_dir,
        use_cache=not args.no_cache
    )
    
    if success:
        print("\n" + "=" * 50)
        print(f"LLaVA-Med模型已成功下载至: {args.output_dir}")
        print("可以使用以下命令生成医学报告:")
        print(f"python mllm_medical_report.py --image path/to/image.jpg --keypoints path/to/keypoints.json --model {args.output_dir}")
        print("=" * 50)
    else:
        print("\n下载失败，请检查网络连接或手动下载模型。")

if __name__ == "__main__":
    main() 