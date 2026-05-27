#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
骨盆X光片多模态医学大语言模型
基于LLaVA-Med实现的医学报告生成系统
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 在文件顶部导入部分添加
try:
    from llava.model.builder import load_pretrained_model
    from llava.model import LlavaMistralForCausalLM  # 添加这一行
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    LLAVA_AVAILABLE = True
except ImportError:
    print("警告: 无法导入LLaVA模块，将使用AutoModelForCausalLM")
    LLAVA_AVAILABLE = False




# 尝试导入角度计算模块
try:
    from hip_model.visualize_angles import calculate_angles, load_keypoints_from_json, angle_between_vectors
except ImportError:
    print("警告: 无法导入角度计算模块，请确保visualize_angles.py在正确位置")

# 导入LLaVA-Med相关模块
from transformers import AutoProcessor, AutoModelForCausalLM
import torch.nn.functional as F

class HipMLLM:
    """骨盆X光片多模态医学大语言模型"""
    
    def __init__(
        self, 
        model_path: str = "liuhaotian/llava-med-v1.0-7b", 
        device: str = None,
        low_resource: bool = False
    ):
        """
        初始化多模态医学大语言模型
        
        参数:
            model_path: 模型路径或Hugging Face模型名称
            device: 设备类型('cuda', 'cpu', 'mps')，如果为None则自动检测
            low_resource: 是否在低资源模式下运行(低显存设备)
        """
        self.model_path = model_path
        
        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 低资源模式设置
        self.low_resource = low_resource
        self.dtype = torch.float16 if not low_resource and self.device != "cpu" else torch.float32
        
        # 加载模型和处理器
        self._load_model()
        
    def _load_model(self):
        """加载LLaVA-Med模型和处理器"""
        print(f"加载多模态医学大语言模型: {self.model_path}")
    
        try:
            # 1. 检测模型类型
            from transformers import AutoConfig, AutoTokenizer
            
            config = AutoConfig.from_pretrained(self.model_path)
            model_type = getattr(config, "model_type", "")
            print(f"检测到模型类型: {model_type}")
            
            # 2. 加载tokenizer (通用部分)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.processor = self.tokenizer
            
            # 3. 根据模型类型选择加载器
            if "llava_mistral" in model_type:
                # Mistral专用加载
                try:
                    # 尝试从llava包导入
                    from llava.model import LlavaMistralForCausalLM
                    self.model = LlavaMistralForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=self.dtype
                    ).to(self.device)
                    print("使用LlavaMistralForCausalLM加载成功")
                except ImportError:
                    # 如果导入失败，使用通用加载器
                    print("无法导入LlavaMistralForCausalLM，使用通用加载器")
                    from transformers import AutoModelForCausalLM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=self.dtype
                    ).to(self.device)
                    print("使用AutoModelForCausalLM加载成功")
            else:
                # 通用加载
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=self.dtype
                ).to(self.device)
                print("使用AutoModelForCausalLM加载成功")
                
        except Exception as e:
            print(f"LLaVA-Med模型加载失败: {str(e)}")
            print("请确保已正确安装LLaVA并下载模型文件")
            raise RuntimeError(f"无法加载LLaVA-Med模型: {str(e)}")
    
    
    def generate_report(
        self,
        image_path: str,
        keypoints_path: Optional[str] = None,
        angles_data: Optional[dict] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        output_dir: Optional[str] = None
    ) -> str:
        """生成骨盆X光片医学报告"""
        
        # 处理prompt为None的情况
        if prompt is None:
            # 提取结构化数据，优先使用传入的角度数据
            if angles_data is not None:
                # 使用已经计算好的角度数据
                structured_data = {
                    "angles": angles_data,
                    "normal_ranges": {
                        "CE": {"min": 25, "max": 40, "description": "Center-Edge角，测量髋臼对股骨头的覆盖程度"},
                        "Sharp": {"min": 33, "max": 38, "description": "股骨头中心到髋臼外缘的连线与水平线的夹角"},
                        "Tonnis": {"min": 0, "max": 10, "description": "髋臼荷重面与水平线的夹角"}
                    }
                }
            else:
                # 如果没有传入角度数据，从关键点计算
                structured_data = self._extract_structured_data(image_path, keypoints_path)
            
            angles_str = self._format_angles_for_prompt(structured_data)
            prompt = self._build_default_prompt(angles_str)
        
        # 读取图像
        image = Image.open(image_path).convert('RGB')    

        # 尝试使用llava专用的处理方式
        try:
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import process_images, tokenizer_image_token
            
            # 设置对话模板
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], f"<image>\n{prompt}")
            conv.append_message(conv.roles[1], None)

            # 处理图像 - 使用llava.mm_utils中的process_images函数
            image_tensor = process_images(
                [image], 
                image_size=336,  # 使用固定大小
                device=self.device
            )

            # 准备输入
            input_ids = tokenizer_image_token(
                self.tokenizer, 
                conv.get_prompt(), 
                [image_tensor], 
                self.model.config.mm_use_im_start_end if hasattr(self.model.config, "mm_use_im_start_end") else False
            )
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

            # 生成文本
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            
            # 处理输出
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            report = output_text.split(conv.sep2)[1].strip()
            
            # 保存报告
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.basename(image_path).split('.')[0]
                output_path = os.path.join(output_dir, f"{base_name}_report.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                    
            return report
        
        except Exception as e:
            print(f"使用LLaVA专用处理方式失败：{str(e)}")
            print("尝试使用通用处理方式...")
            
            # 通用处理方式（备用）
            try:
                # 处理图像
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                
                # 生成文本
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens
                    )
                
                # 解码输出
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                report = generated_text.split("A: ")[-1] if "A: " in generated_text else generated_text
                
                # 保存报告
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.basename(image_path).split('.')[0]
                    output_path = os.path.join(output_dir, f"{base_name}_report.txt")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                
                return report
                
            except Exception as e:
                print(f"通用处理方式也失败了: {str(e)}")
                return "报告生成失败: 无法处理图像和文本"
    
    def _extract_structured_data(self, image_path: str, keypoints_path: Optional[str] = None) -> Dict:
        """提取结构化数据，包括关键点和角度"""
        # 默认角度值
        angles = {
            "left_ce_angle": None,
            "right_ce_angle": None,
            "left_sharp_angle": None,
            "right_sharp_angle": None,
            "left_tonnis_angle": None,
            "right_tonnis_angle": None
        }
        
        # 如果提供了关键点文件，从中计算角度
        if keypoints_path and os.path.exists(keypoints_path):
            try:
                keypoints = load_keypoints_from_json(keypoints_path)
                
                # 读取图像获取尺寸
                import cv2
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
                print("已从关键点文件计算角度")
            except Exception as e:
                print(f"从关键点计算角度失败: {str(e)}")
                print("使用默认角度值")
                angles = {
                    "left_ce_angle": 25.0,
                    "right_ce_angle": 28.0,
                    "left_sharp_angle": 37.0,
                    "right_sharp_angle": 35.0,
                    "left_tonnis_angle": 8.0,
                    "right_tonnis_angle": 7.0
                }
        
        # 构建结构化数据
        return {
            "angles": angles,
            "normal_ranges": {
                "CE": {"min": 25, "max": 40, "description": "Center-Edge角，测量髋臼对股骨头的覆盖程度"},
                "Sharp": {"min": 33, "max": 38, "description": "股骨头中心到髋臼外缘的连线与水平线的夹角"},
                "Tonnis": {"min": 0, "max": 10, "description": "髋臼荷重面与水平线的夹角"}
            }
        }
    
    def _format_angles_for_prompt(self, structured_data: Dict) -> str:
        """将角度数据格式化为可用于提示词的文本"""
        angles = structured_data["angles"]
        
        # 格式化角度信息
        if all(v is not None for v in angles.values()):
            return f"""
测量结果：
- 左侧CE角: {angles['left_ce_angle']:.1f}° (正常范围: 25-40°)
- 右侧CE角: {angles['right_ce_angle']:.1f}° (正常范围: 25-40°)
- 左侧Sharp角: {angles['left_sharp_angle']:.1f}° (正常范围: 33-38°)
- 右侧Sharp角: {angles['right_sharp_angle']:.1f}° (正常范围: 33-38°)
- 左侧Tönnis角: {angles['left_tonnis_angle']:.1f}° (正常范围: 0-10°)
- 右侧Tönnis角: {angles['right_tonnis_angle']:.1f}° (正常范围: 0-10°)
"""
        else:
            return ""
    
    def _build_default_prompt(self, angles_str: str) -> str:
        """构建默认提示词"""
        return f"""USER: <image>
我是放射科医生，请帮我分析这张骨盆X光片，并生成一份详细的医学报告。

{angles_str}

请提供以下内容：
1. 髋关节形态学描述
2. 测量结果解读
3. 诊断意见（是否存在发育性髋关节发育不良，严重程度如何）
4. 治疗建议

请按照标准放射学报告格式输出，使用专业但清晰的语言。

在报告末尾，请添加一个结构化诊断结果，格式如下：

===结构化诊断结果===
诊断标签: [双0/双1/双2/单1]
问题: [有/无/不确定]
侧别: [左侧/右侧/双侧/无]
类型: [正常/DDH/其他]
====================

其中：
- 双0：双侧无问题
- 双1：双侧有问题
- 双2：不确定，需要随访
- 单1：单侧有问题

A: """

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='骨盆X光片多模态医学大语言模型')
    parser.add_argument('--image', type=str, required=True, help='X光片图像路径')
    parser.add_argument('--keypoints', type=str, help='骨盆关键点JSON文件路径')
    parser.add_argument('--angles', type=str, help='骨盆角度数据JSON文件路径')
    parser.add_argument('--model', type=str, default="liuhaotian/llava-med-v1.0-7b", help='模型路径或名称')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='设备类型')
    parser.add_argument('--low_resource', action='store_true', help='低资源模式')
    parser.add_argument('--output_dir', type=str, default="reports", help='报告输出目录')
    parser.add_argument('--prompt', type=str, help='自定义提示词，如果不指定则使用默认提示词')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度，控制随机性（0表示无随机性）')
    parser.add_argument('--max_tokens', type=int, default=1024, help='生成的最大token数量')
    
    args = parser.parse_args()
    
    # 初始化模型
    mllm = HipMLLM(
        model_path=args.model,
        device=args.device,
        low_resource=args.low_resource
    )
    
    # 读取角度数据（如果有）
    angles_data = None
    if args.angles and os.path.exists(args.angles):
        try:
            with open(args.angles, 'r') as f:
                angles_data = json.load(f)
                print(f"已从{args.angles}加载角度数据")
        except Exception as e:
            print(f"读取角度数据失败: {str(e)}")
    
    # 生成报告
    report = mllm.generate_report(
        image_path=args.image,
        keypoints_path=args.keypoints,
        angles_data=angles_data,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        output_dir=args.output_dir
    )
    
    print("\n生成的医学报告:")
    print("-" * 50)
    print(report)
    print("-" * 50)

if __name__ == "__main__":
    main() 
