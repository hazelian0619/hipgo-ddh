import os
import shutil
import random
from pathlib import Path
import hashlib

def calculate_image_hash(image_path):
    """计算图片的MD5哈希值"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def create_clean_dataset(original_dir, expand_dir, target_dir, num_samples=200):
    """
    创建干净的数据集，只包含按顺序编号的图片
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有图片文件
    original_files = [f for f in os.listdir(original_dir) if f.endswith('.jpg')]
    expand_files = [f for f in os.listdir(expand_dir) if f.endswith('.jpg')]
    
    # 计算所有图片的哈希值
    image_hashes = set()
    unique_images = []
    
    # 处理原始数据集
    for file in original_files:
        file_path = os.path.join(original_dir, file)
        file_hash = calculate_image_hash(file_path)
        if file_hash not in image_hashes:
            image_hashes.add(file_hash)
            unique_images.append(('original', file))
    
    # 处理扩展数据集
    for file in expand_files:
        file_path = os.path.join(expand_dir, file)
        file_hash = calculate_image_hash(file_path)
        if file_hash not in image_hashes:
            image_hashes.add(file_hash)
            unique_images.append(('expand', file))
    
    # 随机选择指定数量的图片
    selected_images = random.sample(unique_images, min(num_samples, len(unique_images)))
    
    # 复制并重命名文件
    for idx, (source_type, source_file) in enumerate(selected_images, 1):
        source_dir = original_dir if source_type == 'original' else expand_dir
        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(target_dir, f"{idx:03d}.jpg")  # 简化文件名
        
        # 复制文件
        shutil.copy2(source_path, target_path)
    
    print(f"已创建 {len(selected_images)} 张验证图片在 {target_dir} 目录下")
    print(f"其中包含 {sum(1 for t, _ in selected_images if t == 'original')} 张原始数据集图片")
    print(f"和 {sum(1 for t, _ in selected_images if t == 'expand')} 张扩展数据集图片")
    print("所有图片都经过哈希值检查，确保没有重复")

if __name__ == "__main__":
    original_dir = "data/raw_images"
    expand_dir = "data/data expand"
    target_dir = "expert_validation_dataset"
    create_clean_dataset(original_dir, expand_dir, target_dir) 