import os
import shutil
from pathlib import Path

def create_validation_dataset(source_dir, target_dir):
    """
    创建验证数据集，只使用原始的不同图片
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有jpg文件
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    source_files.sort()  # 确保按顺序处理
    
    # 复制并重命名文件
    for idx, source_file in enumerate(source_files, 1):
        # 生成新的文件名
        new_name = f"validation_{idx:03d}.jpg"
        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(target_dir, new_name)
        
        # 复制文件
        shutil.copy2(source_path, target_path)
        
        # 同时复制对应的JSON文件
        json_source = source_file.replace('.jpg', '.json')
        json_target = new_name.replace('.jpg', '.json')
        if os.path.exists(os.path.join(source_dir, json_source)):
            shutil.copy2(
                os.path.join(source_dir, json_source),
                os.path.join(target_dir, json_target)
            )
            
    print(f"已创建 {len(source_files)} 张验证图片在 {target_dir} 目录下")
    print("注意：由于原始数据集只有60张不同的图片，验证数据集也包含60张不同的图片")

if __name__ == "__main__":
    source_dir = "data/raw_images"
    target_dir = "expert_validation_dataset"
    create_validation_dataset(source_dir, target_dir) 