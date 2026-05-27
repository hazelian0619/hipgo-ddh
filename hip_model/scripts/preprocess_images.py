# -*- coding: utf-8 -*-
import os
from PIL import Image
import shutil

def resize_image(image_path, target_size=1024):
    """缩放图像，保持纵横比，较短边填充黑色"""
    img = Image.open(image_path)
    
    # 计算缩放比例
    ratio = target_size / float(max(img.size))
    new_size = tuple([int(x * ratio) for x in img.size])
    
    # 缩放图像
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # 创建黑色背景
    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # 计算粘贴位置（居中）
    paste_x = (target_size - new_size[0]) // 2
    paste_y = (target_size - new_size[1]) // 2
    
    # 粘贴图像
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def process_dataset():
    """处理整个数据集"""
    src_dir = 'raw_images'
    processed_dir = 'processed_images'
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    for filename in os.listdir(src_dir):
        if filename.endswith('.jpg'):
            print('Processing {}...'.format(filename))
            image_path = os.path.join(src_dir, filename)
            processed_img = resize_image(image_path)
            
            output_path = os.path.join(processed_dir, filename)
            processed_img.save(output_path, quality=95)
            
            json_filename = filename.replace('.jpg', '.json')
            json_src = os.path.join(src_dir, json_filename)
            json_dst = os.path.join(processed_dir, json_filename)
            if os.path.exists(json_src):
                shutil.copy2(json_src, json_dst)

if __name__ == '__main__':
    process_dataset()
    print('预处理完成！')