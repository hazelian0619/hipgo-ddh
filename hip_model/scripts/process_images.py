from PIL import Image
import os

def resize_images():
    input_folder = "raw_images"  # 原始图片目录
    output_folder = "processed_images"  # 处理后图片保存目录
    
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(input_folder, file))
            # 调整大小为1024x1024
            img.thumbnail((1024, 1024))
            new_img = Image.new("RGB", (1024, 1024), (0, 0, 0))
            pos = ((1024 - img.size[0]) // 2, (1024 - img.size[1]) // 2)
            new_img.paste(img, pos)
            new_img.save(os.path.join(output_folder, file)) 