# -*- coding: utf-8 -*-
import os

# Get the maximum number of existing files
def get_max_number():
    files = [f for f in os.listdir('raw_images') if f.startswith('xray_') and f.endswith('.jpg')]
    if not files:
        return 0
    numbers = [int(f.split('_')[1].split('.')[0]) for f in files]
    return max(numbers)

# 重命名新文件
def rename_new_files():
    start_number = get_max_number() + 1
    counter = start_number
    
    for filename in os.listdir('raw_images'):
        # 跳过已重命名的文件
        if filename.startswith('xray_'):
            continue
            
        if filename.endswith('.jpg'):
            # 使用旧版字符串格式化
            new_name = 'xray_{:03d}.jpg'.format(counter)
            old_path = os.path.join('raw_images', filename)
            new_path = os.path.join('raw_images', new_name)
            os.rename(old_path, new_path)
            print('Renamed: {} -> {}'.format(filename, new_name))
            counter += 1

if __name__ == '__main__':
    rename_new_files() 