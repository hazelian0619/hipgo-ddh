import json
import os
from pathlib import Path

def update_coco_with_pubic_points(original_coco_path, new_labelme_dir, output_coco_path):
    """将Labelme标注的耻骨联合点添加到现有COCO数据集"""
    # 加载原始COCO数据
    with open(original_coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 获取最大的annotation ID
    max_ann_id = max([ann['id'] for ann in coco_data['annotations']])
    
    # 添加新类别 - 耻骨联合点
    if len(coco_data['categories']) == 4:  # 确保不重复添加
        coco_data['categories'].append({
            "id": 4,
            "name": "pubic_point",
            "supercategory": "pubic_point"
        })
    
    # 遍历所有新的Labelme标注文件
    for labelme_file in Path(new_labelme_dir).glob('*.json'):
        with open(labelme_file, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        
        # 获取图像文件名
        image_filename = labelme_data['imagePath']
        image_id = None
        
        # 找到对应的image_id
        for img in coco_data['images']:
            if Path(img['file_name']).name == Path(image_filename).name:
                image_id = img['id']
                break
        
        if image_id is None:
            print(f"警告：找不到图像 {image_filename} 对应的ID")
            continue
        
        # 查找耻骨联合点标注
        for shape in labelme_data['shapes']:
            if shape['label'].lower() == 'pubic_point':
                # 获取坐标
                point = shape['points'][0]  # Labelme的点格式是[[x,y]]
                
                # 创建新的标注
                max_ann_id += 1
                new_annotation = {
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [point[0], point[1], 1.0, 1.0],  # 与现有格式保持一致
                    "segmentation": [],
                    "category_id": 4,  # 耻骨联合点
                    "id": max_ann_id,
                    "area": 1
                }
                
                # 添加到COCO数据中
                coco_data['annotations'].append(new_annotation)
                print(f"添加了图像 {image_filename} 的耻骨联合点")
    
    # 保存更新后的COCO数据
    with open(output_coco_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f)
    
    print(f"已更新COCO数据集并保存到 {output_coco_path}")

# 使用示例
update_coco_with_pubic_points(
    'ce_angle_project/labeled_data/train/dataset.json',
    'new_annotations',  # 替换为实际的新标注文件目录
    'ce_angle_project/labeled_data/train/updated_dataset.json'
)
