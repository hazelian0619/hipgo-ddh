import json

def check_labelme_format():
    """检查标注格式并打印结构"""
    try:
        with open('labeled_data/train/annotations/xray_001.json', 'r') as f:
            data = json.load(f)
            
        print("\n=== 标注文件结构 ===")
        print("1. 图片信息:")
        print(f"- 文件名: {data['imagePath']}")
        print(f"- 图片尺寸: {data['imageWidth']} x {data['imageHeight']}")
        
        print("\n2. 标注点信息:")
        for i, shape in enumerate(data['shapes'], 1):
            print(f"\n点 {i}:")
            print(f"- 标签: {shape['label']}")
            print(f"- 坐标: {shape['points']}")
            
    except Exception as e:
        print(f"错误：{str(e)}")

def check_updated_annotations():
    """检查更新后的标注文件"""
    with open('labeled_data/train/annotations/xray_001.json', 'r') as f:
        data = json.load(f)
    
    print("\n=== 更新后的标注信息 ===")
    print("关键点:")
    for shape in data['shapes']:
        print(f"- 标签 {shape['label']}: {shape['points']}")
    
    print("\nCE角度:")
    print(f"- 左侧: {data['ce_angles']['left']:.2f}°")
    print(f"- 右侧: {data['ce_angles']['right']:.2f}°")

if __name__ == "__main__":
    check_labelme_format() 