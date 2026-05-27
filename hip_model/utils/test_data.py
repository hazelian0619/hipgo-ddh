import os
import json
from utils.dataset import CEAngleDataset

def test_data():
    print("Testing data loading...")
    
    # 1. 检查目录
    data_dir = "data/train"  # 使用正确的数据路径
    print(f"Checking directory: {data_dir}")
    assert os.path.exists(data_dir), "Data directory not found"
    
    # 2. 测试标注文件
    anno_file = os.path.join(data_dir, "annotations", "xray_001.json")
    print(f"Testing annotation file: {anno_file}")
    try:
        with open(anno_file, 'r', encoding='gbk') as f:  # 使用GBK编码
            data = json.load(f)
        print("Successfully loaded annotation")
        print(f"Found {len(data['shapes'])} shapes")
    except Exception as e:
        print(f"Error loading annotation: {e}")
    
    # 3. 测试数据集
    try:
        dataset = CEAngleDataset(data_dir)  # 只需要传入data_dir
        print(f"Dataset size: {len(dataset)}")
        
        # 测试第一个样本
        sample = dataset[0]
        if sample is not None:
            print("Successfully loaded first sample")
            image, points, angles = sample
            print(f"Image shape: {image.shape if hasattr(image, 'shape') else 'PIL Image'}")
            print(f"Points shape: {points.shape}")
            print(f"Angles shape: {angles.shape}")
    except Exception as e:
        print(f"Error in dataset: {e}")

if __name__ == "__main__":
    test_data() 