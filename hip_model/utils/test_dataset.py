from utils.dataset import CEAngleDataset  # 注意从utils导入
from torchvision import transforms
import os
from utils.metrics import calculate_bilateral_ce_angles
import matplotlib.pyplot as plt

def test_data_loading():
    # 检查环境
    print("当前目录:", os.getcwd())
    print("图片目录存在?", os.path.exists('labeled_data/train/images'))
    
    # 创建数据集实例
    dataset = CEAngleDataset(
        coco_json='labeled_data/train/dataset.json',
        img_dir='labeled_data/train/images',  # 直接指向images目录
        debug=True
    )
    
    # 测试加载第一张图片
    img, boxes, angles = dataset[0]
    print("成功加载图片!")
    print(f"图片尺寸: {img.shape}")
    print(f"检测框数量: {len(boxes)}")
    print(f"角度数量: {len(angles)}")

def test_keypoints_loading():
    dataset = CEAngleDataset(
        coco_json='labeled_data/train/dataset.json',
        img_dir='labeled_data/train/images',
        debug=True
    )
    
    # 测试第一张图片
    img, keypoints = dataset[0]
    print("图片加载成功!")
    print(f"关键点数据: {keypoints}")

def test_ce_angle():
    dataset = CEAngleDataset(
        coco_json='labeled_data/train/dataset.json',
        img_dir='labeled_data/train/images',
        debug=True
    )
    
    # 测试第一张图片
    img, points = dataset[0]
    print("图片加载成功!")
    
    # 计算双侧CE角度
    left_angle, right_angle = calculate_bilateral_ce_angles(points)
    print(f"左侧CE角度: {left_angle:.2f}°")
    print(f"右侧CE角度: {right_angle:.2f}°")

def test_dataset():
    # 1. 创建数据集实例
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    dataset = CEAngleDataset(
        img_dir='labeled_data/train/images',
        ann_dir='labeled_data/train/annotations',
        transform=transform
    )
    
    # 2. 测试第一个样本
    sample = dataset[0]
    print("\n=== 数据集测试 ===")
    print(f"图片尺寸: {sample['image'].shape}")
    print(f"关键点: {sample['points']}")
    print(f"CE角度: {sample['angles']}")
    
    # 3. 可视化
    img = sample['image'].permute(1, 2, 0)  # CHW -> HWC
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    test_data_loading()
    test_keypoints_loading()
    test_ce_angle()
    test_dataset() 