import os
from utils.dataset import CEAngleDataset

# 测试数据集初始化
data_dir = "data/train"
dataset = CEAngleDataset(data_dir)
print(f"Dataset size: {len(dataset)}")

# 测试单个样本加载
sample = dataset[0]
if sample is not None:
    print("Successfully loaded first sample") 