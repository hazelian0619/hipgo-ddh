#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐点评估脚本：记录每个关键点的MAE，用于改动前后对比。

用法：
    cd hip_model
    python scripts/eval_per_point.py \
        --model_path models/model_best_20250506_163007.pth \
        --data_dir /Users/pluviophile/hip/shared_data/data/raw_images

输出：
    每个点的MAE（归一化坐标，以及换算成像素的估算值）
    整体PCK@0.05 和 PCK@0.10
"""

import sys, os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_gat_model import CNN_GAT
from dataset import HipKeypointDataset, get_transforms

POINT_NAMES = [
    '点1 左股骨头  ', '点2 右股骨头  ',
    '点3 左髋臼外缘', '点4 右髋臼外缘',
    '点5 耻骨联合  ',
    '点6 左荷重面内', '点7 左荷重面外',
    '点8 右荷重面内', '点9 右荷重面外',
]

def evaluate(model_path, data_dir, img_size=512, split_ratio=0.8, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    saved_args = ckpt['args']

    # checkpoint里保存的args可能是Namespace或dict，兼容两种格式
    a = saved_args
    get = lambda key: a[key] if isinstance(a, dict) else getattr(a, key, None)

    model = CNN_GAT(
        feature_dim=get('feature_dim') or 256,
        gat_hidden=get('gat_hidden') or 128,
        gat_output=get('gat_output') or 64,
        edge_features_dim=get('edge_features_dim') or 32,
        num_keypoints=get('num_keypoints') or 9,
        num_angles=get('num_angles') or 6,
        num_gat_layers=get('num_gat_layers') or 2,
        num_heads=get('num_heads') or 8,
        dropout=get('dropout') or 0.1,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_dataset = HipKeypointDataset(
        img_dir=data_dir,
        transform=get_transforms(train=False, img_size=img_size),
        train=False,
        split_ratio=split_ratio,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f'验证集大小: {len(val_dataset)} 张')

    all_errors = []   # [N, 9] 每个样本每个点的欧氏距离（归一化坐标）

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            kps_gt = batch['keypoints'][:, :, :2].to(device)   # [B, 9, 2]

            pred = model(images)
            kps_pred = pred['keypoints']                        # [B, 9, 2]

            # 逐点欧氏距离（归一化坐标）
            dist = torch.sqrt(((kps_pred - kps_gt) ** 2).sum(dim=2))  # [B, 9]
            all_errors.append(dist.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)  # [N, 9]

    # 每个点的MAE（归一化坐标）
    per_point_mae = all_errors.mean(axis=0)           # [9]
    overall_mae   = all_errors.mean()

    # 换算成像素（假设验证图平均512px）
    px_scale = img_size

    # PCK：距离 < threshold 的比例
    def pck(threshold):
        return (all_errors < threshold).mean()

    print('\n' + '='*55)
    print(f'  模型: {os.path.basename(model_path)}')
    print(f'  数据: {data_dir}')
    print('='*55)
    print(f'{"点":16s}  {"MAE(归一化)":>12s}  {"MAE(≈px)":>9s}')
    print('-'*55)
    for i, name in enumerate(POINT_NAMES):
        mae_norm = per_point_mae[i]
        mae_px   = mae_norm * px_scale
        bar = '█' * int(mae_norm / 0.003)   # 直观条形
        print(f'{name}  {mae_norm:>12.4f}  {mae_px:>7.1f}px  {bar}')
    print('-'*55)
    print(f'{"整体平均MAE":16s}  {overall_mae:>12.4f}  {overall_mae*px_scale:>7.1f}px')
    print(f'\nPCK@0.05  = {pck(0.05)*100:.1f}%  (误差<{0.05*px_scale:.0f}px算对)')
    print(f'PCK@0.10  = {pck(0.10)*100:.1f}%  (误差<{0.10*px_scale:.0f}px算对)')
    print('='*55)

    return {
        'per_point_mae': per_point_mae.tolist(),
        'overall_mae': float(overall_mae),
        'pck_05': float(pck(0.05)),
        'pck_10': float(pck(0.10)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/model_best_20250506_163007.pth')
    parser.add_argument('--data_dir',   default='/Users/pluviophile/hip/shared_data/data/raw_images')
    parser.add_argument('--img_size',   type=int, default=512)
    parser.add_argument('--split_ratio',type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir, args.img_size, args.split_ratio, args.batch_size)
