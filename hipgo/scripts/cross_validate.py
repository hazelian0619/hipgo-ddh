#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5折交叉验证脚本

把60张标注图分5折，每折12张，轮流用4折训练、1折验证。
最终报告 MAE 和 PCK 的均值 ± 标准差。

用法:
    cd hip_model
    python scripts/cross_validate.py \
        --data_dir /Users/pluviophile/hip/shared_data/data/raw_images \
        --output_dir outputs/cross_val \
        --epochs 30 --batch_size 4 --lr 0.0001
"""

import os, sys, json, argparse
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_gat_model import CNN_GAT
from dataset import get_transforms
from torch.utils.data import Dataset, DataLoader
import cv2

POINT_NAMES = [
    '点1 左股骨头', '点2 右股骨头',
    '点3 左髋臼外缘', '点4 右髋臼外缘',
    '点5 耻骨联合',
    '点6 左荷重面内', '点7 左荷重面外',
    '点8 右荷重面内', '点9 右荷重面外',
]

# 加权loss（和train.py保持一致）
_point_stds = [0.017, 0.017, 0.051, 0.055, 0.073, 0.029, 0.041, 0.028, 0.048]
_raw = [1.0 / (1.0 + s) for s in _point_stds]
_sum = sum(_raw)
KEYPOINT_WEIGHTS = torch.tensor([w / _sum * 9 for w in _raw], dtype=torch.float32)


class FoldDataset(Dataset):
    """指定图片列表的数据集，用于交叉验证的每个fold"""

    def __init__(self, img_dir, file_list, transform=None, train=True, flip_p=0.5):
        self.img_dir = img_dir
        self.img_files = file_list
        self.transform = transform
        self.train = train
        self.flip_p = flip_p

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_path = os.path.join(self.img_dir, img_name.replace('.jpg', '.json'))
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        keypoints_raw = []
        for shape in ann_data['shapes']:
            label = int(shape['label'])
            point = shape['points'][0]
            keypoints_raw.append([label, point[0], point[1]])
        keypoints_raw.sort(key=lambda x: x[0])
        keypoints = np.array([[kp[1], kp[2]] for kp in keypoints_raw], dtype=np.float32)

        orig_height, orig_width = img.shape[:2]

        # 训练时随机翻转+语义互换
        if self.train and np.random.rand() < self.flip_p:
            img = img[:, ::-1, :].copy()
            keypoints[:, 0] = orig_width - keypoints[:, 0]
            # 语义互换
            swap = [(0, 1), (2, 3), (5, 7), (6, 8)]
            swapped = keypoints.copy()
            for i, j in swap:
                swapped[i], swapped[j] = keypoints[j].copy(), keypoints[i].copy()
            keypoints = swapped

        if self.transform:
            kps_albu = [(x, y) for x, y in keypoints]
            transformed = self.transform(image=img, keypoints=kps_albu)
            img = transformed['image']
            keypoints = np.array(transformed['keypoints'], dtype=np.float32)

        new_h, new_w = img.shape[-2:] if isinstance(img, torch.Tensor) else img.shape[:2]
        norm_kps = keypoints.copy()
        norm_kps[:, 0] /= new_w
        norm_kps[:, 1] /= new_h

        return {
            'image': img,
            'keypoints': torch.tensor(np.column_stack((norm_kps, np.ones(9, dtype=np.float32))), dtype=torch.float32),
            'image_id': img_name,
        }


def train_one_fold(model, train_loader, val_loader, epochs, lr, device, fold_id, output_dir):
    """训练一个fold并返回验证集结果"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    weights = KEYPOINT_WEIGHTS.to(device)
    best_val = float('inf')
    best_model = None
    patience_cnt = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            kps_gt = batch['keypoints'][:, :, :2].to(device)

            out = model(images)
            kps_pred = out['keypoints']
            per_point_mse = ((kps_pred - kps_gt) ** 2).mean(dim=2)
            loss = (per_point_mse * weights.unsqueeze(0)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                kps_gt = batch['keypoints'][:, :, :2].to(device)
                kps_pred = model(images)['keypoints']
                per_point_mse = ((kps_pred - kps_gt) ** 2).mean(dim=2)
                loss = (per_point_mse * weights.unsqueeze(0)).mean()
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= 10:
            break

    # 用最佳模型做详细评估
    model.load_state_dict(best_model)
    model.eval()

    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            kps_gt = batch['keypoints'][:, :, :2].to(device)
            kps_pred = model(images)['keypoints']
            dist = torch.sqrt(((kps_pred - kps_gt) ** 2).sum(dim=2))
            all_errors.append(dist.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)  # [N_val, 9]
    per_point_mae = all_errors.mean(axis=0)
    overall_mae = all_errors.mean()
    pck_05 = (all_errors < 0.05).mean()
    pck_10 = (all_errors < 0.10).mean()

    print(f'  Fold {fold_id}: MAE={overall_mae:.4f}  PCK@0.05={pck_05:.1%}  PCK@0.10={pck_10:.1%}')

    return per_point_mae, overall_mae, pck_05, pck_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/pluviophile/hip/shared_data/data/raw_images')
    parser.add_argument('--output_dir', default='outputs/cross_val')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img_size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 收集所有有标注的图片
    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.jpg') and
                        os.path.exists(os.path.join(args.data_dir, f.replace('.jpg', '.json')))])

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_files = list(kf.split(all_files))

    print(f'数据: {len(all_files)} 张标注图, {args.n_folds}折, 每折≈{len(all_files)//args.n_folds}张')
    print(f'每fold训练 {args.epochs} epochs, device={device}\n')

    results_mae = []
    results_pck05 = []
    results_pck10 = []
    results_per_point = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_files, 1):
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]

        train_ds = FoldDataset(args.data_dir, train_files,
                               get_transforms(train=True, img_size=args.img_size), train=True)
        val_ds = FoldDataset(args.data_dir, val_files,
                             get_transforms(train=False, img_size=args.img_size), train=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = CNN_GAT(num_keypoints=9, num_angles=6, pretrained=False).to(device)

        pp_mae, overall, p5, p10 = train_one_fold(
            model, train_loader, val_loader, args.epochs, args.lr, device, fold_id, args.output_dir)

        results_mae.append(overall)
        results_pck05.append(p5)
        results_pck10.append(p10)
        results_per_point.append(pp_mae)

    results_mae = np.array(results_mae)
    results_pck05 = np.array(results_pck05)
    results_pck10 = np.array(results_pck10)
    pp_all = np.array(results_per_point)  # [5, 9]

    print('\n' + '=' * 60)
    print('  5折交叉验证结果')
    print('=' * 60)
    print(f'{"点":16s}  {"MAE均值":>8s}  {"±std":>6s}')
    print('-' * 40)
    for i, name in enumerate(POINT_NAMES):
        m = pp_all[:, i]
        print(f'{name:16s}  {m.mean():>8.4f}  ±{m.std():>5.4f}')
    print('-' * 40)
    print(f'{"整体MAE":16s}  {results_mae.mean():>8.4f}  ±{results_mae.std():>5.4f}')
    print(f'\nPCK@0.05: {results_pck05.mean():.1%} ± {results_pck05.std():.1%}')
    print(f'PCK@0.10: {results_pck10.mean():.1%} ± {results_pck10.std():.1%}')
    print('=' * 60)

    # 保存结果
    result = {
        'per_point_mae_mean': pp_all.mean(axis=0).tolist(),
        'per_point_mae_std': pp_all.std(axis=0).tolist(),
        'overall_mae': float(results_mae.mean()),
        'overall_mae_std': float(results_mae.std()),
        'pck_05': float(results_pck05.mean()),
        'pck_05_std': float(results_pck05.std()),
        'pck_10': float(results_pck10.mean()),
        'pck_10_std': float(results_pck10.std()),
    }
    with open(os.path.join(args.output_dir, 'cross_val_result.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到 {args.output_dir}/cross_val_result.json')


if __name__ == '__main__':
    main()
