#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验统一入口：同时运行实验A、B、C的5折交叉验证。

用法:
    python scripts/cross_validate_variants.py \
        --data_dir /Users/pluviophile/hip/shared_data/data/raw_images \
        --epochs 30 --batch_size 4 --lr 0.0001

输出:
    outputs/ablation/experiment_A.json  完整CNN-GAT + pretrained
    outputs/ablation/experiment_B.json  CNN-GAT + 无预训练
    outputs/ablation/experiment_C.json  纯CNN + 无GAT
    outputs/ablation/comparison.txt     对比表格
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_gat_model import CNN_GAT
from models.backbone.feature_extractor import FeatureExtractor
from dataset import get_transforms

POINT_NAMES = [
    '点1 左股骨头', '点2 右股骨头',
    '点3 左髋臼外缘', '点4 右髋臼外缘',
    '点5 耻骨联合',
    '点6 左荷重面内', '点7 左荷重面外',
    '点8 右荷重面内', '点9 右荷重面外',
]

_point_stds = [0.017, 0.017, 0.051, 0.055, 0.073, 0.029, 0.041, 0.028, 0.048]
_weights = torch.tensor([(1/(1+s)) / sum([1/(1+ss) for ss in _point_stds]) * 9 for s in _point_stds], dtype=torch.float32)


# ---- 数据集 ----
class FoldDataset(Dataset):
    def __init__(self, img_dir, file_list, transform=None, train=True, flip_p=0.5):
        self.img_dir, self.file_list, self.transform, self.train, self.flip_p = img_dir, file_list, transform, train, flip_p

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, img_name)), cv2.COLOR_BGR2RGB)
        with open(os.path.join(self.img_dir, img_name.replace('.jpg', '.json'))) as f:
            ann = json.load(f)
        shapes = sorted(ann['shapes'], key=lambda x: int(x['label']))
        kps = np.array([[s['points'][0][0], s['points'][0][1]] for s in shapes], dtype=np.float32)
        oh, ow = img.shape[:2]

        if self.train and np.random.rand() < self.flip_p:
            img = img[:, ::-1, :].copy()
            kps[:, 0] = ow - kps[:, 0]
            swap = [(0,1),(2,3),(5,7),(6,8)]
            s = kps.copy()
            for i,j in swap: s[i], s[j] = kps[j].copy(), kps[i].copy()
            kps = s

        if self.transform:
            t = self.transform(image=img, keypoints=[(x,y) for x,y in kps])
            img, kps = t['image'], np.array(t['keypoints'], dtype=np.float32)

        nh = img.shape[-2] if isinstance(img, torch.Tensor) else img.shape[0]
        nw = img.shape[-1] if isinstance(img, torch.Tensor) else img.shape[1]
        kps[:,0] /= nw; kps[:,1] /= nh
        return {'image': img, 'keypoints': torch.tensor(np.column_stack((kps, np.ones(9))), dtype=torch.float32)}


# ---- 模型B: 无GAT的纯CNN关键点预测器 ----
class CNN_Only(nn.Module):
    """只有ResNet50 + 关键点预测头，没有图注意力"""
    def __init__(self, pretrained=True, num_keypoints=9):
        super().__init__()
        self.feature_extractor = FeatureExtractor(pretrained=pretrained, feature_dim=256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_keypoints * 2),
        )
        self.num_keypoints = num_keypoints

    def forward(self, x):
        _, main_feat = self.feature_extractor(x)          # [B, 256, H, W]
        global_feat = self.global_pool(main_feat).view(x.size(0), -1)  # [B, 256]
        pred = self.predictor(global_feat).view(x.size(0), self.num_keypoints, 2)
        return {'keypoints': torch.sigmoid(pred), 'angles': torch.zeros(x.size(0), 6, device=x.device)}


# ---- 训练一个fold ----
def train_one_fold(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    w = _weights.to(device)
    best_val, best_state, patience = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        for b in train_loader:
            imgs, gt = b['image'].to(device), b['keypoints'][:, :, :2].to(device)
            pred = model(imgs)['keypoints']
            loss = (((pred - gt)**2).mean(dim=2) * w.unsqueeze(0)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                imgs, gt = b['image'].to(device), b['keypoints'][:, :, :2].to(device)
                pred = model(imgs)['keypoints']
                vl += (((pred - gt)**2).mean(dim=2) * w.unsqueeze(0)).mean().item()
        vl /= len(val_loader); scheduler.step(vl)

        if vl < best_val:
            best_val = vl; patience = 0
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 10: break

    model.load_state_dict(best_state); model.eval()
    all_err = []
    with torch.no_grad():
        for b in val_loader:
            imgs, gt = b['image'].to(device), b['keypoints'][:, :, :2].to(device)
            all_err.append(torch.sqrt(((model(imgs)['keypoints'] - gt)**2).sum(dim=2)).cpu().numpy())
    err = np.concatenate(all_err, axis=0)
    return err.mean(axis=0), err.mean(), (err < 0.05).mean(), (err < 0.10).mean()


# ---- 主实验 ----
def run_experiment(name, model_fn, data_dir, n_folds, epochs, lr, batch_size, img_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_files = sorted([f for f in os.listdir(data_dir)
                        if f.endswith('.jpg') and os.path.exists(os.path.join(data_dir, f.replace('.jpg', '.json')))])
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(kf.split(all_files))

    print(f'\n{"="*50}')
    print(f'  实验: {name}')
    print(f'{"="*50}')

    pp_mae_list, ov_mae_list, pck05_list, pck10_list = [], [], [], []
    for fid, (ti, vi) in enumerate(folds, 1):
        t_ds = FoldDataset(data_dir, [all_files[i] for i in ti], get_transforms(train=True, img_size=img_size), train=True)
        v_ds = FoldDataset(data_dir, [all_files[i] for i in vi], get_transforms(train=False, img_size=img_size), train=False)
        t_ld = DataLoader(t_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        v_ld = DataLoader(v_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = model_fn().to(device)
        pp, ov, p5, p10 = train_one_fold(model, t_ld, v_ld, epochs, lr, device)
        pp_mae_list.append(pp); ov_mae_list.append(ov); pck05_list.append(p5); pck10_list.append(p10)
        print(f'  Fold {fid}: MAE={ov:.4f}  PCK@0.05={p5:.1%}  PCK@0.10={p10:.1%}')

    pp = np.array(pp_mae_list); ov = np.array(ov_mae_list); p5 = np.array(pck05_list); p10 = np.array(pck10_list)
    return {
        'name': name, 'per_point_mae': pp.mean(0).tolist(), 'per_point_std': pp.std(0).tolist(),
        'overall_mae': float(ov.mean()), 'overall_std': float(ov.std()),
        'pck_05': float(p5.mean()), 'pck_05_std': float(p5.std()),
        'pck_10': float(p10.mean()), 'pck_10_std': float(p10.std()),
    }


def print_table(results):
    print('\n\n' + '=' * 70)
    print('  消融实验对比表')
    print('=' * 70)
    h = f'{"实验":28s} {"MAE":>8s} {"PCK@0.05":>10s} {"PCK@0.10":>10s}'
    print(h); print('-' * 62)
    for r in results:
        print(f'{r["name"]:28s} {r["overall_mae"]:>8.4f}  {r["pck_05"]:>9.1%}  {r["pck_10"]:>9.1%}')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/pluviophile/hip/shared_data/data/raw_images')
    parser.add_argument('--output_dir', default='outputs/ablation')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    # 实验A: CNN-GAT + pretrained
    rA = run_experiment('A: CNN-GAT + pretrained',
        lambda: CNN_GAT(num_keypoints=9, num_angles=6, pretrained=True),
        args.data_dir, 5, args.epochs, args.lr, args.batch_size, 512)
    results.append(rA)

    # 实验B: CNN-GAT + no pretrained
    rB = run_experiment('B: CNN-GAT (no pretrain)',
        lambda: CNN_GAT(num_keypoints=9, num_angles=6, pretrained=False),
        args.data_dir, 5, args.epochs, args.lr, args.batch_size, 512)
    results.append(rB)

    # 实验C: 纯CNN（无GAT）
    rC = run_experiment('C: CNN only (no GAT)',
        lambda: CNN_Only(pretrained=True, num_keypoints=9),
        args.data_dir, 5, args.epochs, args.lr, args.batch_size, 512)
    results.append(rC)

    print_table(results)

    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n完整结果: {args.output_dir}/ablation_results.json')
