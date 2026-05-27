#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量60张训练纯CNN（实验C最佳架构），保存模型用于主动学习。
"""
import os, sys, json, torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone.feature_extractor import FeatureExtractor
from dataset import get_transforms

# ---- 纯CNN模型（实验C） ----
class CNN_Keypoint(nn.Module):
    def __init__(self, pretrained=True, num_kps=9):
        super().__init__()
        self.fe = FeatureExtractor(pretrained=pretrained, feature_dim=256)
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.pred = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, num_kps * 2),
        )
        self.num_kps = num_kps

    def forward(self, x):
        _, mf = self.fe(x)
        gf = self.gp(mf).view(x.size(0), -1)
        return {'keypoints': torch.sigmoid(self.pred(gf).view(x.size(0), self.num_kps, 2)),
                'angles': torch.zeros(x.size(0), 6, device=x.device)}

# ---- 全量数据集（不split，全部训练） ----
class FullDataset(Dataset):
    def __init__(self, data_dir, transform, flip_p=0.5):
        self.data_dir = data_dir
        self.transform = transform
        self.flip_p = flip_p
        self.files = sorted([f for f in os.listdir(data_dir)
                             if f.endswith('.jpg') and os.path.exists(os.path.join(data_dir, f.replace('.jpg', '.json')))])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        n = self.files[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, n)), cv2.COLOR_BGR2RGB)
        with open(os.path.join(self.data_dir, n.replace('.jpg', '.json'))) as f:
            ann = json.load(f)
        shapes = sorted(ann['shapes'], key=lambda x: int(x['label']))
        kps = np.array([[s['points'][0][0], s['points'][0][1]] for s in shapes], dtype=np.float32)
        oh, ow = img.shape[:2]

        # 翻转 + 语义互换
        if np.random.rand() < self.flip_p:
            img = img[:, ::-1, :].copy()
            kps[:, 0] = ow - kps[:, 0]
            swap = [(0, 1), (2, 3), (5, 7), (6, 8)]
            sk = kps.copy()
            for i, j in swap: sk[i], sk[j] = kps[j].copy(), kps[i].copy()
            kps = sk

        t = self.transform(image=img, keypoints=[(x, y) for x, y in kps])
        img, kps = t['image'], np.array(t['keypoints'], dtype=np.float32)
        nh = img.shape[-2] if isinstance(img, torch.Tensor) else img.shape[0]
        nw = img.shape[-1] if isinstance(img, torch.Tensor) else img.shape[1]
        kps[:, 0] /= nw; kps[:, 1] /= nh
        return {'image': img, 'keypoints': torch.tensor(np.column_stack((kps, np.ones(9))), dtype=torch.float32)}

# ---- 训练 ----
if __name__ == '__main__':
    data_dir = '/Users/pluviophile/hip/shared_data/data/raw_images'
    output_dir = 'outputs/best_model'
    os.makedirs(output_dir, exist_ok=True)

    dv = torch.device('cpu')
    ds = FullDataset(data_dir, get_transforms(train=True, img_size=512))
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    print(f'全量训练: {len(ds)} 张, {len(dl)} batches/epoch')

    model = CNN_Keypoint(pretrained=True, num_kps=9).to(dv)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # 加权loss
    _ps = [0.017, 0.017, 0.051, 0.055, 0.073, 0.029, 0.041, 0.028, 0.048]
    _w = torch.tensor([(1/(1+s))/sum(1/(1+ss) for ss in _ps)*9 for s in _ps]).to(dv)

    best_loss = float('inf')
    for ep in range(1, 31):
        model.train()
        total = 0.0
        for b in dl:
            im, gt = b['image'].to(dv), b['keypoints'][:, :, :2].to(dv)
            loss = (((model(im)['keypoints'] - gt) ** 2).mean(dim=2) * _w.unsqueeze(0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(dl)

        if avg < best_loss:
            best_loss = avg
            torch.save({'model_state_dict': model.state_dict(), 'loss': avg, 'epoch': ep},
                       os.path.join(output_dir, 'best_cnn_model.pth'))

        if ep % 5 == 0:
            print(f'  Epoch {ep}/30: loss={avg:.6f}  best={best_loss:.6f}')

    print(f'\n训练完成! 最佳loss={best_loss:.6f}')
    print(f'模型保存: {output_dir}/best_cnn_model.pth')
