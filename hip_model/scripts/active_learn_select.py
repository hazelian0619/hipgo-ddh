#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主动学习不确定性采样（使用纯CNN最佳模型）

1. 加载训练好的纯CNN模型
2. MC Dropout推理未标注图
3. 选出不确定性最高的N张，输出文件列表供手动标注

用法:
    python scripts/active_learn_select.py \
        --model_path outputs/best_model/best_cnn_model.pth \
        --unlabeled_dir /path/to/unlabeled/images \
        --top_k 20 --output_dir outputs/active_round1
"""
import os, sys, json, torch, torch.nn as nn, numpy as np, argparse, cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone.feature_extractor import FeatureExtractor
from utils.transforms import get_prediction_transforms


class CNN_Keypoint(nn.Module):
    def __init__(self, pretrained=False, num_kps=9):
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


class ImgDataset(Dataset):
    def __init__(self, d, t):
        self.files = sorted([f for f in os.listdir(d) if f.endswith(('.jpg','.jpeg','.png')) and not f.endswith('.json')])
        self.d, self.t = d, t
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        n = self.files[i]; img = Image.open(os.path.join(self.d, n)).convert('RGB')
        if self.t: img = self.t(image=np.array(img))['image']
        return {'image': img, 'image_name': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--unlabeled_dir', default='/Users/pluviophile/hip/shared_data/data/raw_images')
    ap.add_argument('--output_dir', default='outputs/active_round1')
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--mc_passes', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dv = torch.device('cpu')
    ckpt = torch.load(args.model_path, map_location=dv)
    model = CNN_Keypoint(num_kps=9).to(dv)
    model.load_state_dict(ckpt['model_state_dict'])

    tf = get_prediction_transforms(img_size=512)
    ds = ImgDataset(args.unlabeled_dir, tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'模型: {args.model_path}')
    print(f'未标注图: {len(ds)} 张, MC Dropout {args.mc_passes}次')

    # TTA不确定度评估：同一张图用不同随机增强推理10次
    # 如果10次预测方差大 → 模型不确定 → 优先人工标注
    model.eval()
    # 用训练时的transform（带随机增强）做多次推理
    from dataset import get_transforms
    tta_transform = get_transforms(train=True, img_size=512)

    results = []
    # 只对未标注的数据做（排除已有JSON的）
    all_files = sorted([f for f in os.listdir(args.unlabeled_dir)
                        if f.endswith(('.jpg','.jpeg','.png')) and not f.endswith('.json')])
    print(f'未标注图: {len(all_files)} 张, TTA {args.mc_passes}次')

    for fname in tqdm(all_files, desc='TTA不确定度'):
        img_path = os.path.join(args.unlabeled_dir, fname)
        all_preds = []
        for _ in range(args.mc_passes):
            # 每次用不同的随机增强
            img = cv2.imread(img_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kps_dummy = [(0, 0)] * 9  # 无标注，用dummy关键点
            t = tta_transform(image=img, keypoints=kps_dummy)
            img_t = t['image'].unsqueeze(0).to(dv)
            with torch.no_grad():
                pred = model(img_t)['keypoints'][0].cpu().numpy()
            all_preds.append(pred)
        all_preds = np.stack(all_preds)  # [mc_passes, 9, 2]
        per_point_std = np.sqrt((all_preds.std(axis=0) ** 2).sum(axis=1))  # [9]
        uncertainty = float(per_point_std.mean())
        results.append({'image_name': fname, 'uncertainty': uncertainty,
                        'per_point_std': per_point_std.tolist(),
                        'mean_kps': all_preds.mean(axis=0).tolist()})

    results.sort(key=lambda r: r['uncertainty'], reverse=True)

    # 输出
    top = results[:args.top_k]
    print(f'\n{"="*55}')
    print(f'  Top {args.top_k} 最不确定的图（优先标注）')
    print(f'{"="*55}')
    for i, r in enumerate(top, 1):
        pts = ' '.join(f'{s:.4f}' for s in r['per_point_std'])
        print(f'  {i:2d}. {r["image_name"]:30s}  uncertainty={r["uncertainty"]:.4f}  [{pts}]')

    # 保存列表（方便去LabelMe打开）
    with open(os.path.join(args.output_dir, 'to_annotate.txt'), 'w') as f:
        for r in top:
            f.write(f"{r['image_name']}\n")

    with open(os.path.join(args.output_dir, 'full_ranking.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    unc = [r['uncertainty'] for r in results]
    print(f'\n不确定度: min={min(unc):.4f} max={max(unc):.4f} median={np.median(unc):.4f}')
    print(f'标注列表: {args.output_dir}/to_annotate.txt')
    print(f'完整排序: {args.output_dir}/full_ranking.json')


if __name__ == '__main__':
    main()
