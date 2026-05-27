#!/usr/bin/env python
"""逐点评估：报告每个关键点的MAE、整体PCK"""
import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader

from hipgo.dataset import HipKeypointDataset, get_transforms
from hipgo.models import CNNKeypoint

NAMES = ['左股骨头','右股骨头','左髋臼外缘','右髋臼外缘','耻骨联合','左荷重面内','左荷重面外','右荷重面内','右荷重面外']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--split_ratio', type=float, default=0.8)
    p.add_argument('--batch_size', type=int, default=4)
    args = p.parse_args()

    dv = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.model_path, map_location=dv)
    model = CNNKeypoint().to(dv)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ds = HipKeypointDataset(args.data_dir, get_transforms(False, args.img_size), False, args.split_ratio)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'验证: {len(ds)} 张')

    errors = []
    with torch.no_grad():
        for batch in dl:
            gt = batch['keypoints'][:, :, :2].to(dv)
            pred = model(batch['image'].to(dv))['keypoints']
            errors.append(torch.sqrt(((pred - gt) ** 2).sum(dim=2)).cpu().numpy())
    errors = np.concatenate(errors, axis=0)
    pp_mae = errors.mean(axis=0)
    overall = errors.mean()

    print(f'整体MAE: {overall:.4f} (≈{overall*args.img_size:.0f}px)')
    print(f'PCK@0.05: {(errors<0.05).mean():.1%}  PCK@0.10: {(errors<0.10).mean():.1%}')
    for i, n in enumerate(NAMES):
        print(f'  {n}: MAE={pp_mae[i]:.4f} (≈{pp_mae[i]*args.img_size:.0f}px)')


if __name__ == '__main__':
    main()
