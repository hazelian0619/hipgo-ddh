#!/usr/bin/env python
"""训练CNN关键点检测模型"""
import os, argparse, torch, numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from hipgo.dataset import HipKeypointDataset, get_transforms
from hipgo.models import CNNKeypoint

# 加权loss权重（基于规范化std，点5耻骨联合因个体差异大降低权重）
_POINT_STDS = [0.017, 0.017, 0.051, 0.055, 0.073, 0.029, 0.041, 0.028, 0.048]
_RAW = [1.0 / (1.0 + s) for s in _POINT_STDS]
_KEYPOINT_WEIGHTS = torch.tensor([w / sum(_RAW) * 9 for w in _RAW], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = HipKeypointDataset(args.data_dir, get_transforms(True, args.img_size), True, args.split_ratio)
    val_ds = HipKeypointDataset(args.data_dir, get_transforms(False, args.img_size), False, args.split_ratio)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CNNKeypoint().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    w = _KEYPOINT_WEIGHTS.to(device)
    best_loss = float('inf')
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            imgs = batch['image'].to(device)
            kps = batch['keypoints'][:, :, :2].to(device)
            pred = model(imgs)['keypoints']
            loss = (((pred - kps) ** 2).mean(dim=2) * w.unsqueeze(0)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch['image'].to(device)
                kps = batch['keypoints'][:, :, :2].to(device)
                pred = model(imgs)['keypoints']
                val_loss += (((pred - kps) ** 2).mean(dim=2) * w.unsqueeze(0)).mean().item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': val_loss, 'epoch': epoch + 1},
                       os.path.join(args.output_dir, 'best_model.pth'))

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  best={best_loss:.6f}')

    print(f'Done. Best val_loss={best_loss:.6f}')


if __name__ == '__main__':
    main()
