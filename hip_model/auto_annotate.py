#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主动学习标注工具：MC Dropout不确定性采样 + 批量自动标注。

两种模式：
  1. uncertainty 模式（主动学习选图）：
     MC Dropout推理，输出"不确定性最高"的N张图，优先人工标注
  2. annotate 模式（批量自动标注）：
     用训练好的模型对所有无标注图预测关键点，保存JSON

用法：
  # 主动学习：选出最不确定的30张图
  python auto_annotate.py uncertainty \
      --model_path models/model_best_20250506_163007.pth \
      --data_dir /path/to/unlabeled/images \
      --output_dir outputs/active_learning_round1 \
      --mc_passes 10 --top_k 30

  # 批量标注：对所有图预测关键点
  python auto_annotate.py annotate \
      --model_path models/model_best_20250506_163007.pth \
      --data_dir /path/to/images \
      --output_dir outputs/annotations
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.cnn_gat_model import CNN_GAT_Model
from utils.transforms import get_prediction_transforms


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png'))
                            and not f.endswith('.json')]
        print(f"找到 {len(self.image_files)} 个图像文件")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']

        return {'image': image, 'image_name': img_name}


def load_model(model_path, device):
    """加载训练好的CNN-GAT模型"""
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = CNN_GAT_Model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model


def mc_dropout_inference(model, dataloader, device, mc_passes=10):
    """
    MC Dropout 推理：多次前向传播，每次dropout随机激活，
    通过多次预测的方差量化不确定性。

    返回:
        results: [{image_name, mean_kps, std_kps, uncertainty}, ...]
        按uncertainty从高到低排序
    """
    model.train()  # 开启dropout（普通eval会关闭）

    results = []
    for batch in tqdm(dataloader, desc="MC Dropout推理"):
        images = batch['image'].to(device)       # [B, 3, 512, 512]
        image_names = batch['image_name']
        B = images.size(0)

        # 多次前向传播
        all_preds = []  # [mc_passes, B, 9, 2]
        with torch.no_grad():
            for _ in range(mc_passes):
                outputs = model(images)
                all_preds.append(outputs['keypoints'].cpu().numpy())

        all_preds = np.stack(all_preds, axis=0)   # [mc_passes, B, 9, 2]

        for b in range(B):
            preds = all_preds[:, b, :, :]          # [mc_passes, 9, 2]
            mean_kps = preds.mean(axis=0)           # [9, 2]
            std_kps  = preds.std(axis=0)            # [9, 2]

            # 每个点的标准差，对所有点取平均 = 该图的不确定度
            per_point_std = np.sqrt((std_kps ** 2).sum(axis=1))  # [9]
            uncertainty   = float(per_point_std.mean())           # 标量

            results.append({
                'image_name':   image_names[b],
                'mean_kps':     mean_kps.tolist(),
                'per_point_std': per_point_std.tolist(),
                'uncertainty':  uncertainty,
            })

    results.sort(key=lambda r: r['uncertainty'], reverse=True)
    return results


def cmd_uncertainty(args):
    """主动学习模式：选出不确定性最高的图"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model = load_model(args.model_path, device)
    print(f"模型加载完成: {args.model_path}")

    transform = get_prediction_transforms(img_size=args.image_size)
    dataset = ImageDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    print(f"无标注图: {len(dataset)} 张, MC Dropout次数: {args.mc_passes}")

    results = mc_dropout_inference(model, dataloader, device, mc_passes=args.mc_passes)

    # ---- 输出 ----
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 完整排序列表
    summary_path = os.path.join(args.output_dir, "uncertainty_ranking.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n完整排序已保存: {summary_path}")

    # 2. Top-K 最不确定的图
    top_k = results[:args.top_k]
    top_path = os.path.join(args.output_dir, f"top{args.top_k}_to_annotate.txt")
    with open(top_path, 'w') as f:
        for r in top_k:
            f.write(f"{r['image_name']}  uncertainty={r['uncertainty']:.6f}\n")
    print(f"优先标注列表: {top_path}")

    # 3. 终端展示 Top-10
    print(f"\n{'='*55}")
    print(f"  Top {min(10, args.top_k)} 最不确定的图（优先人工标注）")
    print(f"{'='*55}")
    print(f"  {'图像':30s}  {'不确定度':>10s}  {'各点std(点1~9)'}")
    print(f"  {'-'*53}")
    for r in top_k[:10]:
        pts = ' '.join(f'{s:.4f}' for s in r['per_point_std'])
        print(f"  {r['image_name']:30s}  {r['uncertainty']:>10.4f}  {pts}")

    # 4. 不确定度分布统计
    uncerts = [r['uncertainty'] for r in results]
    print(f"\n不确定度分布: min={min(uncerts):.4f}, max={max(uncerts):.4f}, "
          f"median={np.median(uncerts):.4f}, mean={np.mean(uncerts):.4f}")

    # 5. 低不确定度图（模型有信心，可考虑直接采信伪标注）
    if args.save_confident:
        low_k = results[-args.top_k:] if args.top_k < len(results) else []
        confident_path = os.path.join(args.output_dir, f"top{args.top_k}_confident.json")
        confident_data = [{'image_name': r['image_name'],
                           'keypoints': r['mean_kps']} for r in low_k]
        with open(confident_path, 'w') as f:
            json.dump(confident_data, f, indent=2)
        print(f"高置信度伪标注: {confident_path}")

    print("\n主动学习选图完成!")


def cmd_annotate(args):
    """批量标注模式：对所有图预测关键点并保存JSON"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model = load_model(args.model_path, device)
    model.eval()
    print(f"模型加载完成: {args.model_path}")

    transform = get_prediction_transforms(img_size=args.image_size)
    dataset = ImageDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    print(f"数据集大小: {len(dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="自动标注"):
            images = batch['image'].to(device)
            image_names = batch['image_name']
            outputs = model(images)
            keypoints = outputs['keypoints'].cpu().numpy()

            for i, img_name in enumerate(image_names):
                result = {
                    'image_name': img_name,
                    'keypoints': keypoints[i].tolist()
                }
                output_path = os.path.join(
                    args.output_dir,
                    os.path.splitext(img_name)[0] + '.json'
                )
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

    print(f"自动标注完成! 输出目录: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='主动学习标注工具')
    subparsers = parser.add_subparsers(dest='cmd', help='模式: uncertainty 或 annotate')

    # ---- uncertainty 子命令 ----
    p_unc = subparsers.add_parser('uncertainty', help='MC Dropout不确定性采样')
    p_unc.add_argument('--model_path', required=True)
    p_unc.add_argument('--data_dir', required=True)
    p_unc.add_argument('--output_dir', default='outputs/active_learning')
    p_unc.add_argument('--mc_passes', type=int, default=10)
    p_unc.add_argument('--top_k', type=int, default=30)
    p_unc.add_argument('--save_confident', action='store_true')
    p_unc.add_argument('--batch_size', type=int, default=4)
    p_unc.add_argument('--num_workers', type=int, default=2)
    p_unc.add_argument('--image_size', type=int, default=512)

    # ---- annotate 子命令 ----
    p_ann = subparsers.add_parser('annotate', help='批量自动标注')
    p_ann.add_argument('--model_path', required=True)
    p_ann.add_argument('--data_dir', required=True)
    p_ann.add_argument('--output_dir', required=True)
    p_ann.add_argument('--batch_size', type=int, default=4)
    p_ann.add_argument('--num_workers', type=int, default=2)
    p_ann.add_argument('--image_size', type=int, default=512)

    args = parser.parse_args()

    if args.cmd == 'uncertainty':
        cmd_uncertainty(args)
    elif args.cmd == 'annotate':
        cmd_annotate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
