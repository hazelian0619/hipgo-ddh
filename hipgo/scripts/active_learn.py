#!/usr/bin/env python
"""主动学习：TTA不确定性采样，选最不确定的图优先标注"""
import os, argparse, json, torch, cv2, numpy as np
from PIL import Image

from hipgo.dataset import get_transforms
from hipgo.models import CNNKeypoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--unlabeled_dir', required=True)
    p.add_argument('--output_dir', default='outputs/active_learning')
    p.add_argument('--top_k', type=int, default=20)
    p.add_argument('--tta_passes', type=int, default=10)
    p.add_argument('--img_size', type=int, default=512)
    args = p.parse_args()

    dv = torch.device('cpu')
    ckpt = torch.load(args.model_path, map_location=dv)
    model = CNNKeypoint().to(dv)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    files = sorted([f for f in os.listdir(args.unlabeled_dir)
                    if f.endswith(('.jpg', '.jpeg', '.png')) and not f.endswith('.json')])
    print(f'未标注图: {len(files)} 张, TTA {args.tta_passes}次')

    train_tf = get_transforms(train=True, img_size=args.img_size)
    results = []

    for fname in files:
        img_path = os.path.join(args.unlabeled_dir, fname)
        all_preds = []
        for _ in range(args.tta_passes):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            t = train_tf(image=img, keypoints=[(0, 0)] * 9)
            img_t = t['image'].unsqueeze(0).to(dv)
            with torch.no_grad():
                pred = model(img_t)['keypoints'][0].cpu().numpy()
            all_preds.append(pred)

        all_preds = np.stack(all_preds)  # [T, 9, 2]
        per_pt_std = np.sqrt((all_preds.std(axis=0) ** 2).sum(axis=1))
        uncertainty = float(per_pt_std.mean())
        results.append({'image_name': fname, 'uncertainty': uncertainty})

    results.sort(key=lambda r: r['uncertainty'], reverse=True)
    os.makedirs(args.output_dir, exist_ok=True)

    top = results[:args.top_k]
    print(f'\nTop {args.top_k} 最不确定的图:')
    for i, r in enumerate(top, 1):
        print(f'  {i:2d}. {r["image_name"]}  uncertainty={r["uncertainty"]:.4f}')

    with open(os.path.join(args.output_dir, 'to_annotate.txt'), 'w') as f:
        for r in top:
            f.write(f'{r["image_name"]}\n')
    print(f'\n标注列表: {args.output_dir}/to_annotate.txt')


if __name__ == '__main__':
    main()
