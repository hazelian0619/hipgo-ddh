#!/usr/bin/env python
"""专家测试集完整管线：关键点预测 → 角度 → 诊断 → 对比医生标签"""
import os, json, argparse, torch, cv2, numpy as np
from collections import Counter

from hipgo.dataset import get_transforms
from hipgo.models import CNNKeypoint
from hipgo.angles import calculate_angles, diagnose


def load_doctor_labels(label_dir):
    """从目录结构加载医生标签 {文件名: 标签}"""
    labels = {}
    for cls_name in os.listdir(label_dir):
        cls_path = os.path.join(label_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for item in os.listdir(cls_path):
            item_path = os.path.join(cls_path, item)
            if os.path.isdir(item_path):
                for f in os.listdir(item_path):
                    if f.endswith('.jpg'):
                        labels[f] = cls_name
            elif item.endswith('.jpg'):
                labels[item] = cls_name
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--expert_dir', required=True, help='专家测试图片目录')
    p.add_argument('--label_dir', required=True, help='医生标签目录（含双0/单1/双1/双2子目录）')
    p.add_argument('--output_dir', default='outputs/expert_test')
    p.add_argument('--ce_thr', type=float, default=25, help='CE角阈值')
    p.add_argument('--img_size', type=int, default=512)
    args = p.parse_args()

    dv = torch.device('cpu')
    ckpt = torch.load(args.model_path, map_location=dv)
    model = CNNKeypoint().to(dv)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    labels = load_doctor_labels(args.label_dir)
    test_files = sorted([f for f in os.listdir(args.expert_dir)
                         if f.endswith('.jpg') and f in labels])
    print(f'测试: {len(test_files)} 张, CE阈值={args.ce_thr}°')

    tf = get_transforms(train=False, img_size=args.img_size)
    results, correct = [], 0

    for fname in test_files:
        img = cv2.cvtColor(cv2.imread(os.path.join(args.expert_dir, fname)), cv2.COLOR_BGR2RGB)
        t = tf(image=img, keypoints=[(0, 0)] * 9)
        img_t = t['image'].unsqueeze(0).to(dv)
        with torch.no_grad():
            pred = model(img_t)['keypoints'][0].cpu().numpy()
        angles = calculate_angles(pred)
        doctor = labels[fname]
        diag = diagnose(angles, ce_thr=args.ce_thr)
        match = diag == doctor
        if match:
            correct += 1
        results.append({'file': fname, 'doctor': doctor, 'pred': diag, 'match': match, **angles})

    acc = correct / len(results)
    print(f'准确率: {correct}/{len(results)} = {acc:.1%}\n')

    cm = Counter((r['doctor'], r['pred']) for r in results)
    print('混淆矩阵:')
    for (d, p), c in sorted(cm.items()):
        ok = '✅' if d == p else '❌'
        print(f'  {ok} 医生={d} 模型={p}: {c}张')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({'accuracy': acc, 'total': len(results), 'thresholds': f'CE<{args.ce_thr},Sharp>45,Tonnis>10',
                   'confusion_matrix': {f'{k[0]}->{k[1]}': v for k, v in cm.items()}}, f, indent=2, ensure_ascii=False)
    print(f'\n结果: {args.output_dir}/results.json')


if __name__ == '__main__':
    main()
