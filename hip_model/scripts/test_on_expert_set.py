#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在200张医生标签测试集上跑完整管线：预测关键点→算角度→对比医生标签"""
import os, sys, json, torch, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone.feature_extractor import FeatureExtractor
from dataset import get_transforms

class CNNO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = FeatureExtractor(pretrained=False, feature_dim=256)
        self.gp = torch.nn.AdaptiveAvgPool2d(1)
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(True),
            torch.nn.Linear(128, 128), torch.nn.ReLU(True),
            torch.nn.Linear(128, 18))
    def forward(self, x):
        _, mf = self.fe(x); gf = self.gp(mf).view(x.size(0), -1)
        return {'keypoints': torch.sigmoid(self.pred(gf).view(x.size(0), 9, 2))}

def calc_angles(kps):
    kps = np.array(kps)
    h_ref = kps[1] - kps[0]; h_ref /= np.linalg.norm(h_ref)
    v_ref = np.array([h_ref[1], -h_ref[0]]) / np.linalg.norm(np.array([h_ref[1], -h_ref[0]]))
    def ang(v1, v2):
        d = np.dot(v1, v2); n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.degrees(np.arccos(np.clip(d/n, -1, 1))))
    return {
        'left_ce': ang(v_ref, kps[2]-kps[0]), 'right_ce': ang(v_ref, kps[3]-kps[1]),
        'left_sharp': ang(h_ref, kps[2]-kps[5]), 'right_sharp': ang(h_ref, kps[3]-kps[7]),
        'left_tonnis': ang(h_ref, kps[6]-kps[5]), 'right_tonnis': ang(h_ref, kps[8]-kps[7]),
    }

# ---- 加载模型和数据 ----
ckpt = torch.load('outputs/best_model/best_cnn_model.pth', map_location='cpu')
model = CNNO(); model.load_state_dict(ckpt['model_state_dict']); model.eval()
print(f'模型: best_cnn_model.pth (loss={ckpt["loss"]:.4f})')

# 医生标签
label_dir = '/Users/pluviophile/hip/专家验证一致性/医生标签'
label_map = {}
for cls in os.listdir(label_dir):
    cp = os.path.join(label_dir, cls)
    if not os.path.isdir(cp): continue
    for sub in os.listdir(cp):
        sp = os.path.join(cp, sub)
        if os.path.isdir(sp):
            for f in os.listdir(sp):
                if f.endswith('.jpg'): label_map[f] = cls
        elif sub.endswith('.jpg'):
            label_map[sub] = cls

print(f'医生标签: {len(label_map)} 张')

# 测试集图片
expert_dir = '/Users/pluviophile/hip/shared_data/expert_validation_dataset'
test_files = sorted([f for f in os.listdir(expert_dir) if f.endswith('.jpg') and f in label_map])
print(f'测试集: {len(test_files)} 张 (有医生标签的)\n')

# ---- 逐张推理 ----
tf = get_transforms(train=False, img_size=512)
results = []
for i, fname in enumerate(test_files):
    img_path = os.path.join(expert_dir, fname)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    t = tf(image=img, keypoints=[(0, 0)] * 9)
    img_t = t['image'].unsqueeze(0)
    with torch.no_grad():
        pred = model(img_t)['keypoints'][0].cpu().numpy()
    angles = calc_angles(pred)
    doctor = label_map[fname]

    # CE角诊断规则: <20°高度怀疑DDH, <25°边界
    left_ddh = angles['left_ce'] < 25
    right_ddh = angles['right_ce'] < 25
    if left_ddh and right_ddh:
        pred_diag = '双1'  # 双侧DDH
    elif left_ddh or right_ddh:
        pred_diag = '单1'  # 单侧
    else:
        pred_diag = '双0'  # 正常

    results.append({'file': fname, 'angles': angles, 'doctor': doctor, 'pred': pred_diag,
                    'left_ce': angles['left_ce'], 'right_ce': angles['right_ce']})

# ---- 统计 ----
correct = sum(1 for r in results if r['pred'] == r['doctor'])
acc = correct / len(results) if results else 0
print(f'诊断准确率: {correct}/{len(results)} = {acc:.1%}')
print(f'\n前10张详情:')
for r in results[:10]:
    ok = '✅' if r['pred'] == r['doctor'] else '❌'
    print(f'  {ok} {r["file"]:15s} 医生={r["doctor"]:4s} 预测={r["pred"]:4s}  CE左={r["left_ce"]:.1f}° CE右={r["right_ce"]:.1f}°')

# 混淆矩阵
from collections import Counter
cm = Counter()
for r in results:
    cm[(r['doctor'], r['pred'])] += 1
print(f'\n混淆矩阵:')
for (d, p), c in sorted(cm.items()):
    print(f'  医生={d} 预测={p}: {c}张')

os.makedirs('outputs/test_expert', exist_ok=True)
with open('outputs/test_expert/expert_predictions.json', 'w') as f:
    json.dump([{k: r[k] for k in ['file','doctor','pred','left_ce','right_ce']} for r in results], f, indent=2, ensure_ascii=False)
print(f'\n详情保存: outputs/test_expert/expert_predictions.json')
