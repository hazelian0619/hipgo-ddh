#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终管线：模型预测关键点 → 6角度 → GPT-4o诊断 → 对比医生标签
"""
import os, sys, json, torch, numpy as np, cv2, requests, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone.feature_extractor import FeatureExtractor
from dataset import get_transforms

# ---- 模型 ----
class CNNO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = FeatureExtractor(pretrained=False, feature_dim=256)
        self.gp = torch.nn.AdaptiveAvgPool2d(1)
        self.pred = torch.nn.Sequential(torch.nn.Linear(256,128),torch.nn.ReLU(True),torch.nn.Linear(128,128),torch.nn.ReLU(True),torch.nn.Linear(128,18))
    def forward(self,x):
        _,mf=self.fe(x); gf=self.gp(mf).view(x.size(0),-1)
        return {'keypoints':torch.sigmoid(self.pred(gf).view(x.size(0),9,2))}

# ---- 角度计算（修正版）----
def calc_angles(kps):
    kps=np.array(kps)
    h_ref=kps[1]-kps[0]; h_ref/=np.linalg.norm(h_ref)
    v_ref=np.array([h_ref[1],-h_ref[0]])/np.linalg.norm(np.array([h_ref[1],-h_ref[0]]))
    def ang(v1,v2):
        d=np.dot(v1,v2); n=np.linalg.norm(v1)*np.linalg.norm(v2)
        return round(float(np.degrees(np.arccos(np.clip(d/n,-1,1)))),1)
    def ang_acute(v1,v2):
        raw=ang(v1,v2); return min(raw, 180-raw)  # 锐角
    return {
        'left_ce': ang(v_ref, kps[2]-kps[0]), 'right_ce': ang(v_ref, kps[3]-kps[1]),
        'left_sharp': ang_acute(h_ref, kps[2]-kps[5]), 'right_sharp': ang_acute(h_ref, kps[3]-kps[7]),
        'left_tonnis': ang(v_ref, kps[6]-kps[5]), 'right_tonnis': ang(v_ref, kps[8]-kps[7]),
    }

# ---- GPT-4o诊断 ----
API='https://aiberm.com/v1/chat/completions'
KEY='YOUR_API_KEY'

def gpt4o_diagnose(angles):
    prompt=f"""你是骨科放射科医生。根据以下骨盆X光片6个角度值判断DDH发育性髋关节发育不良：

左CE角: {angles['left_ce']}° (正常>25°, 边界20-25°, DDH<20°)
右CE角: {angles['right_ce']}° (正常>25°, 边界20-25°, DDH<20°)
左Sharp角: {angles['left_sharp']}° (正常<45°, 边界45-50°, DDH>50°)
右Sharp角: {angles['right_sharp']}° (正常<45°, 边界45-50°, DDH>50°)
左Tönnis角: {angles['left_tonnis']}° (正常<10°, 边界10-15°, DDH>15°)
右Tönnis角: {angles['right_tonnis']}° (正常<10°, 边界10-15°, DDH>15°)

请综合判断，只回复以下四个词之一：双0 双1 单1 双2
双0=双侧正常 单1=单侧DDH 双1=双侧DDH 双2=严重双侧DDH"""
    for _ in range(2):
        try:
            r=requests.post(API,headers={'Authorization':f'Bearer {KEY}','Content-Type':'application/json'},
                json={'model':'gpt-4o','max_tokens':10,'temperature':0,'messages':[{'role':'user','content':prompt}]},timeout=15)
            if r.status_code==200:
                txt=r.json()['choices'][0]['message']['content'].strip()
                m=re.search(r'双[012]|单1',txt)
                if m: return m.group()
            time.sleep(1)
        except: time.sleep(2)
    return '双1'  # 默认

# ---- 加载模型和数据 ----
ckpt=torch.load('outputs/final/best_80img_30ep.pth',map_location='cpu')
model=CNNO(); model.load_state_dict(ckpt['state_dict']); model.eval()
print(f'模型加载: loss={ckpt["loss"]:.4f} epoch={ckpt["epoch"]}')

# 医生标签
label_dir='/Users/pluviophile/hip/专家验证一致性/医生标签'
label_map={}
for cls in os.listdir(label_dir):
    cp=os.path.join(label_dir,cls)
    if not os.path.isdir(cp): continue
    for sub in os.listdir(cp):
        sp=os.path.join(cp,sub)
        if os.path.isdir(sp):
            for f in os.listdir(sp):
                if f.endswith('.jpg'): label_map[f]=cls
        elif sub.endswith('.jpg'): label_map[sub]=cls

# 测试集
expert_dir='/Users/pluviophile/hip/shared_data/expert_validation_dataset'
test_files=sorted([f for f in os.listdir(expert_dir) if f.endswith('.jpg') and f in label_map])
print(f'测试集: {len(test_files)}张\n')

tf=get_transforms(train=False,img_size=512)
results=[]
for i,fname in enumerate(test_files):
    # 预测关键点
    img=cv2.cvtColor(cv2.imread(os.path.join(expert_dir,fname)),cv2.COLOR_BGR2RGB)
    t=tf(image=img,keypoints=[(0,0)]*9); img_t=t['image'].unsqueeze(0)
    with torch.no_grad():
        pred=model(img_t)['keypoints'][0].cpu().numpy()
    angles=calc_angles(pred)
    doctor=label_map[fname]

    # GPT-4o诊断
    gpt_diag=gpt4o_diagnose(angles)
    match='✅' if gpt_diag==doctor else '❌'
    results.append({'file':fname,'doctor':doctor,'gpt':gpt_diag,'angles':angles,'match':match})

    if (i+1)%20==0 or i<5:
        print(f'  [{i+1}/{len(test_files)}] {fname} 医生={doctor} GPT={gpt_diag} {match}  CE左={angles["left_ce"]}° CE右={angles["right_ce"]}°')

# 统计
correct=sum(1 for r in results if r['match']=='✅')
acc=correct/len(results)
print(f'\n=== 最终诊断准确率: {correct}/{len(results)} = {acc:.1%} ===')

# 混淆矩阵
from collections import Counter
cm=Counter()
for r in results: cm[(r['doctor'],r['gpt'])]+=1
print('\n混淆矩阵:')
for (d,p),c in sorted(cm.items()):
    print(f'  医生={d} GPT={p}: {c}张')

os.makedirs('outputs/final',exist_ok=True)
with open('outputs/final/expert_diagnosis.json','w') as f:
    json.dump([{k:r[k] for k in ['file','doctor','gpt','angles','match']} for r in results],f,indent=2,ensure_ascii=False)
print(f'\n详情: outputs/final/expert_diagnosis.json')
