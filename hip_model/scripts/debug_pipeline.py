"""逐环节拆解管线：分析52.3%准确率的根因"""
import os, json, torch, numpy as np, cv2, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone.feature_extractor import FeatureExtractor
from dataset import get_transforms

class CNNO(torch.nn.Module):
    def __init__(self): super().__init__()
        self.fe=FeatureExtractor(pretrained=False,feature_dim=256)
        self.gp=torch.nn.AdaptiveAvgPool2d(1)
        self.pred=torch.nn.Sequential(torch.nn.Linear(256,128),torch.nn.ReLU(True),torch.nn.Linear(128,128),torch.nn.ReLU(True),torch.nn.Linear(128,18))
    def forward(self,x): _,mf=self.fe(x); gf=self.gp(mf).view(x.size(0),-1); return {'keypoints':torch.sigmoid(self.pred(gf).view(x.size(0),9,2))}

ckpt=torch.load('outputs/final/best_80img_30ep.pth',map_location='cpu')
m=CNNO(); m.load_state_dict(ckpt['state_dict']); m.eval()

def calc_angles(kps):
    kps=np.array(kps)
    h_ref=kps[1]-kps[0]; h_ref/=np.linalg.norm(h_ref)
    v_ref=np.array([h_ref[1],-h_ref[0]])/np.linalg.norm(np.array([h_ref[1],-h_ref[0]]))
    def ang(v1,v2): d=np.dot(v1,v2); n=np.linalg.norm(v1)*np.linalg.norm(v2); return round(float(np.degrees(np.arccos(np.clip(d/n,-1,1)))),1)
    def acute(v1,v2): r=ang(v1,v2); return min(r,180-r)
    return {'CE左':ang(v_ref,kps[2]-kps[0]),'CE右':ang(v_ref,kps[3]-kps[1]),'Sharp左':acute(h_ref,kps[2]-kps[5]),'Sharp右':acute(h_ref,kps[3]-kps[7]),'Tonnis左':ang(v_ref,kps[6]-kps[5]),'Tonnis右':ang(v_ref,kps[8]-kps[7])}

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

expert_dir='/Users/pluviophile/hip/shared_data/expert_validation_dataset'
tf=get_transforms(train=False,img_size=512)

# 重算所有角度
test_files=sorted([f for f in os.listdir(expert_dir) if f.endswith('.jpg') and f in label_map])
all_data=[]
for fname in test_files:
    img=cv2.cvtColor(cv2.imread(os.path.join(expert_dir,fname)),cv2.COLOR_BGR2RGB)
    t=tf(image=img,keypoints=[(0,0)]*9); img_t=t['image'].unsqueeze(0)
    with torch.no_grad(): pred=m(img_t)['keypoints'][0].cpu().numpy()
    a=calc_angles(pred)
    doctor=label_map[fname]
    issues=sum([a['CE左']<20,a['CE右']<20,a['Sharp左']>47,a['Sharp右']>47,a['Tonnis左']>12,a['Tonnis右']>12])
    rule='双1' if issues>=3 else ('单1' if issues>=1 else '双0')
    all_data.append({'file':fname,'doctor':doctor,'rule':rule,'match':rule==doctor,**a})

# === 分析1: 正确组 vs 错误组的CE角度 ===
correct=[d for d in all_data if d['match']]
wrong=[d for d in all_data if not d['match']]
print(f'=== 环节1: CE角分布 ===')
print(f'正确组({len(correct)}张): CE左={np.mean([d["CE左"] for d in correct]):.1f}±{np.std([d["CE左"] for d in correct]):.1f}  CE右={np.mean([d["CE右"] for d in correct]):.1f}±{np.std([d["CE右"] for d in correct]):.1f}')
print(f'错误组({len(wrong)}张):  CE左={np.mean([d["CE左"] for d in wrong]):.1f}±{np.std([d["CE左"] for d in wrong]):.1f}  CE右={np.mean([d["CE右"] for d in wrong]):.1f}±{np.std([d["CE右"] for d in wrong]):.1f}')

# === 分析2: 错误原因细分 ===
print(f'\n=== 环节2: 错误原因拆解 (共{len(wrong)}张错误) ===')
ce_only_one = sum(1 for d in wrong if (d['CE左']<20 and d['CE右']>=20) or (d['CE左']>=20 and d['CE右']<20))
ce_both_normal = sum(1 for d in wrong if d['CE左']>=20 and d['CE右']>=20)
ce_both_low = sum(1 for d in wrong if d['CE左']<20 and d['CE右']<20)
print(f'  仅一侧CE<20°(模型判单1,医生标双1): {ce_only_one}张')
print(f'  两侧CE都≥20°(模型判正常,医生标双1): {ce_both_normal}张')
print(f'  两侧CE都<20°但其他指标不够: {ce_both_low}张')

# === 分析3: 是否存在左右系统性偏差 ===
print(f'\n=== 环节3: 左右侧CE角系统性偏差 ===')
all_ce_l = [d['CE左'] for d in all_data]
all_ce_r = [d['CE右'] for d in all_data]
print(f'整体: CE左={np.mean(all_ce_l):.1f}±{np.std(all_ce_l):.1f}  CE右={np.mean(all_ce_r):.1f}±{np.std(all_ce_r):.1f}')
left_lower = sum(1 for d in all_data if d['CE左'] < d['CE右'])
right_lower = sum(1 for d in all_data if d['CE右'] < d['CE左'])
print(f'CE左<CE右: {left_lower}张  CE右<CE左: {right_lower}张')
print(f'→ {\"左侧CE角系统性偏低\" if left_lower > 1.3*right_lower else (\"右侧CE角系统性偏低\" if right_lower > 1.3*left_lower else \"无明显系统偏差\")}')

# === 分析4: 医生标签结构 ===
print(f'\n=== 环节4: 医生标签分布 ===')
from collections import Counter
label_dist = Counter(d['doctor'] for d in all_data)
for k,v in sorted(label_dist.items()):
    print(f'  {k}: {v}张')

# === 分析5: 用训练集GT标注验证角度公式 ===
print(f'\n=== 环节5: 训练集GT角度验证 ===')
data_dir='/Users/pluviophile/hip/shared_data/data/raw_images'
train_files=sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[:5]
for fn in train_files:
    with open(os.path.join(data_dir,fn)) as f: ann=json.load(f)
    W,H=ann['imageWidth'],ann['imageHeight']
    ss=sorted(ann['shapes'],key=lambda x:int(x['label']))
    gt_kps=np.array([[s['points'][0][0]/W,s['points'][0][1]/H] for s in ss])
    a=calc_angles(gt_kps)
    diag='DDH' if a['CE左']<20 or a['CE右']<20 else '正常'
    print(f'{fn}: CE左={a[\"CE左\"]}° CE右={a[\"CE右\"]}° Sharp左={a[\"Sharp左\"]}° Sharp右={a[\"Sharp右\"]}° Tonnis左={a[\"Tonnis左\"]}° Tonnis右={a[\"Tonnis右\"]}° → {diag}')

# === 分析6: 典型错误案例详情 ===
print(f'\n=== 环节6: 前10个错误案例 ===')
for d in wrong[:10]:
    print(f'{d[\"file\"]}: 医生={d[\"doctor\"]} 规则={d[\"rule\"]} | CE左={d[\"CE左\"]:.1f}° CE右={d[\"CE右\"]:.1f}° Sharp左={d[\"Sharp左\"]:.1f}° Sharp右={d[\"Sharp右\"]:.1f}° Tonnis左={d[\"Tonnis左\"]:.1f}° Tonnis右={d[\"Tonnis右\"]:.1f}°')
