#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""消融实验快速启动：B(无预训练) / C(无GAT)"""

import os, sys, json, argparse, numpy as np, torch, torch.nn as nn, cv2
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_gat_model import CNN_GAT
from models.backbone.feature_extractor import FeatureExtractor
from dataset import get_transforms

_ps = [0.017, 0.017, 0.051, 0.055, 0.073, 0.029, 0.041, 0.028, 0.048]
_w = torch.tensor([(1/(1+s))/sum(1/(1+ss) for ss in _ps)*9 for s in _ps])

class FD(Dataset):
    def __init__(s, d, fl, t, tr, fp=0.5): s.d,s.fl,s.t,s.tr,s.fp = d,fl,t,tr,fp
    def __len__(s): return len(s.fl)
    def __getitem__(s, i):
        n = s.fl[i]; p = os.path.join(s.d,n)
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        with open(os.path.join(s.d,n.replace('.jpg','.json'))) as f: a = json.load(f)
        sh = sorted(a['shapes'], key=lambda x: int(x['label']))
        kp = np.array([[x['points'][0][0],x['points'][0][1]] for x in sh], dtype=np.float32)
        oh,ow = im.shape[:2]
        if s.tr and np.random.rand()<s.fp:
            im = im[:,::-1,:].copy(); kp[:,0] = ow-kp[:,0]
            sk = kp.copy()
            for u,v in [(0,1),(2,3),(5,7),(6,8)]: sk[u],sk[v]=kp[v].copy(),kp[u].copy()
            kp = sk
        if s.t: r = s.t(image=im, keypoints=[(x,y) for x,y in kp]); im,kp = r['image'], np.array(r['keypoints'],dtype=np.float32)
        nh = im.shape[-2] if isinstance(im,torch.Tensor) else im.shape[0]
        nw = im.shape[-1] if isinstance(im,torch.Tensor) else im.shape[1]
        kp[:,0]/=nw; kp[:,1]/=nh
        return {'image':im,'keypoints':torch.tensor(np.column_stack((kp,np.ones(9))),dtype=torch.float32)}

class CNN_Only(nn.Module):
    def __init__(s, pt=True, nk=9):
        super().__init__()
        s.fe = FeatureExtractor(pretrained=pt, feature_dim=256)
        s.gp = nn.AdaptiveAvgPool2d(1)
        s.pred = nn.Sequential(nn.Linear(256,128),nn.ReLU(True),nn.Linear(128,128),nn.ReLU(True),nn.Linear(128,nk*2))
        s.nk = nk
    def forward(s,x):
        _,mf = s.fe(x); gf = s.gp(mf).view(x.size(0),-1)
        return {'keypoints':torch.sigmoid(s.pred(gf).view(x.size(0),s.nk,2)),'angles':torch.zeros(x.size(0),6,device=x.device)}

def train_fold(model, tld, vld, ep, lr, dv):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    w = _w.to(dv); bv,bs,pc = float('inf'),None,0
    for e in range(ep):
        model.train()
        for b in tld:
            im,gt = b['image'].to(dv),b['keypoints'][:,:,:2].to(dv)
            loss = (((model(im)['keypoints']-gt)**2).mean(dim=2)*w.unsqueeze(0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vl=0.0
        with torch.no_grad():
            for b in vld:
                im,gt=b['image'].to(dv),b['keypoints'][:,:,:2].to(dv)
                vl+=(((model(im)['keypoints']-gt)**2).mean(dim=2)*w.unsqueeze(0)).mean().item()
        vl/=len(vld); sch.step(vl)
        if vl<bv: bv=vl; pc=0; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: pc+=1
        if pc>=10: break
    model.load_state_dict(bs); model.eval(); ae=[]
    with torch.no_grad():
        for b in vld:
            im,gt=b['image'].to(dv),b['keypoints'][:,:,:2].to(dv)
            ae.append(torch.sqrt(((model(im)['keypoints']-gt)**2).sum(dim=2)).cpu().numpy())
    e=np.concatenate(ae,0)
    return e.mean(0),e.mean(),(e<0.05).mean(),(e<0.10).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=['B','C'])
    ap.add_argument('--data_dir', default='/Users/pluviophile/hip/shared_data/data/raw_images')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.0001)
    args=ap.parse_args()
    dv = torch.device('cpu')
    af = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(args.data_dir,f.replace('.jpg','.json')))])
    kf=KFold(5,shuffle=True,random_state=42); fl = list(kf.split(af))
    if args.exp=='B':
        mf = lambda: CNN_GAT(num_keypoints=9,num_angles=6,pretrained=False)
        nm = 'B: CNN-GAT (无预训练)'
    else:
        mf = lambda: CNN_Only(pt=True)
        nm = 'C: 纯CNN (无GAT)'
    print(f'\n实验 {nm}')
    ppl,ovl,p5l,p10l = [],[],[],[]
    for fi,(ti,vi) in enumerate(fl,1):
        td=FD(args.data_dir,[af[i] for i in ti],get_transforms(True,512),True)
        vd=FD(args.data_dir,[af[i] for i in vi],get_transforms(False,512),False)
        tl=DataLoader(td,batch_size=args.batch_size,shuffle=True,num_workers=0)
        vl=DataLoader(vd,batch_size=args.batch_size,shuffle=False,num_workers=0)
        m=mf().to(dv); pp,ov,p5,p10=train_fold(m,tl,vl,args.epochs,args.lr,dv)
        ppl.append(pp); ovl.append(ov); p5l.append(p5); p10l.append(p10)
        print(f'  Fold {fi}: MAE={ov:.4f}  PCK@0.05={p5:.1%}  PCK@0.10={p10:.1%}')
    pp=np.array(ppl); ov=np.array(ovl); p5=np.array(p5l); p10=np.array(p10l)
    print(f'\n  整体MAE: {ov.mean():.4f} ± {ov.std():.4f}')
    print(f'  PCK@0.05: {p5.mean():.1%} ± {p5.std():.1%}')
    print(f'  PCK@0.10: {p10.mean():.1%} ± {p10.std():.1%}')
    os.makedirs('outputs/ablation', exist_ok=True)
    r={'name':nm,'per_point_mae':pp.mean(0).tolist(),'per_point_std':pp.std(0).tolist(),'overall_mae':float(ov.mean()),'overall_std':float(ov.std()),'pck_05':float(p5.mean()),'pck_05_std':float(p5.std()),'pck_10':float(p10.mean()),'pck_10_std':float(p10.std())}
    with open(f'outputs/ablation/exp_{args.exp}.json','w') as f: json.dump(r,f,indent=2,ensure_ascii=False)
    print(f'结果已保存: outputs/ablation/exp_{args.exp}.json')
if __name__=='__main__': main()
