#!/usr/bin/env python
"""5折交叉验证"""
import os, argparse, json, torch, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from hipgo.dataset import get_transforms
from hipgo.models import CNNKeypoint

_POINT_STDS = [0.017, 0.017, 0.051, 0.055, 0.073, 0.029, 0.041, 0.028, 0.048]
_RAW = [1.0 / (1.0 + s) for s in _POINT_STDS]
_KPW = torch.tensor([w / sum(_RAW) * 9 for w in _RAW], dtype=torch.float32)


class FoldDataset(Dataset):
    def __init__(self, d, files, tf, train, flip_p=0.5):
        self.d, self.files, self.tf, self.train, self.flip_p = d, files, tf, train, flip_p

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        n = self.files[i]
        im = cv2.cvtColor(cv2.imread(os.path.join(self.d, n)), cv2.COLOR_BGR2RGB)
        with open(os.path.join(self.d, n.replace('.jpg', '.json'))) as f:
            ann = json.load(f)
        ss = sorted(ann['shapes'], key=lambda x: int(x['label']))
        kps = np.array([[s['points'][0][0], s['points'][0][1]] for s in ss], dtype=np.float32)
        oh, ow = im.shape[:2]
        if self.train and np.random.rand() < self.flip_p:
            im = im[:, ::-1, :].copy()
            kps[:, 0] = ow - kps[:, 0]
            sk = kps.copy()
            for u, v in [(0, 1), (2, 3), (5, 7), (6, 8)]:
                sk[u], sk[v] = kps[v].copy(), kps[u].copy()
            kps = sk
        r = self.tf(image=im, keypoints=[(x, y) for x, y in kps])
        im = r['image']
        kps = np.array(r['keypoints'], dtype=np.float32)
        nh = im.shape[-2] if isinstance(im, torch.Tensor) else im.shape[0]
        nw = im.shape[-1] if isinstance(im, torch.Tensor) else im.shape[1]
        kps[:, 0] /= nw
        kps[:, 1] /= nh
        return {'image': im, 'keypoints': torch.tensor(np.column_stack((kps, np.ones(9))), dtype=torch.float32)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', default='outputs/cv')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--n_folds', type=int, default=5)
    args = p.parse_args()

    dv = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.data_dir)
                    if f.endswith('.jpg') and os.path.exists(os.path.join(args.data_dir, f.replace('.jpg', '.json')))])
    kf = KFold(args.n_folds, shuffle=True, random_state=42)

    mae_list, pck05_list, pck10_list = [], [], []
    for fi, (ti, vi) in enumerate(kf.split(files), 1):
        td = FoldDataset(args.data_dir, [files[i] for i in ti], get_transforms(True, 512), True)
        vd = FoldDataset(args.data_dir, [files[i] for i in vi], get_transforms(False, 512), False)
        tl = DataLoader(td, batch_size=args.batch_size, shuffle=True, num_workers=0)
        vl = DataLoader(vd, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = CNNKeypoint().to(dv)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        w = _KPW.to(dv)
        best_vl, best_st, patience = float('inf'), None, 0

        for _ in range(args.epochs):
            model.train()
            for b in tl:
                im, gt = b['image'].to(dv), b['keypoints'][:, :, :2].to(dv)
                pred = model(im)['keypoints']
                loss = (((pred - gt) ** 2).mean(dim=2) * w.unsqueeze(0)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            vl_sum = 0.0
            with torch.no_grad():
                for b in vl:
                    im, gt = b['image'].to(dv), b['keypoints'][:, :, :2].to(dv)
                    vl_sum += (((model(im)['keypoints'] - gt) ** 2).mean(dim=2) * w.unsqueeze(0)).mean().item()
            vl_sum /= len(vl)
            if vl_sum < best_vl:
                best_vl = vl_sum
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

        model.load_state_dict(best_st)
        model.eval()
        errors = []
        with torch.no_grad():
            for b in vl:
                gt = b['keypoints'][:, :, :2].to(dv)
                pred = model(b['image'].to(dv))['keypoints']
                errors.append(torch.sqrt(((pred - gt) ** 2).sum(dim=2)).cpu().numpy())
        err = np.concatenate(errors, axis=0)
        mae_list.append(err.mean())
        pck05_list.append((err < 0.05).mean())
        pck10_list.append((err < 0.10).mean())
        print(f'Fold {fi}/{args.n_folds}: MAE={mae_list[-1]:.4f}  PCK@0.05={pck05_list[-1]:.1%}  PCK@0.10={pck10_list[-1]:.1%}')

    print(f'\nFinal: MAE={np.mean(mae_list):.4f}±{np.std(mae_list):.4f}  PCK@0.05={np.mean(pck05_list):.1%}  PCK@0.10={np.mean(pck10_list):.1%}')


if __name__ == '__main__':
    main()
