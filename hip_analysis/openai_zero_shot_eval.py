#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验D2: GPT-4o 零样本关键点检测评估

直接用GPT-4o看图，尝试找出9个解剖关键点，无需训练。
与专训CNN-GAT对比 = 证明通用大模型在精确医学定位上的局限性。

用法:
    python hip_analysis/openai_zero_shot_eval.py \
        --data_dir shared_data/data/raw_images \
        --num_samples 10

前提: export OPENAI_API_KEY="sk-..."
"""

import os, sys, json, base64, argparse, time, re
import numpy as np
from io import BytesIO
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    print("请安装: pip install openai")
    sys.exit(1)

KEYPOINT_PROMPT = """你是一名专业的骨科放射科医生。请仔细查看这张骨盆正位X光片，找出以下9个解剖关键点的精确位置。

对每个点，给出归一化坐标 (x, y)，其中x和y的范围是0到1，以图像左上角为原点：

1. 左侧股骨头中心点 - 左侧股骨头圆形轮廓的几何中心
2. 右侧股骨头中心点 - 右侧股骨头圆形轮廓的几何中心
3. 左侧髋臼外缘点 - 左侧髋臼上外侧骨皮质边缘最外侧点
4. 右侧髋臼外缘点 - 右侧髋臼上外侧骨皮质边缘最外侧点
5. 耻骨联合点 - 左右耻骨在骨盆正中间连接处的上缘中点
6. 左侧髋臼荷重面内侧点 - 左侧髋臼顶部承重关节面最内侧
7. 左侧髋臼荷重面外侧点 - 左侧髋臼顶部承重关节面最外侧
8. 右侧髋臼荷重面内侧点 - 右侧髋臼顶部承重关节面最内侧
9. 右侧髋臼荷重面外侧点 - 右侧髋臼顶部承重关节面最外侧

严格按照以下JSON格式回复，坐标值精确到小数点后4位：
{"1":[x,y],"2":[x,y],"3":[x,y],"4":[x,y],"5":[x,y],"6":[x,y],"7":[x,y],"8":[x,y],"9":[x,y]}

只回复JSON，不要任何其他文字。"""


def encode_image(image_path, max_size=1024):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def call_gpt4o(client, image_path):
    img_b64 = encode_image(image_path)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": KEYPOINT_PROMPT},
                    ]
                }],
                temperature=0.1,
                max_tokens=1024,
            )
            text = response.choices[0].message.content

            # 解析JSON
            json_match = re.search(r'\{[^}]*"1"[^}]*\}', text, re.DOTALL) or re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                coords = []
                for i in range(1, 10):
                    k = str(i)
                    if k in data and len(data[k]) == 2:
                        coords.append([float(data[k][0]), float(data[k][1])])
                    else:
                        coords.append([0.0, 0.0])  # fallback
                return np.array(coords, dtype=np.float32)

        except Exception as e:
            if attempt < 2:
                time.sleep(2)

    return None


def evaluate(args):
    api_key = os.environ.get("OPENAI_API_KEY", args.api_key)
    if not api_key:
        print("请设置 OPENAI_API_KEY 或通过 --api_key 传入")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    all_files = sorted([f for f in os.listdir(args.data_dir)
                        if f.endswith('.jpg') and os.path.exists(os.path.join(args.data_dir, f.replace('.jpg', '.json')))])

    import random; random.seed(42)
    sample_files = random.sample(all_files, min(args.num_samples, len(all_files)))

    print(f"模型: GPT-4o")
    print(f"评估 {len(sample_files)} 张图\n")

    all_errors, successful, failed = [], 0, 0

    for i, fname in enumerate(sample_files, 1):
        img_path = os.path.join(args.data_dir, fname)
        ann_path = os.path.join(args.data_dir, fname.replace('.jpg', '.json'))

        with open(ann_path) as f:
            ann = json.load(f)
        W, H = ann['imageWidth'], ann['imageHeight']
        shapes = sorted(ann['shapes'], key=lambda x: int(x['label']))
        gt_kps = np.array([[s['points'][0][0] / W, s['points'][0][1] / H] for s in shapes], dtype=np.float32)

        print(f"[{i}/{len(sample_files)}] {fname} ...", end=' ', flush=True)
        pred_kps = call_gpt4o(client, img_path)

        if pred_kps is not None and pred_kps.shape == (9, 2):
            dist = np.sqrt(((pred_kps - gt_kps) ** 2).sum(axis=1))
            all_errors.append(dist)
            successful += 1
            print(f"MAE={dist.mean():.4f}")
        else:
            failed += 1
            print("失败")
            all_errors.append(np.full(9, np.nan))

        time.sleep(0.5)  # rate limit

    all_errors = np.array(all_errors)
    valid = all_errors[~np.isnan(all_errors).any(axis=1)]

    if len(valid) == 0:
        print("\n所有样本均失败")
        return

    pp = valid.mean(0); ov = valid.mean()
    p5 = (valid < 0.05).mean(); p10 = (valid < 0.10).mean()

    print(f"\n{'='*55}")
    print(f"  实验D2: GPT-4o 零样本")
    print(f"  成功: {successful}/{len(sample_files)}")
    print(f"{'='*55}")
    print(f'  整体MAE: {ov:.4f} (≈{ov*512:.0f}px)')
    print(f'  PCK@0.05: {p5:.1%}  PCK@0.10: {p10:.1%}')
    print(f'{"="*55}')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'gpt4o_zero_shot.json'), 'w') as f:
        json.dump({'model':'gpt-4o','overall_mae':float(ov),'pck_05':float(p5),'pck_10':float(p10)}, f, indent=2)
    print(f'结果已保存')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../shared_data/data/raw_images')
    parser.add_argument('--output_dir', default='outputs/gpt4o_eval')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--api_key', default=None)
    args = parser.parse_args()
    evaluate(args)
