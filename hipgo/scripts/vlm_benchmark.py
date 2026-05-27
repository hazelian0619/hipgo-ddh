#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验D: 开源视觉大模型零样本关键点检测评估

用 Ollama 本地运行的视觉模型（LLaVA / Qwen-VL / MiniCPM-V）直接看X光片，
尝试找出9个解剖关键点，无需任何训练。

用法:
    python hip_analysis/local_vlm_eval.py \
        --data_dir shared_data/data/raw_images \
        --model llava:13b --num_samples 10

前提: ollama pull llava:13b (或其他视觉模型)
"""

import os, sys, json, base64, argparse, time, re
import numpy as np
from io import BytesIO
from PIL import Image

# Ollama SDK
try:
    import ollama
except ImportError:
    print("请安装: pip install ollama")
    sys.exit(1)

KEYPOINT_PROMPT = """你是骨科放射科医生。看这张骨盆正位X光片，找出以下9个解剖关键点。

请按JSON格式回复，坐标用归一化值(x,y)，范围0-1，左上角为原点：
{
  "kps": {
    "1": [x, y],
    "2": [x, y],
    "3": [x, y],
    "4": [x, y],
    "5": [x, y],
    "6": [x, y],
    "7": [x, y],
    "8": [x, y],
    "9": [x, y]
  }
}

9个点的定义：
1-左侧股骨头圆心 2-右侧股骨头圆心 3-左髋臼外缘 4-右髋臼外缘
5-耻骨联合上缘中点 6-左髋臼荷重面内侧 7-左髋臼荷重面外侧
8-右髋臼荷重面内侧 9-右髋臼荷重面外侧

只回复JSON。"""


def encode_image(image_path, max_size=1024):
    """读图转base64"""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def parse_response(text):
    """从模型回复里提取9个关键点坐标"""
    # 先找JSON块
    json_match = re.search(r'\{[^{}]*"kps"[^{}]*\}', text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"[0-9]"[^{}]*\}', text, re.DOTALL)
    if not json_match:
        # 尝试找任何大括号内容
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group())
            kps = data.get('kps', data)
            coords = []
            for i in range(1, 10):
                key = str(i)
                if key in kps:
                    coords.append(kps[key])
                elif i in kps:
                    coords.append(kps[i])
                else:
                    return None
            return np.array(coords, dtype=np.float32)  # [9, 2]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # 尝试从文本中直接找数字对
    numbers = re.findall(r'[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]', text)
    if len(numbers) >= 9:
        coords = [(float(x), float(y)) for x, y in numbers[:9]]
        return np.array(coords, dtype=np.float32)

    return None


def call_ollama(model, image_path, retries=2):
    """调用Ollama视觉模型"""
    img_b64 = encode_image(image_path)

    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': KEYPOINT_PROMPT,
                    'images': [img_b64],
                }],
                options={'temperature': 0.1, 'num_predict': 1024},
            )
            text = response['message']['content']
            result = parse_response(text)
            if result is not None:
                return result
            if attempt < retries:
                time.sleep(1)

        except Exception as e:
            print(f"    Ollama调用失败(attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(2)

    return None


def evaluate(args):
    print(f"模型: ollama:{args.model}")
    print(f"数据: {args.data_dir}")
    print(f"样本数: {args.num_samples}\n")

    all_files = sorted([f for f in os.listdir(args.data_dir)
                        if f.endswith('.jpg') and os.path.exists(os.path.join(args.data_dir, f.replace('.jpg', '.json')))])

    import random; random.seed(42)
    sample_files = random.sample(all_files, min(args.num_samples, len(all_files)))

    print(f"从{len(all_files)}张中抽样{len(sample_files)}张\n")

    all_errors = []
    successful = 0
    failed = 0

    for i, fname in enumerate(sample_files, 1):
        img_path = os.path.join(args.data_dir, fname)
        ann_path = os.path.join(args.data_dir, fname.replace('.jpg', '.json'))

        with open(ann_path) as f:
            ann = json.load(f)
        W, H = ann['imageWidth'], ann['imageHeight']
        shapes = sorted(ann['shapes'], key=lambda x: int(x['label']))
        gt_kps = np.array([[s['points'][0][0] / W, s['points'][0][1] / H] for s in shapes], dtype=np.float32)

        print(f"[{i}/{len(sample_files)}] {fname} ...", end=' ', flush=True)
        pred_kps = call_ollama(args.model, img_path)

        if pred_kps is not None and pred_kps.shape == (9, 2):
            dist = np.sqrt(((pred_kps - gt_kps) ** 2).sum(axis=1))
            all_errors.append(dist)
            successful += 1
            print(f"MAE={dist.mean():.4f} (≈{dist.mean()*512:.0f}px)")
        else:
            failed += 1
            print("失败（坐标解析失败）")
            all_errors.append(np.full(9, np.nan))

        time.sleep(args.delay)

    all_errors = np.array(all_errors)
    valid = all_errors[~np.isnan(all_errors).any(axis=1)]

    if len(valid) == 0:
        print("\n所有样本均失败")
        return

    pp_mae = valid.mean(0)
    overall_mae = valid.mean()
    pck_05 = (valid < 0.05).mean()
    pck_10 = (valid < 0.10).mean()

    print(f"\n{'='*55}")
    print(f"  实验D: OSS VLM零样本 ({args.model})")
    print(f"  成功: {successful}/{len(sample_files)}, 失败: {failed}")
    print(f"{'='*55}")
    names = ['左股骨头','右股骨头','左髋臼外缘','右髋臼外缘','耻骨联合','左荷重面内','左荷重面外','右荷重面内','右荷重面外']
    for i, n in enumerate(names):
        print(f'  点{i+1} {n:12s}  MAE={pp_mae[i]:.4f}')
    print(f'  {"整体MAE":16s} {overall_mae:>8.4f} (≈{overall_mae*512:.0f}px)')
    print(f'  PCK@0.05 = {pck_05:.1%}')
    print(f'  PCK@0.10 = {pck_10:.1%}')
    print(f'{"="*55}')

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'model': f'ollama:{args.model}', 'successful': successful, 'failed': failed,
        'per_point_mae': pp_mae.tolist(), 'overall_mae': float(overall_mae),
        'pck_05': float(pck_05), 'pck_10': float(pck_10),
    }
    with open(os.path.join(args.output_dir, f'vlm_{args.model.replace(":","_")}_result.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'结果已保存: {args.output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../shared_data/data/raw_images')
    parser.add_argument('--output_dir', default='outputs/vlm_eval')
    parser.add_argument('--model', default='llava:13b')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    evaluate(args)
