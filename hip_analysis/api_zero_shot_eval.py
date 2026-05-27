#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验D: 大模型API零样本关键点检测评估

直接让DeepSeek/Claude看图，不训练模型，看能不能找到解剖关键点。
与你的专训CNN-GAT对比 = 论文里最有力的消融实验之一。

用法:
    python hip_analysis/api_zero_shot_eval.py \
        --data_dir shared_data/data/raw_images \
        --num_samples 10 --output_dir outputs/api_eval

环境变量（已配置）:
    ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL
"""

import os, sys, json, base64, argparse, time
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path

# Anthropic SDK（兼容DeepSeek端点）
try:
    import anthropic
except ImportError:
    print("请安装: pip install anthropic")
    sys.exit(1)

# 9个解剖关键点的医学定义，给LLM看的prompt
KEYPOINT_PROMPT = """你是一个专业的骨科放射科医生。请看这张骨盆正位X光片，找出以下9个解剖关键点：

1. 左侧股骨头中心点 - 左侧股骨头圆形轮廓的几何中心
2. 右侧股骨头中心点 - 右侧股骨头圆形轮廓的几何中心
3. 左侧髋臼外缘点 - 左侧髋臼上外侧骨皮质边缘的最外点
4. 右侧髋臼外缘点 - 右侧髋臼上外侧骨皮质边缘的最外点
5. 耻骨联合点 - 左右耻骨在骨盆正中间连接处的上缘中点
6. 左侧髋臼荷重面内侧点 - 左侧髋臼顶部承重关节面的最内侧点
7. 左侧髋臼荷重面外侧点 - 左侧髋臼顶部承重关节面的最外侧点
8. 右侧髋臼荷重面内侧点 - 右侧髋臼顶部承重关节面的最内侧点
9. 右侧髋臼荷重面外侧点 - 右侧髋臼顶部承重关节面的最外侧点

请以JSON格式回复，每个点的坐标用归一化值（x和y都在0到1之间，以图像左上角为原点）：
{
  "keypoints": {
    "1_左股骨头中心": [x, y],
    "2_右股骨头中心": [x, y],
    "3_左髋臼外缘": [x, y],
    "4_右髋臼外缘": [x, y],
    "5_耻骨联合": [x, y],
    "6_左荷重面内侧": [x, y],
    "7_左荷重面外侧": [x, y],
    "8_右荷重面内侧": [x, y],
    "9_右荷重面外侧": [x, y]
  }
}

只回复JSON，不要有其他文字。"""


def encode_image(image_path, max_size=1024):
    """读X光图并转为base64，同时缩小到大模型能处理的尺寸"""
    img = Image.open(image_path).convert('RGB')
    # 等比缩放到最长边不超过max_size，减少API传输量
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def call_api(client, model, image_path):
    """调用API，返回解析后的关键点坐标"""
    img_b64 = encode_image(image_path)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        }
                    },
                    {"type": "text", "text": KEYPOINT_PROMPT}
                ]
            }]
        )
        text = response.content[0].text
    except Exception as e:
        print(f"    API调用失败: {e}")
        return None

    # 从回复中提取JSON
    try:
        # 尝试直接解析
        data = json.loads(text)
        kps = data['keypoints']
        # 按1-9排序
        coords = []
        for i in range(1, 10):
            for k, v in kps.items():
                if k.startswith(str(i)):
                    coords.append(v)
                    break
        return np.array(coords, dtype=np.float32)  # [9, 2]
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # 尝试用正则从文本里找JSON
        import re
        match = re.search(r'\{.*"keypoints".*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                kps = data['keypoints']
                coords = []
                for i in range(1, 10):
                    for k, v in kps.items():
                        if k.startswith(str(i)):
                            coords.append(v)
                            break
                return np.array(coords, dtype=np.float32)
            except:
                pass
        print(f"    解析失败，原始回复: {text[:200]}...")
        return None


def evaluate_api(args):
    device_info = f"{args.model}"
    print(f"模型: {device_info}")
    print(f"数据: {args.data_dir}")
    print(f"评估样本数: {args.num_samples}\n")

    # 收集有标注的图片
    all_files = sorted([
        f for f in os.listdir(args.data_dir)
        if f.endswith('.jpg') and os.path.exists(os.path.join(args.data_dir, f.replace('.jpg', '.json')))
    ])
    # 取子集（API调一次要钱，先测少量）
    import random
    random.seed(42)
    if args.num_samples < len(all_files):
        sample_files = random.sample(all_files, args.num_samples)
    else:
        sample_files = all_files

    print(f"从 {len(all_files)} 张中抽样 {len(sample_files)} 张\n")

    client = anthropic.Anthropic(
        base_url=os.environ.get('ANTHROPIC_BASE_URL'),
        api_key=os.environ.get('ANTHROPIC_AUTH_TOKEN'),
    )
    model = os.environ.get('ANTHROPIC_MODEL', 'deepseek-v4-pro[1m]')

    all_errors = []      # [N, 9]
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
        pred_kps = call_api(client, model, img_path)

        if pred_kps is not None and pred_kps.shape == (9, 2):
            dist = np.sqrt(((pred_kps - gt_kps) ** 2).sum(axis=1))  # [9]
            mae = dist.mean()
            all_errors.append(dist)
            successful += 1
            print(f"MAE={mae:.4f} ({mae*512:.0f}px)")
        else:
            failed += 1
            print("失败")
            all_errors.append(np.full(9, np.nan))

        # API限速
        if i < len(sample_files):
            time.sleep(args.delay)

    all_errors = np.array(all_errors)  # [N, 9]

    # 汇报
    valid = all_errors[~np.isnan(all_errors).any(axis=1)]
    if len(valid) == 0:
        print("\n所有样本均失败，无法评估")
        return

    pp_mae = np.nanmean(valid, axis=0)
    overall_mae = np.nanmean(valid)
    pck_05 = (valid < 0.05).mean()
    pck_10 = (valid < 0.10).mean()

    print(f"\n{'='*55}")
    print(f"  API零样本评估: {device_info}")
    print(f"  成功: {successful}/{len(sample_files)}, 失败: {failed}")
    print(f"{'='*55}")
    print(f'  {"点":16s}  {"MAE":>8s}')
    print(f'  {"-"*30}')
    names = ['左股骨头','右股骨头','左髋臼外缘','右髋臼外缘','耻骨联合','左荷重面内','左荷重面外','右荷重面内','右荷重面外']
    for i, n in enumerate(names):
        print(f'  点{i+1} {n:12s} {pp_mae[i]:>8.4f}')
    print(f'  {"-"*30}')
    print(f'  {"整体MAE":16s} {overall_mae:>8.4f} (≈{overall_mae*512:.0f}px)')
    print(f'  PCK@0.05 = {pck_05:.1%}')
    print(f'  PCK@0.10 = {pck_10:.1%}')
    print(f'{"="*55}')

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'model': device_info,
        'successful': successful,
        'failed': failed,
        'per_point_mae': pp_mae.tolist(),
        'overall_mae': float(overall_mae),
        'pck_05': float(pck_05),
        'pck_10': float(pck_10),
    }
    with open(os.path.join(args.output_dir, 'api_zero_shot_result.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'结果已保存: {args.output_dir}/api_zero_shot_result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='API零样本关键点检测评估')
    parser.add_argument('--data_dir', default='../shared_data/data/raw_images')
    parser.add_argument('--output_dir', default='outputs/api_eval')
    parser.add_argument('--model', default=None, help='覆盖环境变量中的模型')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--delay', type=float, default=2.0, help='API调用间隔(秒)')
    args = parser.parse_args()
    evaluate_api(args)
