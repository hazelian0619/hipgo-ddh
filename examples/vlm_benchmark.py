#!/usr/bin/env python
"""VLM零样本关键点检测对比（LLaVA / MiniCPM-V / GPT-4o）"""
import os, json, base64, re, time, argparse, numpy as np
from io import BytesIO
from PIL import Image

PROMPT = """你是骨科放射科医生。看这张骨盆正位X光片。
给出9个关键点的归一化坐标(x,y)，范围0-1(左上角原点):
1左股骨头圆心 2右股骨头圆心 3左髋臼外缘 4右髋臼外缘
5耻骨联合上缘中点 6左髋臼荷重面内侧 7左髋臼荷重面外侧
8右髋臼荷重面内侧 9右髋臼荷重面外侧
只回复JSON: {"1":[x,y],"2":[x,y],...}"""


def encode_image(path, max_size=1024):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    r = max_size / max(w, h)
    img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def parse_kps(text):
    m = re.search(r'\{[^{}]*"1"[^{}]*\}', text, re.DOTALL) or re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        coords = []
        for i in range(1, 10):
            k = str(i)
            if k in d and len(d[k]) == 2:
                coords.append([float(d[k][0]), float(d[k][1])])
            else:
                coords.append([0.0, 0.0])
        return np.array(coords, dtype=np.float32)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def eval_ollama(model, image_dir, num_samples=10):
    """本地Ollama视觉模型评估"""
    try:
        import ollama
    except ImportError:
        print('请安装: pip install ollama')
        return

    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(image_dir, f.replace('.jpg', '.json')))])
    import random
    random.seed(42)
    samples = random.sample(files, min(num_samples, len(files)))

    errors = []
    for fname in samples:
        img_path = os.path.join(image_dir, fname)
        with open(img_path.replace('.jpg', '.json')) as f:
            ann = json.load(f)
        W, H = ann['imageWidth'], ann['imageHeight']
        ss = sorted(ann['shapes'], key=lambda x: int(x['label']))
        gt = np.array([[s['points'][0][0] / W, s['points'][0][1] / H] for s in ss], dtype=np.float32)

        r = ollama.chat(model=model, messages=[{'role': 'user', 'content': PROMPT, 'images': [encode_image(img_path)]}],
                        options={'temperature': 0.1, 'num_predict': 512})
        pred = parse_kps(r['message']['content'])
        if pred is not None and pred.shape == (9, 2):
            errors.append(np.sqrt(((pred - gt) ** 2).sum(axis=1)).mean())
            print(f'  {fname}: MAE={errors[-1]:.4f}')
        else:
            print(f'  {fname}: 解析失败')
        time.sleep(1)

    if errors:
        print(f'\n整体MAE: {np.mean(errors):.4f} (≈{np.mean(errors)*512:.0f}px)')


def eval_openai(api_key, image_dir, num_samples=10):
    """OpenAI API视觉模型评估"""
    try:
        from openai import OpenAI
    except ImportError:
        print('请安装: pip install openai')
        return

    client = OpenAI(api_key=api_key)
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(image_dir, f.replace('.jpg', '.json')))])
    import random
    random.seed(42)
    samples = random.sample(files, min(num_samples, len(files)))

    errors = []
    for fname in samples:
        b64 = encode_image(os.path.join(image_dir, fname))
        with open(os.path.join(image_dir, fname.replace('.jpg', '.json'))) as f:
            ann = json.load(f)
        W, H = ann['imageWidth'], ann['imageHeight']
        ss = sorted(ann['shapes'], key=lambda x: int(x['label']))
        gt = np.array([[s['points'][0][0] / W, s['points'][0][1] / H] for s in ss], dtype=np.float32)

        r = client.chat.completions.create(
            model='gpt-4o', max_tokens=512, temperature=0.1,
            messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                                                    {'type': 'text', 'text': PROMPT}]}])
        pred = parse_kps(r.choices[0].message.content)
        if pred is not None and pred.shape == (9, 2):
            errors.append(np.sqrt(((pred - gt) ** 2).sum(axis=1)).mean())
            print(f'  {fname}: MAE={errors[-1]:.4f}')
        else:
            print(f'  {fname}: 解析失败')
        time.sleep(0.5)

    if errors:
        print(f'\n整体MAE: {np.mean(errors):.4f} (≈{np.mean(errors)*512:.0f}px)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['ollama', 'openai'], default='ollama')
    ap.add_argument('--model', default='llava:13b')
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--api_key', default=None)
    ap.add_argument('--num_samples', type=int, default=10)
    args = ap.parse_args()

    if args.backend == 'ollama':
        eval_ollama(args.model, args.image_dir, args.num_samples)
    else:
        eval_openai(args.api_key or os.environ.get('OPENAI_API_KEY', ''), args.image_dir, args.num_samples)
