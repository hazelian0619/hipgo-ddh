# HipGo-DDH

用极少量骨盆X光标注数据训练CNN模型，精确检测9个解剖关键点，计算DDH（发育性髋关节发育不良）角度，辅助临床诊断。

## 实验结论

| 模型 | MAE ↓ | PCK@0.10 ↑ |
|---|---|---|
| **CNN + ImageNet预训练 (80张)** | **0.049 ≈ 25px** | **93.5%** |
| CNN + ImageNet预训练 (60张) | 0.064 ≈ 33px | 87.6% |
| CNN-GAT + 预训练 | 0.076 ≈ 39px | 68.3% |
| GPT-4o 零样本 | 0.127 ≈ 65px | 35.6% |
| LLaVA 13B 零样本 | 0.236 ≈ 121px | 10.0% |

**专家测试集诊断准确率: 90.3%（195张，CE<25°/Sharp>45°/Tönnis>10°）**

## 数据集

| 数据集 | 数量 | 标注 | 说明 |
|---|---|---|---|
| `data/train/` | 80张 | 9关键点JSON | 训练集（60张原始+20张主动学习） |
| `data/unlabeled/` | 200张 | 无 | 未标注池（主动学习候选） |
| `data/expert_test/` | 200张 | 医生诊断标签 | 独立测试集 |

来源：社交媒体公开的骨盆X光片，已去标识化。

## 项目结构

```
data/                   # 数据集
├── train/              # 80张已标注
├── unlabeled/          # 200张未标注
└── expert_test/        # 200张专家测试 (images/ + labels.json)

hipgo/                  # 核心包
├── dataset.py
├── transforms.py
├── angles.py           # 角度计算+诊断规则
├── models/
│   ├── backbone.py
│   └── cnn_keypoint.py
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── cross_validate.py
    ├── active_learn.py
    └── test_expert.py

examples/
└── vlm_benchmark.py
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练
python hipgo/scripts/train.py --data_dir path/to/annotated/images --epochs 30

# 评估
python hipgo/scripts/evaluate.py --model_path outputs/best_model.pth --data_dir path/to/images

# 5折交叉验证
python hipgo/scripts/cross_validate.py --data_dir path/to/images --epochs 30

# 主动学习选图
python hipgo/scripts/active_learn.py --model_path outputs/best_model.pth --unlabeled_dir path/to/images --top_k 20

# 专家测试集管线
python hipgo/scripts/test_expert.py --model_path outputs/best_model.pth --expert_dir path/to/images --label_dir path/to/labels
```

## 数据格式

LabelMe JSON格式，9个关键点标签为1-9，坐标像素值：

```json
{
  "shapes": [
    {"label": "1", "points": [[x1, y1]]},
    {"label": "2", "points": [[x2, y2]]},
    ...
    {"label": "9", "points": [[x9, y9]]}
  ]
}
```

## 关键点定义

| 编号 | 名称 | 解剖位置 |
|---|---|---|
| 1 | 左股骨头中心 | 左侧股骨头圆形轮廓的圆心 |
| 2 | 右股骨头中心 | 右侧股骨头圆形轮廓的圆心 |
| 3 | 左髋臼外缘 | 左髋臼上外侧骨皮质最外侧点 |
| 4 | 右髋臼外缘 | 右髋臼上外侧骨皮质最外侧点 |
| 5 | 耻骨联合 | 骨盆正中耻骨联合上缘中点 |
| 6 | 左荷重面内侧 | 左髋臼承重关节面最内侧 |
| 7 | 左荷重面外侧 | 左髋臼承重关节面最外侧 |
| 8 | 右荷重面内侧 | 右髋臼承重关节面最内侧 |
| 9 | 右荷重面外侧 | 右髋臼承重关节面最外侧 |

## DDH角度

| 角度 | 参考线 | 正常 | DDH |
|---|---|---|---|
| CE角 | 垂直线（⊥股骨头连线）| >25° | <20° |
| Sharp角 | 水平线（股骨头连线）| <45° | >50° |
| Tönnis角 | 垂直线（⊥股骨头连线）| <10° | >15° |

## License

MIT
