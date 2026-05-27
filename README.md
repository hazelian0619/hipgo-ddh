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

## 项目结构

```
hipgo/              # 核心包
├── dataset.py      # 数据加载 + 增强（等比padding, CLAHE, 加权loss）
├── transforms.py   # 图像变换
├── angles.py       # 角度计算（CE/Sharp/Tönnis）
└── models/
    ├── backbone.py      # ResNet50 + FPN 特征提取器
    └── cnn_keypoint.py  # 纯CNN关键点检测（最佳架构）

scripts/            # 入口脚本
├── train.py        # 训练
├── evaluate.py     # 逐点评估（MAE/PCK）
├── cross_validate.py   # 5折交叉验证
├── active_learn.py     # 主动学习TTA不确定性采样
├── test_expert.py      # 专家测试集完整管线
└── vlm_benchmark.py    # VLM零样本对比

hip_model/          # 实验代码（含CNN-GAT消融、HRNet等）
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练
python scripts/train.py --data_dir path/to/annotated/images --epochs 30

# 评估
python scripts/evaluate.py --model_path outputs/best_model.pth --data_dir path/to/images

# 5折交叉验证
python scripts/cross_validate.py --data_dir path/to/images --epochs 30

# 主动学习选图
python scripts/active_learn.py --model_path outputs/best_model.pth --top_k 20

# 专家测试集管线
python scripts/test_expert.py
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
