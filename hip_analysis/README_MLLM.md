# 骨盆X光片多模态医学大语言模型系统

基于LLaVA-Med实现的骨盆X光片分析与医学报告生成系统，支持自动生成专业医学报告，辅助髋关节发育不良(DDH)的诊断和治疗。

## 功能特点

- **多模态视觉理解**：基于LLaVA-Med模型，能够直接理解X光片图像内容
- **自动计算临床角度**：支持CE角、Sharp角和Tönnis角等多种临床角度的自动计算
- **结构化数据整合**：将图像特征与结构化角度参数融合，实现更精准的分析
- **专业医学报告生成**：自动生成包含形态学描述、测量结果、诊断意见和治疗建议的专业医学报告
- **低资源设备支持**：提供低资源模式，适配不同硬件环境

## 系统要求

- Python 3.9+
- PyTorch 2.0+
- 推荐使用GPU或Apple Silicon (MPS)加速

## 安装步骤

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 下载LLaVA-Med模型

```bash
python download_llava_med.py
```

或使用自定义模型路径：

```bash
python download_llava_med.py --model liuhaotian/llava-med-v1.0-7b --output_dir models/llava_med
```

## 使用方法

### 命令行运行

```bash
# 基本用法
python mllm_medical_report.py --image path/to/xray.jpg

# 提供关键点JSON文件
python mllm_medical_report.py --image path/to/xray.jpg --keypoints path/to/keypoints.json

# 指定设备类型
python mllm_medical_report.py --image path/to/xray.jpg --device cuda/cpu/mps

# 低资源模式
python mllm_medical_report.py --image path/to/xray.jpg --low_resource
```

### 使用运行脚本

提供了更便捷的运行脚本：

```bash
# 给脚本添加执行权限
chmod +x run_mllm_report.sh

# 基本用法
./run_mllm_report.sh --image path/to/xray.jpg

# 完整参数
./run_mllm_report.sh --image path/to/xray.jpg --keypoints path/to/keypoints.json --model models/llava_med --output reports --device cuda --low-resource
```

## 代码示例

```python
from mllm_medical_report import HipMLLM

# 初始化模型
mllm = HipMLLM(model_path="models/llava_med", device="cuda")

# 生成报告
report = mllm.generate_report(
    image_path="data/samples/xray_001.jpg",
    keypoints_path="data/json_annotations/xray_001.json"
)

print(report)
```

## 系统架构

该系统采用以下架构：

1. **输入处理层**：处理X光片图像和关键点数据
2. **特征提取层**：使用LLaVA-Med视觉编码器提取图像特征  
3. **结构化参数融合**：将图像特征与临床角度数据结合
4. **报告生成层**：基于多模态融合特征生成专业医学报告

## 常见问题

### Q: 系统报错"CUDA out of memory"
A: 尝试使用低资源模式运行：`--low_resource`

### Q: 如何更换其他模型版本？
A: 使用`--model`参数指定其他版本，如`liuhaotian/llava-med-v1.0-13b`

### Q: 模型下载失败怎么办？
A: 请检查网络连接，或手动下载模型到`models/llava_med`目录

## 致谢

- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)：医学多模态大语言模型
- [MONAI](https://monai.io/)：医学影像深度学习框架
- [Hugging Face](https://huggingface.co/)：提供模型托管和API

## 许可证

MIT License 