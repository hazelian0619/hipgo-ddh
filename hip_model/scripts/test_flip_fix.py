#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 HorizontalFlip 关键点语义互换修复是否正确。

检查逻辑：
  - 翻转前，点1（左股骨头）应该在图像左侧（x < 0.5）
  - 翻转后，点1（左股骨头）应该仍然在图像左侧（x < 0.5）
  - 翻转前，点1.x + 点2.x ≈ 1.0（左右对称）
  - 翻转后同样成立
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset import _swap_keypoints_after_flip, FLIP_PAIRS

# --- 构造一个已知的标注样本 ---
# 模拟一张正常骨盆图的归一化关键点（左侧x < 0.5，右侧x > 0.5）
mock_keypoints = np.array([
    [0.245, 0.554],  # 点1：左股骨头中心    x < 0.5 ✓
    [0.756, 0.553],  # 点2：右股骨头中心    x > 0.5 ✓
    [0.216, 0.482],  # 点3：左髋臼外缘      x < 0.5 ✓
    [0.781, 0.482],  # 点4：右髋臼外缘      x > 0.5 ✓
    [0.499, 0.652],  # 点5：耻骨联合        x ≈ 0.5 ✓
    [0.325, 0.578],  # 点6：左荷重面内侧    x < 0.5 ✓
    [0.294, 0.500],  # 点7：左荷重面外侧    x < 0.5 ✓
    [0.673, 0.576],  # 点8：右荷重面内侧    x > 0.5 ✓
    [0.700, 0.501],  # 点9：右荷重面外侧    x > 0.5 ✓
], dtype=np.float32)

print("=== 翻转前 ===")
print(f"点1（左股骨头）x = {mock_keypoints[0, 0]:.3f}  期望 < 0.5")
print(f"点2（右股骨头）x = {mock_keypoints[1, 0]:.3f}  期望 > 0.5")
print(f"点1.x + 点2.x = {mock_keypoints[0,0] + mock_keypoints[1,0]:.3f}  期望 ≈ 1.0")

# 模拟水平翻转：x → 1 - x
flipped = mock_keypoints.copy()
flipped[:, 0] = 1.0 - flipped[:, 0]

print("\n=== 翻转坐标后（未交换语义）===")
print(f"点1（左股骨头）x = {flipped[0, 0]:.3f}  ← 现在在右侧，语义错误！")
print(f"点2（右股骨头）x = {flipped[1, 0]:.3f}  ← 现在在左侧，语义错误！")

# 应用语义互换
corrected = _swap_keypoints_after_flip(flipped)

print("\n=== 翻转坐标后（已交换语义）===")
print(f"点1（左股骨头）x = {corrected[0, 0]:.3f}  期望 < 0.5")
print(f"点2（右股骨头）x = {corrected[1, 0]:.3f}  期望 > 0.5")
print(f"点1.x + 点2.x = {corrected[0,0] + corrected[1,0]:.3f}  期望 ≈ 1.0")

# --- 自动断言 ---
passed = True

if corrected[0, 0] >= 0.5:
    print("\n❌ FAIL：翻转后点1（左股骨头）应该仍在左侧（x < 0.5）")
    passed = False
else:
    print("\n✅ 点1 翻转后仍在左侧")

if corrected[1, 0] <= 0.5:
    print("❌ FAIL：翻转后点2（右股骨头）应该仍在右侧（x > 0.5）")
    passed = False
else:
    print("✅ 点2 翻转后仍在右侧")

# 检查所有互换对
for i, j in FLIP_PAIRS:
    # 翻转后第i点的坐标应该是原来第j点翻转后的坐标
    expected_x = 1.0 - mock_keypoints[j, 0]
    expected_y = mock_keypoints[j, 1]
    if not (abs(corrected[i, 0] - expected_x) < 1e-4 and abs(corrected[i, 1] - expected_y) < 1e-4):
        print(f"❌ FAIL：点对 ({i+1},{j+1}) 互换不正确")
        passed = False
    else:
        print(f"✅ 点对 ({i+1},{j+1}) 互换正确")

print(f"\n{'=== 全部通过 ===' if passed else '=== 存在问题，请检查 ==='}")
