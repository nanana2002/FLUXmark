#!/usr/bin/env python3
"""
验证 OrthoFlow 的"绝对盲提取"可行性：
不用 S_orig 做差分，直接看 S_tamp 的绝对值能否定位篡改。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

base_dir = "watermark_extra"
img_id = "wm_000"

# 加载 S_orig
S_orig = np.load(os.path.join(base_dir, f"{img_id}_S_orig.npy"))
print(f"S_orig shape: {S_orig.shape}, mean={S_orig.mean():.4f}, std={S_orig.std():.4f}")

# 测试的攻击类型
test_attacks = [
    ("black_block_center", "黑块中心"),
    ("black_block_random", "黑块随机"),
    ("copy_move", "复制移动"),
    ("splicing", "拼接"),
    ("fluxfill_center", "FluxFill中心"),
    ("sdedit_0.3", "SDEdit 0.3"),
    ("jpeg_50", "JPEG 50"),
    ("blur_2.0", "模糊 2.0"),
]

fig, axes = plt.subplots(len(test_attacks), 3, figsize=(12, 3 * len(test_attacks)))

for row_idx, (attack_name, label) in enumerate(test_attacks):
    S_tamp_path = os.path.join(base_dir, f"{img_id}_{attack_name}_S_tamp.npy")
    if not os.path.exists(S_tamp_path):
        print(f"⚠️  跳过 {attack_name}: 文件不存在")
        continue
    
    S_tamp = np.load(S_tamp_path)
    S_diff = S_tamp - S_orig
    
    print(f"\n🔬 {label} ({attack_name}):")
    print(f"   S_tamp  mean={S_tamp.mean():.4f}  std={S_tamp.std():.4f}  min={S_tamp.min():.4f}  max={S_tamp.max():.4f}")
    print(f"   S_diff  mean={S_diff.mean():.4f}  std={S_diff.std():.4f}  min={S_diff.min():.4f}  max={S_diff.max():.4f}")
    
    # 尝试找篡改区域：S_tamp 中显著低于均值的区域
    # 和 S_diff 中显著低于 0 的区域对比
    threshold_tamp = S_tamp.mean() - 2 * S_tamp.std()
    threshold_diff = -2 * S_diff.std()
    
    mask_tamp = S_tamp < threshold_tamp
    mask_diff = S_diff < threshold_diff
    
    overlap = np.logical_and(mask_tamp, mask_diff).sum()
    union = np.logical_or(mask_tamp, mask_diff).sum()
    iou = overlap / union if union > 0 else 0
    
    print(f"   阈值法 IoU (S_tamp vs S_diff): {iou:.3f}")
    print(f"   mask_tamp 覆盖率: {mask_tamp.sum()}/{mask_tamp.size} = {mask_tamp.mean():.3f}")
    print(f"   mask_diff  覆盖率: {mask_diff.sum()}/{mask_diff.size} = {mask_diff.mean():.3f}")
    
    ax1 = axes[row_idx, 0]
    ax2 = axes[row_idx, 1]
    ax3 = axes[row_idx, 2]
    
    im1 = ax1.imshow(S_orig, cmap='viridis', vmin=-0.3, vmax=0.5)
    ax1.set_title(f"S_orig\nmean={S_orig.mean():.3f}", fontsize=9)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    im2 = ax2.imshow(S_tamp, cmap='viridis', vmin=-0.3, vmax=0.5)
    ax2.set_title(f"S_tamp ({label})\nmean={S_tamp.mean():.3f}", fontsize=9)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    im3 = ax3.imshow(S_diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax3.set_title(f"S_diff = S_tamp - S_orig\nmean={S_diff.mean():.3f}", fontsize=9)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

plt.suptitle("OrthoFlow Blind Extraction Test: Can S_tamp alone locate tampering?", fontsize=12)
plt.tight_layout()
out_path = "orthoflow_blind_extraction_test.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✅ 热力图已保存: {out_path}")
