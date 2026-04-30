#!/usr/bin/env python3
"""
验证自监督背景估计：不用 S_orig，直接从 S_tamp 推断 baseline 来定位篡改。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter

base_dir = "watermark_extra"
img_id = "wm_000"

S_orig = np.load(os.path.join(base_dir, f"{img_id}_S_orig.npy"))

# 多种自监督 baseline 估计方法
def estimate_baseline_median(S, kernel_size=5):
    """中值滤波：假设邻居大多是正常的"""
    return median_filter(S, size=kernel_size)

def estimate_baseline_gaussian(S, sigma=2):
    """高斯滤波：平滑估计背景"""
    return gaussian_filter(S, sigma=sigma)

def estimate_baseline_global_median(S):
    """全局中位数：假设 >50% 区域正常"""
    return np.full_like(S, np.median(S))

def estimate_baseline_global_mean(S):
    """全局均值（对异常不 robust）"""
    return np.full_like(S, np.mean(S))

def estimate_baseline_percentile(S, p=75):
    """百分位数：假设正常区域占多数"""
    return np.full_like(S, np.percentile(S, p))

# 测试的攻击
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

methods = [
    ("S_orig (Oracle)", lambda S: S_orig, None),
    ("Median filter(5)", estimate_baseline_median, {"kernel_size": 5}),
    ("Median filter(7)", estimate_baseline_median, {"kernel_size": 7}),
    ("Gaussian(σ=2)", estimate_baseline_gaussian, {"sigma": 2}),
    ("Global median", estimate_baseline_global_median, None),
    ("Global 75th percentile", estimate_baseline_percentile, {"p": 75}),
]

print("="*80)
print("自监督背景估计效果对比（IoU 越高越好，对比 S_diff = S_tamp - S_orig）")
print("="*80)

for attack_name, label in test_attacks:
    S_tamp_path = os.path.join(base_dir, f"{img_id}_{attack_name}_S_tamp.npy")
    if not os.path.exists(S_tamp_path):
        continue
    
    S_tamp = np.load(S_tamp_path)
    S_diff = S_tamp - S_orig
    
    # 用 S_diff 的阈值作为"真实篡改区域"的参考
    threshold_ref = -2 * S_diff.std()
    mask_ref = S_diff < threshold_ref
    
    print(f"\n🔬 {label} ({attack_name}):")
    print(f"   S_diff std={S_diff.std():.4f}, threshold={threshold_ref:.4f}")
    print(f"   参考 mask 覆盖率: {mask_ref.mean():.3f}")
    
    for method_name, method_fn, kwargs in methods:
        if kwargs:
            baseline = method_fn(S_tamp, **kwargs)
        else:
            baseline = method_fn(S_tamp)
        
        S_residual = S_tamp - baseline
        
        # 自适应阈值：用 residual 的 std
        threshold = -2 * S_residual.std()
        mask_pred = S_residual < threshold
        
        # 计算 IoU
        overlap = np.logical_and(mask_pred, mask_ref).sum()
        union = np.logical_or(mask_pred, mask_ref).sum()
        iou = overlap / union if union > 0 else 0
        
        # 覆盖率
        pred_coverage = mask_pred.mean()
        
        print(f"   {method_name:25s}  IoU={iou:.3f}  coverage={pred_coverage:.3f}")

# ==========================================
# 可视化最佳方法 vs S_orig
# ==========================================
best_method = ("Median filter(5)", estimate_baseline_median, {"kernel_size": 5})

fig, axes = plt.subplots(len(test_attacks), 4, figsize=(14, 3 * len(test_attacks)))

for row_idx, (attack_name, label) in enumerate(test_attacks):
    S_tamp_path = os.path.join(base_dir, f"{img_id}_{attack_name}_S_tamp.npy")
    if not os.path.exists(S_tamp_path):
        for c in range(4):
            axes[row_idx, c].axis('off')
        continue
    
    S_tamp = np.load(S_tamp_path)
    S_diff = S_tamp - S_orig
    
    # 最佳自监督方法
    baseline = estimate_baseline_median(S_tamp, kernel_size=5)
    S_residual = S_tamp - baseline
    
    ax1 = axes[row_idx, 0]
    ax2 = axes[row_idx, 1]
    ax3 = axes[row_idx, 2]
    ax4 = axes[row_idx, 3]
    
    im1 = ax1.imshow(S_orig, cmap='viridis', vmin=-0.3, vmax=0.5)
    ax1.set_title(f"S_orig\nmean={S_orig.mean():.3f}", fontsize=9)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    im2 = ax2.imshow(S_tamp, cmap='viridis', vmin=-0.3, vmax=0.5)
    ax2.set_title(f"S_tamp ({label})\nmean={S_tamp.mean():.3f}", fontsize=9)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    im3 = ax3.imshow(S_diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax3.set_title(f"S_diff (Oracle)\nstd={S_diff.std():.3f}", fontsize=9)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    im4 = ax4.imshow(S_residual, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax4.set_title(f"S_residual (Self-sup)\nstd={S_residual.std():.3f}", fontsize=9)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

plt.suptitle("Self-Supervised Baseline Estimation (Median filter 5x5)", fontsize=12)
plt.tight_layout()
out_path = "self_supervised_baseline_test.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✅ 热力图已保存: {out_path}")
