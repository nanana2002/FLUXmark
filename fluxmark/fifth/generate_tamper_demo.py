#!/usr/bin/env python3
"""
生成指定 wm_id 的篡改检测示例图到 result/pic/tamper_img/

用法:
  python3 generate_tamper_demo.py [wm_id] [attack_name]

示例:
  python3 generate_tamper_demo.py wm_001 sdxl_style_sketch   # 只生成 sdxl_style_sketch
  python3 generate_tamper_demo.py wm_001                     # 遍历所有攻击类型
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# 读取配置
with open('config.json', 'r') as f:
    config = json.load(f)

img_id = sys.argv[1] if len(sys.argv) > 1 else 'wm_001'
target_attack = sys.argv[2] if len(sys.argv) > 2 else None  # None 表示遍历所有攻击

output_base_dir = config['output_base_dir']
result_dir = os.path.join(output_base_dir, 'result')
pic_result_dir = os.path.join(result_dir, 'pic', 'tamper_img')
os.makedirs(pic_result_dir, exist_ok=True)

watermark_extra_dir = os.path.join(output_base_dir, 'watermark_extra')
attack_dir = os.path.join(output_base_dir, 'pic', 'attack_watermarked_img')
watermarked_dir = os.path.join(output_base_dir, 'pic', 'watermarked_img')

# 读取提取摘要获取攻击类型列表
extraction_summary_path = os.path.join(watermark_extra_dir, 'extraction_summary.json')
if not os.path.exists(extraction_summary_path):
    print("❌ 未找到 extraction_summary.json，请先运行 extract_watermarks.py")
    exit(1)

with open(extraction_summary_path, 'r') as f:
    extraction_summary = json.load(f)
attack_configs = extraction_summary.get('attack_types', [])

# 辅助函数：生成可视化图
def visualize_tamper_heatmap(img_orig, img_attacked, S_orig, S_tamp, true_mask,
                              save_path, attack_name, metrics=None):
    img_size = img_orig.size if hasattr(img_orig, 'size') else (512, 512)
    diff = S_orig - S_tamp
    heatmap_8x8 = np.abs(diff)

    # 平滑热力图
    smooth_heatmap = cv2.resize(heatmap_8x8, img_size, interpolation=cv2.INTER_CUBIC)
    smooth_heatmap = cv2.GaussianBlur(smooth_heatmap, (21, 21), 0)

    threshold = np.mean(heatmap_8x8) + 1.5 * np.std(heatmap_8x8)
    threshold = max(threshold, 0.04)
    raw_mask = smooth_heatmap > threshold
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title('Original Watermarked')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_attacked)
    axes[0, 1].set_title(f'Attacked: {attack_name}')
    axes[0, 1].axis('off')

    im1 = axes[0, 2].imshow(heatmap_8x8, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('8x8 Raw Heatmap')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

    if true_mask is not None:
        axes[0, 3].imshow(true_mask, cmap='Reds', interpolation='nearest')
        axes[0, 3].set_title('True Tampered Regions')
    else:
        axes[0, 3].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center')
        axes[0, 3].set_title('True Mask (N/A)')
    axes[0, 3].axis('off')

    im2 = axes[1, 0].imshow(smooth_heatmap, cmap='hot', interpolation='bilinear')
    axes[1, 0].set_title(f'High-Res Smooth Heatmap\n({img_size[0]}x{img_size[1]})')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    img_attacked_resized = img_attacked.resize(img_size, Image.LANCZOS)
    axes[1, 1].imshow(img_attacked_resized, alpha=0.6)
    axes[1, 1].imshow(smooth_heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[1, 1].set_title('Heatmap Overlay on Image')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cleaned_mask, cmap='Blues', interpolation='nearest')
    axes[1, 2].set_title('Detected Regions (High-Res)')
    axes[1, 2].axis('off')

    comparison = np.zeros((*cleaned_mask.shape, 3))
    if true_mask is not None:
        true_mask_hr = cv2.resize(true_mask.astype(np.uint8), img_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        tp = np.logical_and(cleaned_mask, true_mask_hr)
        fp = np.logical_and(cleaned_mask, ~true_mask_hr)
        fn = np.logical_and(~cleaned_mask, true_mask_hr)
        comparison[tp] = [0, 1, 0]
        comparison[fp] = [1, 1, 0]
        comparison[fn] = [1, 0, 0]
        title = 'Detection Comparison (High-Res)\n(G=TP, Y=FP, R=FN)'
    else:
        title = 'Predicted Mask (High-Res)'
    axes[1, 3].imshow(comparison, interpolation='nearest')
    if metrics:
        title += f"\nIoU={metrics.get('iou', 0):.3f}, F1={metrics.get('f1', 0):.3f}"
    axes[1, 3].set_title(title)
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 额外生成一张高分辨率叠加图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_attacked)
    axes[0].set_title('Attacked Image')
    axes[0].axis('off')

    axes[1].imshow(smooth_heatmap, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f'Smooth Heatmap ({img_size[0]}x{img_size[1]})')
    axes[1].axis('off')
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_attacked_resized, alpha=0.5)
    axes[2].imshow(smooth_heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    save_path_hr = save_path.replace('.png', '_highres.png')
    plt.savefig(save_path_hr, dpi=200, bbox_inches='tight')
    plt.close()


# 遍历攻击类型，为指定 img_id 生成示例图
generated = 0
for attack_name in attack_configs:
    # 如果命令行指定了攻击名，则只处理该攻击；否则处理所有攻击
    if target_attack is not None and attack_name != target_attack:
        continue

    S_orig_path = os.path.join(watermark_extra_dir, f"{img_id}_S_orig.npy")
    S_tamp_path = os.path.join(watermark_extra_dir, attack_name, f"{img_id}_{attack_name}_S_tamp.npy")
    if not os.path.exists(S_tamp_path):
        S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")

    if not os.path.exists(S_orig_path) or not os.path.exists(S_tamp_path):
        print(f"⚠️  跳过 {attack_name}: 缺少签名文件")
        continue

    S_orig = np.load(S_orig_path)
    S_tamp = np.load(S_tamp_path)

    # 加载原图和攻击图
    orig_img_path = os.path.join(watermarked_dir, f"{img_id}.png")
    if not os.path.exists(orig_img_path):
        orig_img_path = os.path.join(watermarked_dir, f"{img_id}.jpg")

    attacked_img_path = os.path.join(attack_dir, attack_name, f"{img_id}.png")
    if not os.path.exists(attacked_img_path):
        attacked_img_path = os.path.join(attack_dir, attack_name, f"{img_id}.jpg")

    if not os.path.exists(orig_img_path) or not os.path.exists(attacked_img_path):
        print(f"⚠️  跳过 {attack_name}: 缺少图像文件")
        continue

    # 加载真值 mask（如果有）
    true_mask_path = os.path.join(attack_dir, attack_name, f"{img_id}_mask.npy")
    true_mask = np.load(true_mask_path) if os.path.exists(true_mask_path) else None

    # 简单计算 IoU/F1 用于标题显示
    metrics = None
    if true_mask is not None:
        diff = S_orig - S_tamp
        heatmap = np.abs(diff)
        threshold = np.mean(heatmap) + 0.5 * np.std(heatmap)
        pred_mask = (heatmap > threshold).astype(int)

        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        iou = float(intersection / (union + 1e-8))

        tp = np.logical_and(pred_mask, true_mask).sum()
        fp = np.logical_and(pred_mask, ~true_mask.astype(bool)).sum()
        fn = np.logical_and(~pred_mask.astype(bool), true_mask).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        metrics = {'iou': iou, 'f1': f1}

    orig_img = Image.open(orig_img_path)
    attacked_img = Image.open(attacked_img_path)
    save_path = os.path.join(pic_result_dir, f"{attack_name}_{img_id}_heatmap.png")

    visualize_tamper_heatmap(orig_img, attacked_img, S_orig, S_tamp,
                              true_mask, save_path, attack_name, metrics)
    print(f"✅ 已生成: {save_path}")
    generated += 1

print(f"\n🎉 共生成 {generated} 张篡改检测示例图到 {pic_result_dir}")
